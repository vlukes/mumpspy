import ctypes
from mpi4py import MPI
import numpy as nm
import re
from .mumps_lib_c_struc import (define_mumps_c_struc, c_pointer,
                                PMumpsComplex, PMumpsInt)


def load_library(libname):
    """Load the shared library in a system dependent way."""
    import sys

    if sys.platform.startswith('win'):  # Windows system
        from ctypes.util import find_library

        lib_fname = find_library(libname)
        if lib_fname is None:
            lib_fname = find_library('lib' + libname)

    else:  # Linux system
        lib_fname = 'lib' + libname + '.so'

    lib = ctypes.cdll.LoadLibrary(lib_fname)

    return lib


mumps_libs = {
    'dmumps': load_library('dmumps').dmumps_c,
    'zmumps': load_library('zmumps').zmumps_c,
}


def get_lib_version():
    """Determine the MUMPS library version."""
    aux_mumps_c_struc = define_mumps_c_struc()
    struct = aux_mumps_c_struc()
    struct.par = 1
    struct.sym = 0
    struct.comm_fortran = MPI.COMM_WORLD.py2f()
    struct.job = -1  # init package instance

    mumps_c = mumps_libs['dmumps']
    mumps_c.restype = None
    mumps_c.argtypes = [c_pointer(aux_mumps_c_struc)]
    mumps_c(ctypes.byref(struct))

    arr = nm.ctypeslib.as_array(struct.aux)
    idxs = nm.logical_and(arr >= ord('.'), arr <= ord('9'))
    s = (arr[idxs].tobytes()).decode('utf-8')
    vnums = re.findall(r'^.*(\d)\.(\d+)\.(\d+).*$', s)[-1]
    version = '.'.join(vnums)

    struct.job = -2  # terminate package instance
    struct.icntl[0] = -1  # suppress error messages
    struct.icntl[1] = -1  # suppress diagnostic messages
    struct.icntl[2] = -1  # suppress global info
    struct.icntl[3] = 0

    mumps_c(ctypes.byref(struct))

    return version


mumps_lib_version = get_lib_version()

mumps_c_struc = define_mumps_c_struc(mumps_lib_version)


class MumpsSolver(object):
    """MUMPS object."""

    def __init__(self, is_sym=False, mpi_comm=None,
                 system='real', silent=True, mem_relax=20):
        """
        Init the MUMUPS solver.

        Parameters
        ----------
        is_sym : bool
            Symmetric matrix?
        mpi_comm : MPI Communicator or None
            If None, use MPI.COMM_WORLD
        system : 'real' or 'complex'
            Use real or complex linear solver
        silent : bool
            If True, no MUMPS error, warning, and diagnostic messages
        mem_relax : int
            The percentage increase in the estimated working space
        """
        self.struct = None

        if system == 'real':
            self._mumps_c = mumps_libs['dmumps']
            self.dtype = nm.float64
        elif system == 'complex':
            self._mumps_c = mumps_libs['zmumps']
            self.dtype = nm.complex128

        self.mpi_comm = MPI.COMM_WORLD if mpi_comm is None else mpi_comm
        self._mumps_c.restype = None

        # init mumps library
        self._mumps_c.argtypes = [c_pointer(mumps_c_struc)]

        self.struct = mumps_c_struc()
        self.struct.par = 1
        self.struct.sym = 2 if is_sym else 0
        self.struct.n = 0
        self.struct.comm_fortran = self.mpi_comm.py2f()

        self._mumps_call(job=-1)  # init

        self.rank = self.mpi_comm.rank
        self._data = {}

        # be silent
        if silent:
            self.set_silent()

        self.struct.icntl[13] = mem_relax

    def set_silent(self):
        self.struct.icntl[0] = -1  # suppress error messages
        self.struct.icntl[1] = -1  # suppress diagnostic messages
        self.struct.icntl[2] = -1  # suppress global info
        self.struct.icntl[3] = 0

    def set_verbose(self):
        self.struct.icntl[0] = 6  # error messages
        self.struct.icntl[1] = 0  # diagnostic messages
        self.struct.icntl[2] = 6  # global info
        self.struct.icntl[3] = 2

    def __del__(self):
        """Finish MUMPS."""
        if self.struct is not None:
            self._mumps_call(job=-2)  # done

        self.struct = None

    def set_mtx(self, mtx, factorize=True):
        """
        Set the sparse matrix.

        Parameters
        ----------
        mtx : scipy sparse martix
            Sparse matrix in COO format
        """
        assert mtx.shape[0] == mtx.shape[1]

        rr = mtx.row + 1
        cc = mtx.col + 1
        data = mtx.data

        if self.struct.sym > 0:
            idxs = nm.where(cc >= rr)[0]  # upper triangular matrix
            rr, cc, data = rr[idxs], cc[idxs], data[idxs]

        self.set_rcd_mtx(rr, cc, data, mtx.shape[0], factorize)

    def set_rcd_mtx(self, ir, ic, data, n, factorize=True):
        """
        Set the matrix using row and column indices and a data vector.
        The matrix shape is determined by the maximal values of
        the row and column indices. The indices start with 1.

        Parameters
        ----------
        ir : array
            Row idicies
        ic : array
            Column idicies
        data : array
            Matrix entries
        n : int
            Matrix dimension
        """
        assert ir.shape[0] == ic.shape[0] == data.shape[0]

        ir = nm.asarray(ir, dtype=nm.int32)
        ic = nm.asarray(ic, dtype=nm.int32)
        data = nm.asarray(data, dtype=self.dtype)

        self._data.update(ir=ir, ic=ic, vals=data, factorized=factorize)
        self.struct.n = n
        self.struct.nz = ir.shape[0]
        if hasattr(self.struct, 'nnz'):
            self.struct.nnz = ir.shape[0]
        self.struct.irn = ir.ctypes.data_as(PMumpsInt)
        self.struct.jcn = ic.ctypes.data_as(PMumpsInt)
        self.struct.a = data.ctypes.data_as(PMumpsComplex)

        if factorize:
            self._mumps_call(4)

    def set_rhs(self, rhs):
        """Set the right hand side of the linear system."""
        rhs = nm.asarray(rhs, order='F', dtype=self.dtype)

        n = self.struct.n
        if rhs.shape[0] != n:
            msg = ('Wrong size of the right hand side vector/matrix! '
                   f'(rhs: {rhs.shape}, mtx: ({n}, {n}))')
            raise ValueError(msg)

        self._data.update(rhs=rhs)
        self.struct.rhs = rhs.ctypes.data_as(PMumpsComplex)
        self.struct.lrhs = rhs.shape[0]

        if len(rhs.shape) == 1:
            self.struct.nrhs = 1
        elif len(rhs.shape) == 2:
            self.struct.nrhs = rhs.shape[-1]
        else:
            raise ValueError('The right hand side must be a vector/matrix!')


    def __call__(self, job):
        """Set the job and call MUMPS."""
        if 'vals' not in self._data:
            raise ValueError('The matrix is not set!')

        if job in [3, 5, 6] and 'rhs' not in self._data:
            raise ValueError('The right hand side vector is not set!')

        self._mumps_call(job)

    def schur_complement(self, schur_list):
        """Get the Schur matrix and the condensed right-hand side vector.

        Parameters
        ----------
        schur_list : array
            List of Schur DOFs (indexing starts with 1)

        Returns
        -------
        schur_arr : array
            Schur matrix
        """
        # Schur
        schur_list = nm.asarray(schur_list, dtype=nm.int32)
        schur_size = schur_list.shape[0]
        schur_arr = nm.empty((schur_size, schur_size),
                             dtype=self.dtype, order='C')

        self.struct.size_schur = schur_size
        self.struct.listvar_schur = schur_list.ctypes.data_as(PMumpsInt)
        self.struct.schur = schur_arr.ctypes.data_as(PMumpsComplex)

        # get matrix
        self.struct.schur_lld = schur_size
        self.struct.nprow = 1
        self.struct.npcol = 1
        self.struct.mblock = 100
        self.struct.nblock = 100

        self.struct.icntl[18] = 3  # centr. Schur complement stored by columns
        self.struct.job = 4  # analyze + factorize
        self._mumps_c(ctypes.byref(self.struct))

        return schur_arr

    def schur_reduction(self, b=None):
        """Schur recuction/condensation phase.

        Parameters
        ----------
        b : array
            RHS vector

        Returns
        -------
        schur_rhs : array
            Reduced/condensed RHS
        """
        if b is not None:
            self.set_rhs(b.copy())

        if 'rhs' not in self._data:
            raise ValueError('The right hand side vector is not set!')

        schur_size = self.struct.size_schur

        nrhs = self.struct.nrhs
        schur_rhs = nm.empty((schur_size, nrhs), dtype=self.dtype, order='F')
        self._schur_rhs = schur_rhs
        self.struct.lredrhs = schur_size
        self.struct.redrhs = schur_rhs.ctypes.data_as(PMumpsComplex)

        # get reduced/condensed RHS
        self.struct.icntl[25] = 1  # Reduction/condensation phase
        self.struct.job = 3  # solve
        self._mumps_c(ctypes.byref(self.struct))

        return schur_rhs

    def schur_expansion(self, x2):
        """Expansion to a complete solution.

        Parameters
        ----------
        x2 : array
            Partial solution

        Returns
        -------
        x : array
            Complete solution
        """
        self._schur_rhs[:] = x2
        self.struct.icntl[25] = 2  # Expansion phase
        self.struct.job = 3  # solve
        self._mumps_c(ctypes.byref(self.struct))

        return self._data['rhs']

    def _mumps_call(self, job):
        """Set the job and call MUMPS.

        Jobs:
        -----
        1: analyse
        2: factorize
        3: solve
        4: analyse, factorize
        5: factorize, solve
        6: analyse, factorize, solve
        """
        self.struct.job = job
        self._mumps_c(ctypes.byref(self.struct))

        if self.struct.infog[0] < 0:
            raise RuntimeError('MUMPS error: {self.struct.infog[0]}')

    def solve(self, b=None):
        """Solve the linear system.

        Parameters
        ----------
        b : array
            Right hand side

        Returns
        -------
        x : array
            Solution: x = inv(A) * b
        """
        if b is not None:
            self.set_rhs(b.copy())

        if 'factorized' in self._data and self._data['factorized']:
            self(3)
        else:
            self(6)

        return self._data['rhs']

    def schur_solve(self, schur_list, b=None):
        """Solve the linear system using the Schur complement method.

        Parameters
        ----------
        schur_list : array
            List of Schur DOFs (indexing starts with 1)
        b : array
            Right hand side

        Returns
        -------
        x : array
            Solution: x = inv(A) * b
        """
        import scipy.linalg as sla

        S = self.schur_complement(schur_list)
        y2 = self.schur_reduction(b)
        assume_a = 'sym' if self.struct.sym else 'gen'
        x2 = sla.solve(S.T, y2, assume_a=assume_a)

        return self.schur_expansion(x2)
