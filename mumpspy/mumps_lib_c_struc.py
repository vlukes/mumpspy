## file built from headers (include directory) of MUMPS library
## tools that can be used: ctypesgen (pip install ctypesgen)
## on include folder: run for file in `ls *.h`;do ctypesgen -l${file%.*} $file -o ${file%.*}.py;done


import ctypes

MIN_SUPPORTED_VERSION = "4.10.0"
MAX_SUPPORTED_VERSION = "5.7.999"

AUX_LENGTH = 16 * 1024

c_pointer = ctypes.POINTER

## define structures for complex numbers
class complex8(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float),
                ("i", ctypes.c_float)]
class complex16(ctypes.Structure):
    _fields_ = [("r", ctypes.c_double),
                ("i", ctypes.c_double)]


MumpsInt = ctypes.c_int
PMumpsInt = c_pointer(MumpsInt)
MumpsInt8 = ctypes.c_uint64
PMumpsInt8 = c_pointer(MumpsInt8)
#
MumpsReal = ctypes.c_float
PMumpsReal = c_pointer(MumpsReal)
#
MumpsComplex = complex8
PMumpsComplex = c_pointer(MumpsComplex) 
#
MumpsReal8 = ctypes.c_double
PMumpsReal8 = c_pointer(MumpsReal8)
#
MumpsComplex16 = complex16 #MumpsReal8
PMumpsComplex16 = c_pointer(MumpsComplex16) 
#


def get_all_fields(mumps_type="d"):
    """Get all declared and updated fields for MUMPS C structure for a given precision.

    Args:
        mumps_type (str, optional): MUMPS type letter (s, d, c, z). Defaults to 'd'.

    Returns:
        mumps_c_fields: basics fields for MUMPS C structure
        mumps_c_updates: incremental updates for MUMPS C structure
    """
    cMumpsInt = MumpsInt
    cPMumpsInt = PMumpsInt
    cMumpsInt8 = MumpsInt8
    cPMumpsInt8 = PMumpsInt8

    # adapt type to precision
    if mumps_type.startswith("s"):
        cTypeData = MumpsReal
        cPTypeData = PMumpsReal
        cTypeDataReal = MumpsReal
        cPTypeDataReal = PMumpsReal
    elif mumps_type.startswith("d"):
        cTypeData = MumpsReal8
        cPTypeData = PMumpsReal8
        cTypeDataReal = MumpsReal8
        cPTypeDataReal = PMumpsReal8
    elif mumps_type.startswith("c"):
        cTypeData = MumpsComplex
        cPTypeData = PMumpsComplex
        cTypeDataReal = MumpsReal
        cPTypeDataReal = PMumpsReal
    elif mumps_type.startswith("z"):
        cTypeData = MumpsComplex16
        cPTypeData = PMumpsComplex16
        cTypeDataReal = MumpsReal8
        cPTypeDataReal = PMumpsReal8
    else:
        raise ValueError(f"Precision {mumps_type} not supported!")

    #

    mumps_c_fields = [  # MUMPS 4.10.0
        ("sym", cMumpsInt),
        ("par", cMumpsInt),
        ("job", cMumpsInt),
        ("comm_fortran", cMumpsInt),
        ("icntl", cMumpsInt * 40),
        ("cntl", cTypeDataReal * 15),
        ("n", cMumpsInt),
        #
        ("nz_alloc", cMumpsInt),
        # /* Assembled entry */
        ("nz", cMumpsInt),
        ("irn", cPMumpsInt),
        ("jcn", cPMumpsInt),
        ("a", cPTypeData),
        # /* Distributed entry */
        ("nz_loc", cMumpsInt),
        ("irn_loc", cPMumpsInt),
        ("jcn_loc", cPMumpsInt),
        ("a_loc", cPTypeData),
        # /* Element entry */
        ("nelt", cMumpsInt),
        ("eltptr", cPMumpsInt),
        ("eltvar", cPMumpsInt),
        ("a_elt", cPTypeData),
        # /* Ordering, if given by user */
        ("perm_in", cPMumpsInt),
        # /* Orderings returned to user */
        ("sym_perm", cPMumpsInt),
        ("uns_perm", cPMumpsInt),
        # /* Scaling (input only in this version) */
        ("colsca", cPTypeDataReal),
        ("rowsca", cPTypeDataReal),
        # /* RHS, solution, ouptput data and statistics */
        ("rhs", cPTypeData),
        ("redrhs", cPTypeData),
        ("rhs_sparse", cPTypeData),
        ("sol_loc", cPTypeData),
        ("irhs_sparse", cPMumpsInt),
        ("irhs_ptr", cPMumpsInt),
        ("isol_loc", cPMumpsInt),
        ("nrhs", cMumpsInt),
        ("lrhs", cMumpsInt),
        ("lredrhs", cMumpsInt),
        ("nz_rhs", cMumpsInt),
        ("lsol_loc", cMumpsInt),
        ("schur_mloc", cMumpsInt),
        ("schur_nloc", cMumpsInt),
        ("schur_lld", cMumpsInt),
        ("mblock", cMumpsInt),
        ("nblock", cMumpsInt),
        ("nprow", cMumpsInt),
        ("npcol", cMumpsInt),
        ("info", cMumpsInt * 40),
        ("infog", cMumpsInt * 40),
        ("rinfo", cPTypeDataReal * 40),
        ("rinfog", cPTypeDataReal * 40),
        # /* Null space */
        ("deficiency", cMumpsInt),
        ("pivnul_list", cPMumpsInt),
        ("mapping", cPMumpsInt),
        # /* Schur */
        ("size_schur", cMumpsInt),
        ("listvar_schur", cPMumpsInt),
        ("schur", cPTypeData),
        # /* Internal parameters */
        ("instance_number", cMumpsInt),
        ("wk_user", cPTypeData),
        # /* Version number:
        #  length in FORTRAN + 1 for final \0 + 1 for alignment */
        ("version_number", ctypes.c_char * 16),
        # /* For out-of-core */
        ("ooc_tmpdir", ctypes.c_char * 256),
        ("ooc_prefix", ctypes.c_char * 64),
        # /* To save the matrix in matrix market format */
        ("write_problem", ctypes.c_char * 256),
        ("lwk_user", cMumpsInt),
    ]

    mumps_c_updates = {  # incremental updates related to version 4.10.0
        "5.0.0": [
            ("new_after", "icntl", ("keep", cMumpsInt * 500)),
            (
                "new_after",
                "cntl",
                [
                    ("dkeep", cTypeDataReal * 130),
                    ("keep8", cMumpsInt8 * 150),
                ],
            ),
            (
                "new_after",
                "rowsca",
                [
                    ("colsca_from_mumps", cMumpsInt),
                    ("rowsca_from_mumps", cMumpsInt),
                ],
            ),
            ("replace", "version_number", ctypes.c_char * 27),
        ],
        "5.1.0": [
            ("replace", "dkeep", cTypeDataReal * 230),
            ("new_after", "nz", ("nnz", cMumpsInt8)),
            ("new_after", "nz_loc", ("nnz_loc", cMumpsInt8)),
            ("replace", "version_number", ctypes.c_char * 32),
            (
                "new_after",
                "lwk_user",
                [
                    # /* For save/restore feature */
                    ("save_dir", ctypes.c_char * 256),
                    ("save_prefix", ctypes.c_char * 256),
                ],
            ),
        ],
        "5.2.0": [
            ("replace", "icntl", cMumpsInt * 60),
            ("new_after", "sol_loc", ("rhs_loc", cPTypeData)),
            ("new_after", "isol_loc", ("irhs_loc", cPMumpsInt)),
            (
                "new_after",
                "lsol_loc",
                [
                    ("nloc_rhs", cMumpsInt),
                    ("lrhs_loc", cMumpsInt),
                ],
            ),
            ("replace", "info", cMumpsInt * 80),
            ("replace", "infog", cMumpsInt * 80),
            ("new_after", "save_prefix", ("metis_options", cMumpsInt * 40)),
        ],
        "5.3.0": [
            ("new_after", "n", ("nblk", cMumpsInt)),
            (
                "new_after",
                "a_elt",
                [
                    # /* Matrix by blocks */
                    ("blkptr", cPMumpsInt),
                    ("blkvar", cPMumpsInt),
                ],
            ),
        ],
        "5.7.0": [
            ("new_after", "rowsca_from_mumps", 
                [
                    ("colsca_loc", cPTypeDataReal),
                    ("rowsca_loc", cPTypeDataReal),
                    ("rowind", cPMumpsInt),
                    ("colind", cPMumpsInt),
                    ("pivots", cPTypeData)
                ]
            ),
            ("new_after", "rhs_loc", ("rhsintr", cPTypeData)),
            ("new_after", "irhs_loc", 
                [
                    ("glob2loc_rhs", cPMumpsInt),
                    ("glob2loc_sol", cPMumpsInt)
                ]
            ),
            ("new_after", "lrhs_loc", ("nsol_loc", cMumpsInt)),
            ("new_after", "npcol", ("ld_rhsintr", cMumpsInt)),
            ("new_after", "mapping", ("singular_values", cPTypeDataReal)),
            ("delete", "instance_number"),
            ("replace", "ooc_tmpdir", ctypes.c_char * 1024),
            ("replace", "ooc_prefix", ctypes.c_char * 256),
            ("replace", "write_problem", ctypes.c_char * 1024),
            ("replace", "save_dir", ctypes.c_char * 1024),
            ("new_after", "metis_options", ("instance_number", cMumpsInt)),
        ],
    }

    return mumps_c_fields, mumps_c_updates


def version_to_int(v):
    """Convert the version string to an integer ('5.2.1' --> 5002001)."""
    return sum(int(vk) * 10 ** (3 * k) for k, vk in enumerate(v.split(".")[::-1]))


def get_mumps_c_fields(version=None, mumps_type='d'):
    """Return the MUMPS C fields for a given MUMPS version."""

    def update_fields(f, update_f):
        for uf in update_f:
            fk = [k for k, _ in f]
            idx = fk.index(uf[1])

            if uf[0] == "replace":
                f[idx] = (uf[1], uf[2])
            elif uf[0] == "delete":
                del f[idx]
            elif uf[0] == "new_after":
                if isinstance(uf[2], list):
                    f[(idx + 1) : (idx + 1)] = uf[2]
                else:
                    f.insert(idx + 1, uf[2])

        return f

    mumps_c_fields, mumps_c_updates = get_all_fields(mumps_type=mumps_type)

    if version is None:
        fields = mumps_c_fields[:5] + [("aux", ctypes.c_uint8 * AUX_LENGTH)]
    else:
        update_keys = list(mumps_c_updates.keys())
        update_keys.sort()

        vnum = version_to_int(version)
        if vnum < version_to_int(MIN_SUPPORTED_VERSION):
            msg = (
                f"MUMPS version {version} not supported! "
                f"({version} < {MIN_SUPPORTED_VERSION})"
            )
            raise ValueError(msg)

        if vnum > version_to_int(MAX_SUPPORTED_VERSION):
            msg = (
                f"MUMPS version {version} not supported! "
                f"({version} > {MAX_SUPPORTED_VERSION})"
            )
            raise ValueError(msg)

        fields = mumps_c_fields.copy()
        for ukey in update_keys:
            if version_to_int(ukey) > vnum:
                break

            fields = update_fields(fields, mumps_c_updates[ukey])

    return fields


def define_mumps_c_struc(version=None, mumps_type='d'):
    """Return MUMPS_C_STRUC class with given fields."""

    class Mumps_c_struc(ctypes.Structure):
        _fields_ = get_mumps_c_fields(version, mumps_type=mumps_type)

    return Mumps_c_struc
