MUMPSpy
=======

A python wrapper for the parallel sparse direct solver.

The wrapper allows to:

* real and complex arithmetic
* parallel run
* Schur complement calculation

Requirements
------------

* [MUMPS](http://mumps-solver.org) - MUltifrontal Massively Parallel sparse
  direct Solver
* [mpi4py](http://mpi4py.scipy.org/) - Python bindings for MPI

Ubuntu/Debian users can use the following command to install the required
packages:

    apt-get install python-mpi4py libmumps-dev

Installation
------------

* Download the code from the git repository:

      git clone git://github.com/vlukes/mumpspy

or

* Use [pip](https://pypi.org/project/pip/):

      pip install git+git://github.com/vlukes/mumpspy

Usage
-----

```python
import mumpspy

solver = mumpspy.MumpsSolver(system='real')  # initialize solver, real-valued system
solver.set_mtx_centralized(A)  # set sparse matrix
x = b.copy()
solver.set_rhs(x)  # set right-hand side
solver(6)  # analyse, factorize, solve
print x
del(solver)  # cleanup
```

Compatibility
-------------

Tested for the following MUMPS library versions:

* 4.10.0
* 5.0.1, 5.0.2
* 5.1.2
* 5.2.1
* 5.4.1