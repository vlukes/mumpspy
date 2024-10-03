MUMPSpy
=======

A python wrapper for the sparse direct solver.

The wrapper allows:

* real and complex factorization of symmetric or non-symmetric matrices
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

      pip install mumpspy

Usage
-----

```python
import mumpspy
import numpy as np
import scipy.sparse as sp

row = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
col = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1])
val = np.array([1, 2, 2, 1, 1, 3, -1, 2, 1, 1, 3, 1], dtype='d')
b = np.array([15, 12, 3, 5], dtype='d')

A = sp.coo_matrix((val, (row, col)), shape=(4, 4))

solver = mumpspy.MumpsSolver()  # initialize solver, real-valued system
solver.set_mtx(A)  # set sparse matrix
x = solver.solve(b)  # solve system for a given right-hand side
print(x)
```

Compatibility
-------------

Tested for the following MUMPS library versions:

* 4.10.0
* 5.0.2
* 5.1.2
* 5.2.1
* 5.4.1
