MUMPSpy
=======

![PyPI - Version](https://img.shields.io/pypi/v/mumpspy)

A python wrapper for the sparse direct solver.

The wrapper allows:

* real and complex factorization of symmetric or non-symmetric matrices
* single and double precision
* Schur complement calculation

Requirements
------------

* [MUMPS](http://mumps-solver.org) - MUltifrontal Massively Parallel sparse
  direct Solver
* [mpi4py](http://mpi4py.scipy.org/) - Python bindings for MPI

Ubuntu/Debian users can use the following command to install the required
packages:

    apt-get install python-mpi4py libmumps-dev

The cmake version of MUMPS could be also used. See [https://github.com/scivision/mumps](https://github.com/scivision/mumps) for details and [CI file](.github/workflows/CI-Ubuntu_tests.yml) for some syntax.

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

Additional examples of syntax can be found in [test file](mumpspy/test/mumpspy_test.py).

Testing
-------------

`pytest` can be used to run testing and coverage in the package.

Compatibility
-------------

Tested for the following MUMPS library versions (see Actions for current working versions):

<!-- ![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.4.1)
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.5.0) 
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.5.1) 
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.6.0) 
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.6.1)
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.6.2) 
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.7.0) 
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.7.1)
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.7.2)
![](https://byob.yarr.is/luclaurent/mumpspy/Macos-3.12_5.7.3)  -->

* 4.10.0
* 5.0.2
* 5.1.2
* 5.2.1
* 5.4.1
* 5.6.1
