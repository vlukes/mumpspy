"""
MUMPS test examples.

Expected results:
    real system: [ 1. 2. 3. 4.]
    complex system: [1 + 0i, 1 + 1i, 0 + 2i, 2 + 2i]
"""

import numpy as np
import mumpspy
import scipy.sparse as sps


# real-valued system
print('real-valued system - expected results: [1, 2, 3, 4]')
ridx = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4], dtype='i')
cidx = np.array([1, 2, 4, 1, 2, 3, 3, 4, 1, 2, 3, 4], dtype='i')
val = np.array([2, 3, -1, 1, 2, -3, 2, -1, 2, -5, 2, 1], dtype='d')
b = np.array([4, -4, 2, 2], dtype='d')

A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

solver = mumpspy.MumpsSolver()
x = b.copy()
solver.set_b(x)
solver.set_rcd_centralized(ridx, cidx, val, 4)
solver(6)  # analyse, factorize, solve
print('  solution (set_rcd_centralized):  [%s]'
      % ', '.join('%.3f' % k for k in x))

solver.set_A_centralized(sps.coo_matrix(A))
x = b.copy()
solver.set_b(x)
solver(6)  # analyse, factorize, solve
print('  solution (set_A_centralized):  [%s]'
      % ', '.join('%.3f' % k for k in x))

del(solver)


# complex-valued system
print('real-valued system - expected results: [1+0i, 1+1i, 0+2i, 2+2i]')
ridx = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], dtype='i')
cidx = np.array([1, 2, 3, 1, 2, 3, 2, 3, 4, 3, 4], dtype='i')
val = np.array([1 + 2j, -2j, 3 + 3j, 1j, 7, 3j, -1, 1j, 1, 3 + 5j, 2 - 3j],
               dtype='D')
b = np.array([-3 + 6j, 1 + 8j, -1 + 1j, 4j], dtype='D')

A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

solver = mumpspy.MumpsSolver(system='complex')
solver.set_A_centralized(sps.coo_matrix(A))
x = b.copy()
solver.set_b(x)
solver(6)  # analyse, factorize, solve
print('  solution (set_A_centralized):  [%s]'
      % ', '.join('%.3f+%.3fi' % (k.real, k.imag) for k in x))

del(solver)
