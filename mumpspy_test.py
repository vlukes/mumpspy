"""
MUMPS test examples.

Expected results:
    real system: [ 1. 2. 3. 4.]
    complex system: [1 + 0i, 1 + 1i, 0 + 2i, 2 + 2i]
"""

import numpy as np
import mumpspy
import scipy.sparse as sps
import scipy.linalg as sla


def print_arr(a):
    return (str(a)[1:-1]).strip()


def check_solution(exp, sol):
    return 'passed' if np.linalg.norm(exp_x - x) < 1e-9 else 'failed!'


# real-valued system
print('real-valued system')
print('==================')
ridx = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4], dtype='i')
cidx = np.array([1, 2, 4, 1, 2, 3, 3, 4, 1, 2, 3, 4], dtype='i')
val = np.array([2, 3, -1, 1, 2, -3, 2, -1, 2, -5, 2, 1], dtype='d')
b = np.array([4, -4, 2, 2], dtype='d')
exp_x = np.array([1, 2, 3, 4], dtype='d')

A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

s = '\n       '.join([print_arr(ii) for ii in A.toarray()])
print(f'  A: [ {s} ]')
print(f'  b: [ {print_arr(b)} ]')
print(f'  expected result:        [ {print_arr(exp_x)} ]')

solver = mumpspy.MumpsSolver()
solver.set_rcd_mtx(ridx, cidx, val, 4)
x = solver.solve(b)

check = check_solution(exp_x, x)
print(f'  solution (set_rcd_mtx): [ {print_arr(np.round(x, 9))} ] ({check})')

solver.set_mtx(A)
solver.set_rhs(b)  # b will be overwritten!
x = solver.solve()  # x is b: True

check = check_solution(exp_x, x)
print(f'  solution (set_mtx):     [ {print_arr(np.round(x, 9))} ] ({check})\n')

del solver


# complex-valued system
print('complex-valued system')
print('=====================')
ridx = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], dtype='i')
cidx = np.array([1, 2, 3, 1, 2, 3, 2, 3, 4, 3, 4], dtype='i')
val = np.array([1 + 2j, -2j, 3 + 3j, 1j, 7, 3j, -1, 1j, 1, 3 + 5j, 2 - 3j],
               dtype='D')
b = np.array([-3 + 6j, 1 + 8j, -1 + 1j, 4j], dtype='D')
exp_x = np.array([1 + 0j, 1 + 1j, 0 + 2j, 2 + 2j], dtype='D')

A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

s = '\n       '.join([print_arr(ii) for ii in A.toarray()])
print(f'  A: [ {s} ]')
print(f'  b: [ {print_arr(b)} ]')
print(f'  expected result: [ {print_arr(exp_x)} ]')

solver = mumpspy.MumpsSolver(system='complex')
solver.set_mtx(A)
x = solver.solve(b)

check = check_solution(exp_x, x)
print(f'  solution:        [ {print_arr(np.round(x, 9))} ] ({check})\n')

del solver


# Schur complement
print('Schur complement')
print('================')
ridx = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4], dtype='i')
cidx = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1, 2], dtype='i')
val = np.array([1, 2, 2, 1, 1, 3, -1, 2, 1, 1, 3, 1], dtype='d')
b = np.array([15, 12, 3, 5], dtype='d')
schur_list = np.array([3, 4], dtype='i')
exp_x = np.array([1, 2, 3, 4], dtype='d')

A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

s = '\n       '.join([print_arr(ii) for ii in A.toarray()])
print('  A: [ %s ]' % s)
print('  b: [ %s ]' % print_arr(b))
print('  Schur list: [ %s ]' % print_arr(schur_list))
print('  expected result: [ %s ]' % print_arr(exp_x))

solver = mumpspy.MumpsSolver()
solver.set_mtx(A)
x = solver.schur_solve(schur_list, b)

check = check_solution(exp_x, x)
print(f'  solution:        [ {print_arr(np.round(x, 9))} ] ({check})\n')

del solver

