"""
MUMPS test examples.

Expected results:
    real system: [ 1. 2. 3. 4.]
    complex system: [1 + 0i, 1 + 1i, 0 + 2i, 2 + 2i]
"""

import numpy as nm
import mumpspy
import scipy.sparse as sps
import scipy.linalg as sla


OUTPUT_LEVEL = 1  # 1=brief or 2=verbose


def print_arr(a):
    return (str(a)[1:-1]).strip()


def check_solution(exp, sol):
    return 'passed' if nm.linalg.norm(exp_x - x) < 1e-9 else 'failed!'


def print_system(title, A, b, x, exp_x):
    check = check_solution(exp_x, x)

    title = f'{title}: {check}'
    print(title)

    if OUTPUT_LEVEL > 1:
        print('=' * len(title))
        s = '\n       '.join([print_arr(ii) for ii in A.toarray()])
        print(f'  A: [ {s} ]')
        print(f'  b: [ {print_arr(b)} ]')
        print(f'  expected result  : [ {print_arr(exp_x)} ]')
        print(f'  calculated result: [ {print_arr(nm.round(x, 9))} ]')
        print()


exp_x = nm.array([1, 2, 3, 4], dtype='d')
schur_list = nm.array([3, 4])

title = 'real-valued symmetric system'
ridx = nm.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 4])
cidx = nm.array([1, 3, 4, 2, 4, 1, 3, 1, 2, 4])
val = nm.array([1, 2, -1, 3, -2, 2, -2, -1, -2, 1])
b = nm.array([3, -2, -4, -1])
A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

solver = mumpspy.MumpsSolver()
solver.set_rcd_mtx(ridx, cidx, val, 4)
solver.set_rhs(b)  # b will be overwritten!
x = solver.solve()  # x is b: True
del solver

print_system(title + ' (set_rcd_mtx)', A, b, x, exp_x)

solver = mumpspy.MumpsSolver(is_sym=True)
solver.set_mtx(A)
x = solver.solve(b)
del solver

print_system(title + ' (set_mtx)', A, b, x, exp_x)

solver = mumpspy.MumpsSolver(is_sym=True)
solver.set_mtx(A)
bb = nm.vstack([2*b, b]).T
exp_xx = nm.vstack([2*exp_x, exp_x]).T
xx = solver.solve(bb)
del solver

print_system(title + ' (multiple RHS)', A, bb, xx, exp_xx)

solver = mumpspy.MumpsSolver(is_sym=True)
solver.set_mtx(A, factorize=False)
x = solver.schur_solve(schur_list, b)
del solver

print_system(title + ' (Schur complement)', A, b, x, exp_x)

title = 'real-valued non-symmetric system'
ridx = nm.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
cidx = nm.array([1, 2, 4, 1, 2, 3, 3, 4, 1, 2, 3, 4])
val = nm.array([2, 3, -1, 1, 2, -3, 2, -1, 2, -5, 2, 1])
b = nm.array([4, -4, 2, 2])
A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

solver = mumpspy.MumpsSolver()
solver.set_mtx(A)
solver.set_rhs(b)  # b will be overwritten!
x = solver.solve()  # x is b: True
del solver

print_system(title, A, b, x, exp_x)

solver = mumpspy.MumpsSolver()
solver.set_mtx(A, factorize=False)
x = solver.schur_solve(schur_list, b)
del solver

print_system(title + ' (Schur complement)', A, b, x, exp_x)

title = 'complex-valued non-symmetrix system'
ridx = nm.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
cidx = nm.array([1, 2, 3, 1, 2, 3, 2, 3, 4, 3, 4])
val = nm.array([1 + 2j, -2j, 3 + 3j, 1j, 7, 3j, -1, 1j, 1, 3 + 5j, 2 - 3j])
b = nm.array([-3 + 6j, 1 + 8j, -1 + 1j, 4j])
exp_x = nm.array([1 + 0j, 1 + 1j, 0 + 2j, 2 + 2j])
A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

solver = mumpspy.MumpsSolver(system='complex')
solver.set_mtx(A)
x = solver.solve(b)
del solver

print_system(title, A, b, x, exp_x)

solver = mumpspy.MumpsSolver(system='complex')
solver.set_mtx(A, factorize=False)
x = solver.schur_solve(schur_list, b)
del solver

print_system(title + ' (Schur complement)', A, b, x, exp_x)
