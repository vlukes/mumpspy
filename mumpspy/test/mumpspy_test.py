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
import pytest


OUTPUT_LEVEL = 1  # 1=brief or 2=verbose
_RTOL_double = 1e-9
_RTOL_single = 1e-6

optionsreal = ["float32", "float64"]
optionscomplex = ["complex64", "complex128"]


def print_arr(a):
    return (str(a)[1:-1]).strip()


def check_solution(exp, sol):
    return "passed" if nm.linalg.norm(exp_x - x) < 1e-9 else "failed!"


def print_system(title, A, b, x, exp_x):
    check = check_solution(exp_x, x)

    title = f"{title}: {check}"
    print(title)

    if OUTPUT_LEVEL > 1:
        print("=" * len(title))
        s = "\n       ".join([print_arr(ii) for ii in A.toarray()])
        print(f"  A: [ {s} ]")
        print(f"  b: [ {print_arr(b)} ]")
        print(f"  expected result  : [ {print_arr(exp_x)} ]")
        print(f"  calculated result: [ {print_arr(nm.round(x, 9))} ]")
        print()


def loadSystem(name="real-sym"):
    if name == "real-sym":
        title = "real-valued symmetric system"
        ridx = nm.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 4])
        cidx = nm.array([1, 3, 4, 2, 4, 1, 3, 1, 2, 4])
        val = nm.array([1, 2, -1, 3, -2, 2, -2, -1, -2, 1])
        b = nm.array([3, -2, -4, -1])
        schur_list = nm.array([3, 4])
        exp_x = nm.array([1, 2, 3, 4], dtype="d")
    elif name == "real-nonsym":
        title = "real-valued non-symmetric system"
        ridx = nm.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
        cidx = nm.array([1, 2, 4, 1, 2, 3, 3, 4, 1, 2, 3, 4])
        val = nm.array([2, 3, -1, 1, 2, -3, 2, -1, 2, -5, 2, 1])
        b = nm.array([4, -4, 2, 2])
        schur_list = []
        exp_x = nm.array([1, 2, 3, 4], dtype="d")
    elif name == "complex-nonsym":
        title = "complex-valued non-symmetrix system"
        ridx = nm.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
        cidx = nm.array([1, 2, 3, 1, 2, 3, 2, 3, 4, 3, 4])
        val = nm.array([1 + 2j, -2j, 3 + 3j, 1j, 7, 3j, -1, 1j, 1, 3 + 5j, 2 - 3j])
        b = nm.array([-3 + 6j, 1 + 8j, -1 + 1j, 4j])
        schur_list = nm.array([3, 4])
        exp_x = nm.array([1 + 0j, 1 + 1j, 0 + 2j, 2 + 2j])

    # build matrix
    A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))
    data = dict()
    data["title"] = title
    data["A"] = A
    data["b"] = b
    data["exp_x"] = exp_x
    data["ridx"] = ridx
    data["cidx"] = cidx
    data["val"] = val
    data["schur_list"] = schur_list
    data["exp_x"] = exp_x
    return data


@pytest.mark.parametrize("precision", optionsreal)
def testRealRCDsym(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(silent=False, system=precision)
    solver.set_rcd_mtx(data["ridx"], data["cidx"], data["val"], 4)
    solver.set_rhs(data["b"])  # b will be overwritten!
    x = solver.solve()  # x is b: True
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixsym(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(is_sym=True, silent=False, system=precision)
    solver.set_mtx(data["A"])
    x = solver.solve(data["b"])
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixSymDirectSolve(precision):
    # load data
    data = loadSystem(name="real-nonsym")
    #
    solver = mumpspy.MumpsSolver(is_sym=False, system=precision)
    x = solver.solve(data["A"], data["b"])
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixNonsym(precision):
    # load data
    data = loadSystem(name="real-nonsym")
    #
    solver = mumpspy.MumpsSolver(is_sym=False, system=precision)
    solver.set_mtx(data["A"])
    x = solver.solve(data["b"])
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixsymMultipleRHS(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(is_sym=True, system=precision)
    solver.set_mtx(data["A"])
    bb = nm.vstack([2 * data["b"], data["b"]]).T
    exp_xx = nm.vstack([2 * data["exp_x"], data["exp_x"]]).T
    xx = solver.solve(bb)
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(xx, exp_xx, rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixsymSchur(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(is_sym=True, system=precision)
    solver.set_mtx(data["A"], factorize=False)
    x = solver.schur_solve(data["schur_list"], data["b"])
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionsreal)
def testRealMatrixNonsymSchur(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(is_sym=True, system=precision)
    solver.set_mtx(data["A"], factorize=False)
    x = solver.schur_solve(data["schur_list"], data["b"])
    del solver
    if precision == "float32":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionscomplex)
def testComplexMatrixNonsym(precision):
    # load data
    data = loadSystem(name="complex-nonsym")
    #
    solver = mumpspy.MumpsSolver(system=precision)
    solver.set_mtx(data["A"])
    x = solver.solve(data["b"])
    del solver
    if precision == "complex64":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


@pytest.mark.parametrize("precision", optionscomplex)
def testComplexMatrixsymSchur(precision):
    # load data
    data = loadSystem(name="real-sym")
    #
    solver = mumpspy.MumpsSolver(system=precision)
    solver.set_mtx(data["A"], factorize=False)
    x = solver.schur_solve(data["schur_list"], data["b"])
    del solver
    if precision == "complex64":
        _RTOL = _RTOL_single
    else:
        _RTOL = _RTOL_double
    nm.testing.assert_allclose(x, data["exp_x"], rtol=_RTOL)


# title = 'real-valued symmetric system'
# ridx = nm.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 4])
# cidx = nm.array([1, 3, 4, 2, 4, 1, 3, 1, 2, 4])
# val = nm.array([1, 2, -1, 3, -2, 2, -2, -1, -2, 1])
# b = nm.array([3, -2, -4, -1])
# A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

# solver = mumpspy.MumpsSolver()
# solver.set_rcd_mtx(ridx, cidx, val, 4)
# solver.set_rhs(b)  # b will be overwritten!
# x = solver.solve()  # x is b: True
# del solver

# print_system(title + ' (set_rcd_mtx)', A, b, x, exp_x)

# solver = mumpspy.MumpsSolver(is_sym=True)
# solver.set_mtx(A)
# x = solver.solve(b)
# del solver

# print_system(title + ' (set_mtx)', A, b, x, exp_x)

# solver = mumpspy.MumpsSolver(is_sym=True)
# solver.set_mtx(A)
# bb = nm.vstack([2*b, b]).T
# exp_xx = nm.vstack([2*exp_x, exp_x]).T
# xx = solver.solve(bb)
# del solver

# print_system(title + ' (multiple RHS)', A, bb, xx, exp_xx)

# solver = mumpspy.MumpsSolver(is_sym=True)
# solver.set_mtx(A, factorize=False)
# x = solver.schur_solve(schur_list, b)
# del solver

# print_system(title + ' (Schur complement)', A, b, x, exp_x)

# title = 'real-valued non-symmetric system'
# ridx = nm.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
# cidx = nm.array([1, 2, 4, 1, 2, 3, 3, 4, 1, 2, 3, 4])
# val = nm.array([2, 3, -1, 1, 2, -3, 2, -1, 2, -5, 2, 1])
# b = nm.array([4, -4, 2, 2])
# A = sps.coo_matrix((val, (ridx - 1, cidx - 1)), shape=(4, 4))

# solver = mumpspy.MumpsSolver()
# solver.set_mtx(A)
# solver.set_rhs(b)  # b will be overwritten!
# x = solver.solve()  # x is b: True
# del solver

# print_system(title, A, b, x, exp_x)

# solver = mumpspy.MumpsSolver()
# solver.set_mtx(A, factorize=False)
# x = solver.schur_solve(schur_list, b)
# del solver

# print_system(title + ' (Schur complement)', A, b, x, exp_x)


# solver = mumpspy.MumpsSolver(system='complex')
# solver.set_mtx(A)
# x = solver.solve(b)
# del solver

# print_system(title, A, b, x, exp_x)

# solver = mumpspy.MumpsSolver(system='complex')
# solver.set_mtx(A, factorize=False)
# x = solver.schur_solve(schur_list, b)
# del solver

# print_system(title + ' (Schur complement)', A, b, x, exp_x)
