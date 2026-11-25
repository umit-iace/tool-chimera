import logging

import pytest
import sympy as sp
from sympy import Integer, Matrix, sqrt, pprint
from sympy.core.random import sample
from sympy.polys.matrices import DomainMatrix

from chimera import QPoly, QuasiPolynomialRing
from chimera.algos import (
    canonical_form,
    check_col_degrees_on_diagonal,
    make_entire,
    move_col_degrees_on_diagonal,
    qpld_single,
    qplde_single,
    qplde,
    reduce_matrix_shifts,
)
from chimera.poly import random_col_reduced_mat
from chimera.utils import (
    check_singularities,
    get_lccm,
    get_matrix_terms,
    show_matrix_degrees,
    show_matrix_degree,
)

# debugging
lg = logging.getLogger("chimera.algos").setLevel(logging.DEBUG)

# generators
sigma = sp.Symbol("sigma", positive=True, real=True)
s = sp.Symbol("s")


def test_qpld_single():
    xl = [
        3 * sigma ** sp.sqrt(5) + 1 * sigma + 5,
        5 + 2 * sigma ** (-2 * sp.pi),
        3 * sigma ** sp.sqrt(5) + 1 * sigma + 5 + 5 + 2 * sigma ** (-2 * sp.pi),
    ]
    yl = [
        3 * sigma ** sp.sqrt(3) + 1 * sigma + 5,
        5 + 2 * sigma ** (-sp.pi),
        3 * sigma ** sp.sqrt(3) + 1 * sigma + 5 + 5 + 2 * sigma ** (-sp.pi),
    ]

    for x, y in zip(xl, yl):
        print(f"Dividing {x} by {y}:")
        xp = QPoly(x, sigma)
        yp = QPoly(y, sigma)
        qp, rp = qpld_single(xp, yp, sigma)
        r = rp.as_expr()
        q = qp.as_expr()
        print(f"q: {q}")
        print(f"r: {r}")

        assert rp.degmax() <= yp.degmax()
        assert rp.degree() <= yp.degree()

        err = x - q * y - r
        err = err.simplify()
        print(f"error: {err}")
        assert err == 0


def test_make_entire():
    exprs = [
        2 * s / s**3 * (3 * s + 4) * sigma**3 + 1 / (s**2 + 2) * sigma,
        1 / s**4 * sigma ** sp.sqrt(3),
        1 / (1 - s) ** 4 * sigma ** sp.sqrt(3),
        # 2 * s / (3 * s**2 - 5 * s + 4) * sigma**3 + (s**4 + 1) / s**2 * (s**2 - 2) * sigma,
    ]
    for expr in exprs:
        print(f"Testing {expr}:")
        gp = QPoly(expr, sigma)
        egp, p = make_entire(gp, s)

        print("Checking for singularities in quotient and remainder:")
        check_singularities(egp, s, sigma, sp.exp(s))

        # all coefficients should stay the same, except for the constant part
        for exp, cff in egp.as_dict().items():
            if exp == 0:
                continue
            err = (gp.as_dict()[exp] - cff).as_expr().simplify()
            assert err == 0


def test_qplde_single():
    xl = [
        (-3 * s ** 2 / 94 + 17 * s / 94 + sp.Rational(70, 47)) / sigma ** (sp.log(2) / 4),
        s * sigma ** sp.sqrt(5) + 1,
        (3 * s + 2 * s**2) * sigma ** sp.sqrt(5) + (s**2 + 2 * s) * sigma + 2 * s + 1,
        5 * s + (s + sp.Rational(1, 2)) * sigma ** (-2 * sp.pi),
        (3 * s + 2 * s**2) * sigma ** sp.sqrt(5)
        + (s**2 + 2 * s) * sigma
        + 2 * s
        + 1
        + 5 * s
        + (2 * s + 1) * sigma ** (-2 * sp.pi),
    ]
    yl = [
        -3 * s ** 3 + 29 * s ** 2 + 25 * s + 4,
        (s**2 - 2) * sigma ** sp.sqrt(3) + 4,
        (3 * s**4 + 2 * s**2) * sigma ** sp.sqrt(2)
        + (s**2 + 2 * s) * sigma
        + 2 * s
        + 1,
        5 * s + s * (s + 1) * (s - 4) * sigma ** (-sp.pi),
        (3 * s + 2 * s**3) * sigma ** sp.sqrt(2)
        + (s**2 + 2 * s) * sigma
        + 2 * s
        + 1
        + 5 * s
        + (2 * s**2 + 1) * sigma ** (-sp.pi),
    ]
    for x, y in zip(xl, yl):
        print(f"Dividing")
        pprint(x, wrap_line=False)
        print(f"by")
        pprint(y, wrap_line=False)

        xp = QPoly(x, sigma)
        yp = QPoly(y, sigma)
        qep, rep = qplde_single(xp, yp, s)
        qe = qep.as_expr()
        re = rep.as_expr()
        print(f"Quotient:")
        pprint(qe, wrap_line=False)
        print(f"Remainder:")
        pprint(re, wrap_line=False)

        test_rep = xp - qep * yp
        err = (test_rep - rep).as_expr().simplify()
        assert err == 0

        test_re = x - qe * y
        err = (test_re - re).simplify()
        assert err == 0

        # check degrees of remainder
        assert rep.degmax() <= yp.degmax()
        assert rep.degree() <= yp.degree()

        print("Checking for singularities in quotient...")
        assert check_singularities(qep, s, sigma, sp.exp(s), verbose=True)

        print("Checking for singularities in remainder...")
        assert check_singularities(rep, s, sigma, sp.exp(s), verbose=True)


def test_qplde():
    # scalar case
    x = (-3 * s ** 2 / 94 + 17 * s / 94 + sp.Rational(70, 47)) / sigma ** (sp.log(2) / 4)
    y = -3 * s ** 3 + 29 * s ** 2 + 25 * s + 4

    if 0:
        x = s * sigma ** sp.sqrt(5) + 1,
        y = (s ** 2 - 2) * sigma ** sp.sqrt(3) + 4,

    # convert to domain system
    coeff_dom = sp.EX.frac_field(s)
    qp_ring = QuasiPolynomialRing(coeff_dom, sigma)

    m = DomainMatrix.from_list([[y], [x]], qp_ring)
    qv, rv = qplde(m, row=1, c_gen=s)
    qq = qv[0, 0]
    rq = rv[0, 0]

    pprint(qq.element.as_expr(), wrap_line=False)
    pprint(rq.element.as_expr(), wrap_line=False)

    test_r = m[1,0] - qq * m[0,0]
    err = (test_r - rq).element.as_expr().expand()
    # pprint(err, wrap_line=False)
    assert err == 0

    # check degrees of remainder
    assert rq.element.degmax() <= m[0,0].element.degmax()
    assert rq.element.degree() <= m[0,0].element.degree()

    print("Checking for singularities in quotient...")
    assert check_singularities(qq.element, s, sigma, sp.exp(s), verbose=True)

    print("Checking for singularities in remainder...")
    assert check_singularities(rq.element, s, sigma, sp.exp(s), verbose=True)


def _generate_test_case(n, nm, np):
    """
    n: ODE state dimension
    nm: Negative transport systems cnt
    np: Positive transport systems cnt
    """
    # ODE
    d_mat = random_col_reduced_mat(nm, nm, n, -5, 5, s)
    n_mat = sp.Matrix.zeros(n, nm)

    # Coupling
    q0_mat = sp.Matrix.zeros(np, nm)
    while q0_mat.rank() < np:
        q0_mat = sp.randMatrix(np, nm, -10, 10)
    q1_mat = sp.Matrix.zeros(nm, np)
    while q1_mat.rank() < np:
        q1_mat = sp.randMatrix(nm, np, -10, 10)
    c_mat = sp.Matrix.zeros(np, n)

    # PDE
    tau_m = sorted([sp.sqrt(2 * s + 1) for s in sample(range(1, 6), nm)])
    tau_p = sorted([-sp.sqrt(2 * s + 1) for s in sample(range(1, 6), np)])[::-1]

    c = d_mat, n_mat, c_mat, q0_mat, q1_mat, tau_m, tau_p
    return c


def _run_reduction_test(case):
    print("----- Running reduction test ------")
    print(f" case = {case}")
    print("-----------------------------------")

    (d_mat, n_mat, c_mat, q0_mat, q1_mat, tau_m, tau_p) = case

    assert q0_mat.rank() == q1_mat.rank() == len(tau_p)

    pred_mat = sp.Matrix.diag([sigma**tau for tau in tau_m])
    delay_mat = sp.Matrix.diag([sigma**tau for tau in tau_p])

    des_state_dim = (sp.Add(*tau_m) - sp.Add(*tau_p)).simplify()
    print(f"System predictions: {tau_m} ({[sp.N(t) for t in tau_m]})")
    print(f"System delays: {tau_p} ({[sp.N(t) for t in tau_p]})")
    print(
        f"Original state dimensions {[(tm - tp, (tm - tp).evalf()) for tm, tp in zip(tau_m, tau_p)]}"
    )
    print(f"Original full state dim: {des_state_dim} ({des_state_dim.evalf()})")

    print("G Matrix:")
    h_mat = q0_mat @ d_mat + c_mat @ n_mat
    m = pred_mat @ d_mat - q1_mat @ delay_mat @ h_mat
    # sp.pprint(m, wrap_line=False)

    # convert to domain system
    coeff_dom = sp.EX.frac_field(s)
    qp_ring = QuasiPolynomialRing(coeff_dom, sigma)
    m = DomainMatrix.from_list(m.tolist(), domain=qp_ring)
    show_matrix_degree(m)
    show_matrix_degrees(m)

    (_, l_cd), (_, t_cd) = get_matrix_terms(m, sigma)
    cds = l_cd - t_cd
    cd = [max(cds[:, c]) for c in range(cds.shape[1])]
    print("Initial Column degrees:")
    sp.pprint(cd)
    sp.pprint([c.as_expr().evalf() for c in cd])
    init_state_dim = sum(cd)
    print(f"Initial state dimension: {init_state_dim} ({init_state_dim.evalf()})")

    print("Coupling matrix Q1:")
    sp.pprint(q1_mat)
    print("After preliminary trafo:")
    q1_mat = DomainMatrix.from_Matrix(q1_mat).to_field()
    L, U, p = q1_mat[:, ::-1].lu()
    P = DomainMatrix.from_Matrix(sp.eye(q1_mat.shape[0]).permuteFwd(p))
    q1b = L.inv() * P * q1_mat
    sp.pprint(q1b.to_Matrix())
    m = L.inv() * P * m
    show_matrix_degree(m)
    show_matrix_degrees(m)

    l, m, r = canonical_form(m, s, sigma)
    print("In canonical form:")
    show_matrix_degree(m)
    show_matrix_degrees(m)

    # the actual function call
    mrg, lg = reduce_matrix_shifts(m, s, sigma)
    sp.pprint(lg.to_Matrix())

    print("Reduction results:")
    show_matrix_degree(mrg)
    show_matrix_degrees(mrg)

    print("Checking results:")
    # 1) all elements should have same degrees in each column
    (_, l_cd), (_, t_cd) = get_matrix_terms(mrg, sigma)
    cds = l_cd - t_cd
    cd = [max([cds[r, c] for r in range(cds.shape[0])]) for c in range(cds.shape[1])]
    print("New column degrees:")
    sp.pprint(cd)
    new_state_dim = sum(cd)
    print(f"New state dimension {new_state_dim} ({new_state_dim.evalf()})")
    dim_err = new_state_dim - des_state_dim
    print(f"dimension error: {dim_err} ({dim_err.evalf()})")
    assert dim_err <= 0

    # 2) leading coefficient matrices wrt s and sigma must commute and have full rank
    lcms = get_lccm(mrg, s)
    lcme = get_lccm(mrg, sigma)
    lcmse = get_lccm(lcms, sigma)
    lcmes = get_lccm(lcme, s)
    assert lcmes == lcmse
    assert lcmes.rank() == m.shape[0]


def test_reduce_matrix_shifts_2x2():
    print("2 x 2 cases:")
    cs = []

    if 1:
        case = (
            Matrix([[5, 1], [-(s**2) + s + 4, -3]]),
            Matrix([[0, 0], [0, 0]]),
            Matrix([[0, 0], [0, 0]]),
            Matrix([[-3, -1], [1, 8]]),
            Matrix([[4, -8], [1, 10]]),
            [sqrt(3), 3],
            [-sqrt(5), -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[-s, -3], [3 * s, 4 * s + 4]]),
            Matrix([[0, 0], [0, 0]]),
            Matrix([[0, 0], [0, 0]]),
            Matrix([[-3, -8], [4, 9]]),
            Matrix([[10, 8], [5, 5]]),
            [sqrt(5), sqrt(11)],
            [-sqrt(5), -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[4 - s, -3], [4, -5]]),
            Matrix([[0, 0]]),
            Matrix([[0], [0]]),
            Matrix([[2, -7], [10, 3]]),
            Matrix([[1, 9], [-4, -3]]),
            [sqrt(5), 3],
            [-sqrt(7), -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[3 - 3 * s, -4], [-3 * s, 4]]),
            Matrix([[0, 0]]),
            Matrix([[0], [0]]),
            Matrix([[0, -1], [-10, -4]]),
            Matrix([[9, -10], [10, 8]]),
            [sqrt(7), sp.Integer(3)],
            [-sqrt(5), -sp.Integer(3)],
        )
        cs.append(case)
    if 0:
        # pathological case: Due to the nearly similar predictions,
        # this will need several internal steps to reduce one entry
        # with sympys current derivative computation speed, this takes quite long.
        case = (
            Matrix([[-4, -5 * s**2 + 5 * s - 3], [-5 * s - 3, -4 * s**2 + s + 2]]),
            Matrix([[0, 0], [0, 0], [0, 0]]),
            Matrix([[0, 0, 0], [0, 0, 0]]),
            Matrix([[-1, 5], [-3, 0]]),
            Matrix([[5, 2], [0, 3]]),
            [sqrt(159), sqrt(163)],
            [-sqrt(15), -sqrt(67)],
        )
        cs.append(case)

    # add some random ones, too
    for d in range(1, 2):
        for k in range(3):
            cs.append(_generate_test_case(d, 2, 2))

    for c in cs:
        _run_reduction_test(c)


@pytest.mark.skip("Extensive Benchmark, takes quite long")
def test_reduce_matrix_shifts_3x2():
    print("3 x 2 cases:")
    cs = []

    if 1:
        # this one takes particularly long
        case = (
            Matrix([[1, 3 * s + 2, -4], [5, -2, -2], [-5, 4 - s, 2]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0]]),
            Matrix([[6, 3, 10], [-10, -5, -4]]),
            Matrix([[5, 7], [10, -8], [5, 1]]),
            [sqrt(5), sqrt(7), sqrt(11)],
            [-sqrt(5), -3],
        )
        cs.append(case)
    for k in range(10):
        cs.append(_generate_test_case(1, 3, 2))
        cs.append(_generate_test_case(2, 3, 2))

    for c in cs:
        _run_reduction_test(c)


@pytest.mark.skip("Extensive Benchmark, takes quite long")
def test_reduce_matrix_shifts_3x3():
    print("3 x 3 cases:")
    cs = []

    if 1:
        case = (
            Matrix([[4 * s - 1, s + 1, -5], [-1, -s - 4, -3], [3, 2 - 4 * s, -1]]),
            Matrix([[0, 0, 0], [0, 0, 0]]),
            Matrix([[0, 0], [0, 0], [0, 0]]),
            Matrix([[-7, -9, 8], [3, 6, -8], [-6, 0, 4]]),
            Matrix([[3, 7, 2], [4, -3, -2], [2, 0, 2]]),
            [sqrt(5), sqrt(7), Integer(3)],
            [-sqrt(3), -sqrt(7), -Integer(3)],
        )
        cs.append(case)
        case = (
            Matrix(
                [[-5, 3, 4], [-1, 4, -3 * s**2 + s - 2], [2, -3, -3 * s**2 + 5 * s - 4]]
            ),
            Matrix([[0, 0, 0], [0, 0, 0]]),
            Matrix([[0, 0], [0, 0], [0, 0]]),
            Matrix([[6, 6, 10], [1, -4, 7], [-1, -1, 8]]),
            Matrix([[-1, 4, 10], [5, -5, 2], [-4, 1, 5]]),
            [sqrt(3), sqrt(7), 3],
            [-sqrt(3), -3, -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[-3, -1, -2], [1, -5, 3 * s + 4], [-5, 2, 5 - s]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0], [0]]),
            Matrix([[5, -2, -5], [8, 0, -5], [3, -5, -2]]),
            Matrix([[3, 1, -10], [-6, -4, 3], [-5, -9, -4]]),
            [sqrt(3), sqrt(7), 3],
            [-sqrt(5), -sqrt(7), -sqrt(11)],
        )
        cs.append(case)
        # highest degree wrt s not in highest coeff wrt sigma in orig G mat
        case = (
            Matrix([[-4, 2, 5], [-3, 4, 1], [5 * s + 3, -4, 1]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0], [0]]),
            Matrix([[9, 1, 10], [-4, -2, 1], [5, 0, -10]]),
            Matrix([[4, -4, 10], [2, 0, -10], [1, -8, 7]]),
            [sqrt(3), sqrt(7), sqrt(21)],
            [-sqrt(7), -sqrt(13), -sqrt(19)],
        )
        cs.append(case)
        case = (
            Matrix([[-5, 5, -4], [-2, 5, -4], [-4, -2, -5]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0], [0]]),
            Matrix([[1, 1, -6], [-1, -2, 1], [1, 4, -5]]),
            Matrix([[0, -7, 5], [4, -7, 7], [3, 7, -4]]),
            [sqrt(5), sqrt(7), sqrt(11)],
            [-sqrt(5), -3, -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[-4, 2 * s - 5, -5], [-3, -3, -5], [-5, -2, -3]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0], [0]]),
            Matrix([[-1, 2, 2], [-2, 10, -10], [-8, 10, 1]]),
            Matrix([[6, 6, -5], [-9, 8, -10], [8, -4, -7]]),
            [sqrt(5), sqrt(7), sqrt(11)],
            [-sqrt(5), -3, -sqrt(11)],
        )
        cs.append(case)
        case = (
            Matrix([[-4, -4, 4], [-5, 3, 5 * s - 2], [-3, 2, 2]]),
            Matrix([[0, 0, 0]]),
            Matrix([[0], [0], [0]]),
            Matrix([[-1, -10, -8], [2, -6, 5], [1, -5, 7]]),
            Matrix([[-7, 6, -5], [9, -5, 1], [1, 10, 9]]),
            [sqrt(3), sqrt(7), 3],
            [-sqrt(5), -sqrt(7), -sqrt(11)],
        )
        cs.append(case)

    for k in range(10):
        cs.append(_generate_test_case(8, 3, 3))

    for c in cs:
        _run_reduction_test(c)


@pytest.mark.skip("Extensive Benchmark, takes quite long")
def test_reduce_matrix_shifts_4x4():
    print("4 x 4 cases:")
    cs = []

    if 1:
        case = (
            Matrix(
                [[-5, -5, -3, -3], [-5, -5, 4, -1], [-4, -1, 2, 1], [-5, -5, -3, 2]]
            ),
            Matrix(0, 4, []),
            Matrix(4, 0, []),
            Matrix([[2, 9, -8, 9], [1, -6, 9, 0], [-1, 3, -4, 8], [-9, -6, 10, 1]]),
            Matrix([[7, -4, 4, -2], [6, 9, 8, 7], [6, -5, -10, 10], [-2, -10, -10, 3]]),
            [sqrt(3), sqrt(5), sqrt(7), sqrt(11)],
            [-sqrt(5), -sqrt(7), -3, -sqrt(11)],
        )
        cs.append(case)

    for k in range(1, 10):
        cs.append(_generate_test_case(k, 4, 4))

    for c in cs:
        _run_reduction_test(c)


def test_move_col_degrees_on_diagonal():
    mats = []

    # add manual reproducers
    mat = sp.Matrix(
        [
            [
                5 * s**2 + s * (-4 * s - 3) / 5 + 3 * s - 1,
                -(s**3) - 2 * s**2 - 3 * s - 3,
                -4 * s - 3,
            ],
            [
                -(s**3) + s * (5 * s**2 - 1) / 5 - s - 4,
                3 * s**3 - 2 * s**2 - 5 * s - 3,
                5 * s**2 - 1,
            ],
            [s * (3 * s + 3) / 5 + 5 * s - 3, 4 * s**2 + 2 * s - 2, 3 * s + 3],
        ]
    )
    mats.append(mat)

    # add some random ones
    for k in range(1, 5):
        rand_mat = random_col_reduced_mat(k, k, k, -10, 10, s)
        mats.append(rand_mat)

    def _run_test(_m):
        print(_m)
        res, r_mat = move_col_degrees_on_diagonal(_m, s, sigma)
        assert check_col_degrees_on_diagonal(res, s)

    coeff_dom = sp.EX.frac_field(s)
    qp_ring = QuasiPolynomialRing(coeff_dom, sigma)
    for m in mats:
        dm = DomainMatrix.from_list(m.tolist(), domain=qp_ring)
        _run_test(dm)
