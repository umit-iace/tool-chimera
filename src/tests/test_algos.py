import random

import sympy as sp
import pytest

from chimera import QPoly
from chimera.algos import (
    reduce_matrix_shifts,
    qpld,
    make_entire,
    qplde,
    column_reduced_form,
    canonical_form,
    move_col_degrees_on_diagonal,
    check_col_degrees_on_diagonal,
)
from chimera.utils import (
    get_leading_column_degrees,
    get_lccm,
    check_singularities,
    is_col_reduced,
    random_poly_mat,
    get_matrix_terms,
    random_partition_m_parts
)

# generators
sigma = sp.Symbol("sigma", positive=True)
s = sp.Symbol("s")


def test_gpld():
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
        qp, rp = qpld(xp, yp, sigma)
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
        2 * s / (3 * s**2 - 5 * s + 4) * sigma**3
        + (s**4 + 1) / s**2 * (s**2 - 2) * sigma,
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
            err = (gp.as_dict()[exp] - cff).simplify()
            assert err == 0


# @pytest.mark.skip("Takes too long")
def test_gplde():
    xl = [
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
        print(f"Dividing {x} by {y}:")
        xp = QPoly(x, sigma)
        yp = QPoly(y, sigma)
        qep, rep = qplde(xp, yp, s)
        qe = qep.as_expr()
        re = rep.as_expr()
        # print(f"q: {qe}")
        # print(f"r: {re}")

        # check degrees
        assert rep.degmax() <= yp.degmax()
        assert rep.degree() <= yp.degree()

        # check residual
        err = x - qe * y - re
        err = err.simplify()
        print(f"error: {err}")
        assert err == 0

        # TODO speed this check up so that we can actually test it
        # print("Checking for singularities in remainder:")
        # check_singularities(rep, s, sigma, sp.exp(s), verbose=True)


def test_reduce_matrix_shifts():
    n = 1
    nm = 3
    np = 3

    rand = False
    rand = True

    # ode part
    print("ODE parametrisation matrices D and N:")
    if rand:
        print("Generating random matrix")
        cds = random_partition_m_parts(n, nm)
        assert len(cds) == nm
        assert sum(cds) == n
        d_mat = random_poly_mat(nm, nm, cds, -5, 5, s)
        assert get_leading_column_degrees(d_mat, s) == cds
        d_mat, _ = column_reduced_form(d_mat, s)
    elif nm == 2:
        d_mat = sp.Matrix([[4*s + 5, 1 - 5*s], [3*s + 4, -4*s**2 + 5*s - 5]])
    elif nm == 3:
        d_mat = sp.Matrix([[-1, 3 * s - 1, -5], [-5, 4, 1], [-1, -1, 5]])
    else:
        d_mat = sp.Matrix(
            [
                [
                    -3 * s**2 + 4 * s - 3,
                    5,
                    -2 * s**3
                    - 15 * s**2 / 2
                    - 2 * s * (-3 * s**2 + 4 * s - 3) / 3
                    + 5 * s
                    + 2,
                ],
                [
                    -3,
                    -2 * s - 3,
                    -(s**3) - s**2 * (-2 * s - 3) / 2 - 2 * s**2 - 2 * s - 2,
                ],
                [-2, 4, -2 * s**2 + 4 * s / 3 - 1],
            ]
        )
    assert is_col_reduced(d_mat, s)
    n_mat = sp.Matrix.zeros(n, nm)
    print(d_mat)
    print(n_mat)

    # pde part
    if rand:
        tau_m = sorted([sp.sqrt(2*random.randint(1, 100)+1) for n in range(nm)])
        tau_p = sorted([-sp.sqrt(2*random.randint(1, 100)+1) for n in range(np)])[::-1]
    else:
        if nm == 3 and np == 2:
            tau_m = [sp.sqrt(33), sp.sqrt(123), sp.sqrt(161)]
            tau_p = [-sp.sqrt(91), -sp.sqrt(161)]
        else:
            pass

    pred_mat = sp.Matrix.diag([sigma**tau for tau in tau_m])
    delay_mat = sp.Matrix.diag([sigma**tau for tau in tau_p])
    des_state_dim = (sp.Add(*tau_m) - sp.Add(*tau_p)).simplify()
    print(f"System predictions: {tau_m}")
    print(f"System delays: {tau_p}")
    print(f"Original state dimensions {[tm - tp for tm,tp in zip(tau_m, tau_p)]}")
    print(f"Original full state dim: {des_state_dim} ({des_state_dim.evalf()})")

    print("PDE boundary matrices Q0, Q1 and C")
    if rand:
        q0_mat = sp.Matrix.zeros(np, nm)
        while q0_mat.rank() < np:
            q0_mat = sp.randMatrix(np, nm, -10, 10)
        q1_mat = sp.Matrix.zeros(nm, np)
        while q1_mat.rank() < np:
            q1_mat = sp.randMatrix(nm, np, -10, 10)
    elif nm == np == 2:
        # 2x2 cases:
        q0_mat = sp.Matrix([[-10, -10], [-8, -1]])
        q1_mat = sp.Matrix([[7, 7], [-10, 8]])
    elif nm == 3 and np == 2:
        q0_mat = sp.Matrix([[-5, 9, -9], [2, -10, 5]])
        q1_mat = sp.Matrix([[4, -9], [8, 2], [9, -7]])
    c_mat = sp.Matrix.zeros(np, n)
    print(q0_mat)
    print(q1_mat)
    print(c_mat)

    print("G Matrix:")
    h_mat = q0_mat @ d_mat + c_mat @ n_mat
    m = pred_mat @ d_mat - q1_mat @ delay_mat @ h_mat
    # sp.pprint(m, wrap_line=False)
    mq = m.applyfunc(lambda x: QPoly(x, sigma))
    sp.pprint(mq.applyfunc(lambda x: x.degree()), wrap_line=False)
    sp.pprint(mq.applyfunc(lambda x: (x.degmin(), x.degmax())), wrap_line=False)

    (_, l_cd), (_, t_cd) = get_matrix_terms(m, sigma)
    cds = l_cd - t_cd
    cd = [max(cds[:, c]) for c in range(cds.shape[1])]
    print("Initial Column degrees:")
    sp.pprint(cd)
    init_state_dim = sp.Add(*cd).simplify()
    print(f"Initial state dimension: {init_state_dim}")

    print("After preliminary trafo:")
    L,U,p = q1_mat[:,::-1].LUdecomposition()
    m = L.inv() @ m
    mq = m.applyfunc(lambda x: QPoly(x, sigma))
    sp.pprint(mq.applyfunc(lambda x: x.degree()), wrap_line=False)
    sp.pprint(mq.applyfunc(lambda x: (x.degmin(), x.degmax())), wrap_line=False)

    l, m, r = canonical_form(m, s, sigma)
    print("In canonical form:")
    mq = m.applyfunc(lambda x: QPoly(x, sigma))
    sp.pprint(mq.applyfunc(lambda x: x.degree().evalf()), wrap_line=False)
    sp.pprint(mq.applyfunc(lambda x: (x.degmin().evalf(), x.degmax().evalf())), wrap_line=False)

    if 0:
        # test some ideas
        for r in range(mq.rows):
            md = mq.applyfunc(lambda x: x.degree().evalf())
            if not all([md[r,0] == md[r,c] for c in range(mq.cols)]):
                sp.pprint(md)
                sp.pprint(mq.applyfunc(lambda x: (x.degmin().evalf(), x.degmax().evalf())), wrap_line=False)
                sp.pprint(get_lccm(m, s))
                raise ValueError("Does not work")
        return

    mrg, lg = reduce_matrix_shifts(m, s, sigma)

    print("Reduction results:")
    mq = mrg.applyfunc(lambda x: QPoly(x, sigma))
    sp.pprint(mq, wrap_line=False)
    sp.pprint(mq.applyfunc(lambda x: x.degree().evalf()), wrap_line=False)
    sp.pprint(mq.applyfunc(lambda x: (x.degmin().evalf(), x.degmax().evalf())), wrap_line=False)

    print("Checking results:")
    # 1) all elements should have same degrees in each column
    (_, l_cd), (_, t_cd) = get_matrix_terms(mq, sigma)
    cds = l_cd - t_cd
    cd = [max(cds[:, c]) for c in range(cds.shape[1])]
    print("New column degrees:")
    sp.pprint(cd)
    new_state_dim = sp.Add(*cd).simplify()
    print(f"New state dimension {new_state_dim}")
    dim_err = (new_state_dim - des_state_dim).simplify()
    print(f"dimension error: {dim_err}")
    assert dim_err <= 0

    # 2) leading coefficient matrices wrt s and sigma must commute and have full rank
    lcms = get_lccm(mq, s)
    assert lcms.rank() == m.shape[0]
    lcme = get_lccm(mq, sigma)
    assert lcms.rank() == m.shape[0]
    lcmse = get_lccm(lcms, sigma)
    assert lcmse.rank() == m.shape[0]
    lcmes = get_lccm(lcme, s)
    assert lcmes.rank() == m.shape[0]
    assert lcmes == lcmse


def test_move_col_degrees_on_diagonal():
    # print()
    if 1:
        rand_mat = random_poly_mat(3, 3, 3, -5, 5, s)
        mat, _ = column_reduced_form(rand_mat, s)
        print(mat)
    else:
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
    m, r_mat = move_col_degrees_on_diagonal(mat, s)
    # sp.pprint(r_mat)
    # sp.pprint(r_mat)
    # sp.pprint(get_lccm(m, s))
    assert check_col_degrees_on_diagonal(m, s)


def test_sor_cols_random():
    for k in range(100):
        test_move_col_degrees_on_diagonal()


def test_red_shifts_random():
    for k in range(100):
        test_reduce_matrix_shifts()