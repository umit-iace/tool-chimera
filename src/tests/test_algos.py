import pickle

import sympy as sp
import pytest

from chimera import GenPoly
from chimera.algos import (
    reduce_matrix_shifts,
    gpld,
    make_entire,
    gplde,
    column_reduced_form,
    canonical_form,
    move_col_degrees_on_diagonal,
    check_col_degrees_on_diagonal,
)
from chimera.utils import (
    get_column_degrees,
    get_lccm,
    check_singularities,
    is_col_reduced,
    random_poly_mat,
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
        xp = GenPoly(x, sigma)
        yp = GenPoly(y, sigma)
        qp, rp = gpld(xp, yp, sigma)
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


@pytest.mark.skip("Takes too long")
def test_make_entire():
    exprs = [
        2 * s / s**3 * (3 * s + 4) * sigma**3 + 1 / (s**2 + 2) * sigma,
        2 * s / (3 * s**2 - 5 * s + 4) * sigma**3
        + (s**4 + 1) / s**2 * (s**2 - 2) * sigma,
    ]
    for expr in exprs:
        print(f"Testing {expr}:")
        ep = GenPoly(expr, sigma)
        e_epxr = make_entire(ep, s)
        check_singularities(e_epxr, s, sigma, sp.exp(s))

        # all coefficients should stay the same, except for the constant part
        err = expr - e_epxr
        assert err.exponents() == [0]


@pytest.mark.skip("Takes too long")
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
        xp = GenPoly(x, sigma)
        yp = GenPoly(y, sigma)
        qep, rep = gplde(xp, yp, s)
        qe = qep.as_expr()
        re = rep.as_expr()
        print(f"q: {qe}")
        print(f"r: {re}")

        # check degrees
        assert rep.degmax() <= yp.degmax()
        assert rep.degree() <= yp.degree()

        # check residual
        err = x - qe * y - re
        err = err.simplify()
        print(f"error: {err}")
        assert err == 0

        print("Checking for singularities in quotient and remainder:")
        for t in qep, rep:
            check_singularities(t, s, sigma, sp.exp(s))
            print("All good")


@pytest.mark.skip("Takes too long")
def test_reduce_matrix_shifts():
    np = 2
    nm = n = 3
    rand = False

    # ode part
    if rand:
        m = random_poly_mat(n, n, n, -5, 5, s)
        m, _ = column_reduced_form(m, s)
        d_mat, _ = move_col_degrees_on_diagonal(m, s)
        print(d_mat)
    elif nm == 2:
        d_mat = sp.Matrix(
            [
                [3 * s / 5 + 2, sp.Rational(4, 3)],
                [
                    -3 * s**2
                    + 9 * s * (20 * s / 3 + sp.Rational(19, 3)) / 20
                    - 2 * s
                    - 1,
                    20 * s / 3 + sp.Rational(19, 3),
                ],
            ]
        )
    elif nm == 3:
        d_mat = sp.Matrix(
            [
                [-4 * s**3 - 2 * s**2 + 5 * s + 1, -2 * s**2 - 2 * s + 5, 2 * s - 2],
                [2 * s - 1, 2 * s**3 - 1, 2 * s - 2],
                [-(s**2) + 4 * s + 2, 2, 3 * s - 1],
            ]
        )
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

    # pde part
    if 0:
        # non-rational exponents cause parsing errors
        tau_m = [sp.prime(2*n+1) for n in range(nm)]
        tau_p = [-sp.prime(2*n+2) for n in range(np)]
    else:
        tau_m = [sp.Rational(1,sp.prime(2*n+1)) for n in range(nm)]
        tau_p = [-sp.Rational(1,sp.prime(2*n+2)) for n in range(np)]
    pred_mat = sp.Matrix.diag([sigma**tau for tau in tau_m])
    delay_mat = sp.Matrix.diag([sigma**tau for tau in tau_p])
    des_state_dim = (sp.Add(*tau_m) - sp.Add(*tau_p)).simplify()
    print(f"Desired state dim: {des_state_dim}")

    if rand:
        q0_mat = sp.Matrix.zeros(np, nm)
        while q0_mat.rank() < np:
            q0_mat = sp.randMatrix(np, nm, -1, 1)
        print(q0_mat)
        q1_mat = sp.Matrix.zeros(nm, np)
        while q1_mat.rank() < np:
            q1_mat = sp.randMatrix(nm, np, -1, 1)
        print(q1_mat)
    elif nm == np == 2:
        # 2x2 cases:
        q0_mat = sp.Matrix([[-1, 1], [0, 1]])
        q1_mat = sp.Matrix([[0, -1], [-1, 0]])
    elif nm ==3 and np == 2:
        # q0_mat = sp.Matrix([[1, 1, 0], [0, -1, 1]])
        # q1_mat = sp.Matrix([[-1, 0], [0, -1], [-1, 1]])
        q0_mat = sp.Matrix([[-1, 0, 0], [0, -1, -1]])
        q1_mat = sp.Matrix([[-1, -1], [-1, 1], [-1, 0]])

    # g matrix
    m = pred_mat @ d_mat - q1_mat @ delay_mat @ q0_mat @ d_mat
    l, m, r = canonical_form(m, s, sigma)

    sp.pprint(m, wrap_line=False)
    sp.pprint(m.applyfunc(lambda x: GenPoly(x, sigma)), wrap_line=False)
    l_cd, t_cd = get_column_degrees(m, sigma)
    cd = [lcd - tcd for lcd, tcd in zip(l_cd, t_cd)]
    print("Initial Column degrees:")
    sp.pprint(cd)
    orig_state_dim = sp.Add(*cd).simplify()
    print(f"Initial state dimension {orig_state_dim}")
    if 0:
        try:
            with open("tmp.pkl", "rb") as f:
                mrg = pickle.load(f)
        except FileNotFoundError:
            mrg, lg = reduce_matrix_shifts(m, s, sigma)

        with open("tmp.pkl", "wb") as f:
            pickle.dump(mrg, f)
    else:
        mrg, lg = reduce_matrix_shifts(m, s, sigma)
    sp.pprint(mrg, wrap_line=False)

    print("Checking results:")

    # 1) all elements should have same degrees in each column
    l_cd, t_cd = get_column_degrees(mrg, sigma)
    cd = [lcd - tcd for lcd, tcd in zip(l_cd, t_cd)]
    print("New column degrees:")
    sp.pprint(cd)
    new_state_dim = sp.Add(*cd).simplify()
    print(f"New state dimension {new_state_dim}")
    dim_err = (new_state_dim - des_state_dim).simplify()
    print(f"dimension error: {dim_err}")
    assert dim_err == 0

    # 2) leading coefficient matrices wrt s and sigma must commute and have full rank
    lcms = get_lccm(mrg, s)
    assert lcms.rank() == m.shape[0]
    lcme = get_lccm(mrg, sigma)
    assert lcms.rank() == m.shape[0]
    lcmse = get_lccm(lcms, sigma)
    assert lcmse.rank() == m.shape[0]
    lcmes = get_lccm(lcme, s)
    assert lcmes.rank() == m.shape[0]
    assert lcmes == lcmse
    # assert check_derivative_condition(mrg, s)
    # assert check_prediction_condition(mrg, sigma)


def test_move_col_degrees_on_diagonal():
    if 1:
        rand_mat = random_poly_mat(3, 3, 3, -5, 5, s)
        mat, _ = column_reduced_form(rand_mat, s)
    else:
        mat = sp.Matrix(
            [
                [2 - 3 * s**2, 1, -5 * s**2 - 5],
                [s**3 - s**2 - 5 * s + 5, -3 * s**2 + s - 1, 3 * s**2 - s - 5],
                [s**3 - 4 * s**2 + 3 * s + 3, 1, -5 * s**2 + 3 * s + 4],
            ]
        )
    # sp.pprint(mat)
    m, r_mat = move_col_degrees_on_diagonal(mat, s)
    assert check_col_degrees_on_diagonal(m, s)
