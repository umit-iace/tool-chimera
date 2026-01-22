import pickle

import sympy as sp
import pytest

from chimera import GenPoly
from chimera.algos import reduce_matrix_shifts, gpld, make_entire, gplde
from chimera.utils import (
    get_column_degrees,
    get_lccm,
    check_singularities,
    is_col_reduced,
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
    a = 4 * s**2 + 2 * s + 7
    b = 1 * s - 10
    c = 17
    d_mat = sp.Matrix(
        [
            [a, b, c],
            [c, a, b],
            [b, c, a],
        ]
    )
    assert is_col_reduced(d_mat, s)
    tau_m = [sp.sqrt(2), sp.sqrt(5), sp.sqrt(7)]
    tau_p = [0, -sp.sqrt(3), -sp.sqrt(9)]

    pred_mat = sp.Matrix.diag([sigma**tau for tau in tau_m])
    delay_mat = sp.Matrix.diag([sigma**tau for tau in tau_p])
    m = pred_mat @ d_mat + delay_mat @ d_mat

    # gens = list(
    #     chain(*[[sigma**tau, 1 / sigma**tau] for tau in tau_m + tau_p if tau != 0])
    # )
    # gens.append(s)
    # mp = m.applyfunc(lambda x: sp.Poly(x, gens))

    print()
    sp.pprint(m, wrap_line=False)
    try:
        with open("tmp.pkl", "rb") as f:
            mrg = pickle.load(f)
    except FileNotFoundError:
        mrg, lg = reduce_matrix_shifts(m, s, sigma)

    with open("tmp.pkl", "wb") as f:
        pickle.dump(mrg, f)

    print("Checking results:")
    # 1) all elements should have same degrees in each column
    l_cd, t_cd = get_column_degrees(mrg, sigma)
    for i in range(len(tau_m)):
        assert l_cd[i] <= tau_m[i]
        assert t_cd[i] >= tau_p[i]
    print("Column degrees are sound")

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
