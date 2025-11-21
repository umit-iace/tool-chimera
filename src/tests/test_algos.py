from itertools import chain

import sympy as sp

from pyform import GenPoly
from pyform.algos import reduce_matrix_shifts, gpld, make_entire, gplde
from pyform.utils import (
    get_leading_column_degrees,
    check_singularities,
    is_col_reduced,
    check_derivative_condition,
    check_prediction_condition,
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
        q, r = gpld(x, y, sigma)
        print(f"q: {q}")
        print(f"r: {r}")

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
        e_epxr = make_entire(expr, rat_sym=s, gp_gen=sigma, gen_expr=sp.exp(s))
        check_singularities(e_epxr, s, sigma, sp.exp(s))


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
        qe, re = gplde(x, y, s, sigma, sp.exp(s))
        print(f"q: {qe}")
        print(f"r: {re}")

        # check residual
        err = x - qe * y - re
        err = err.simplify()
        print(f"error: {err}")
        assert err == 0

        print("Checking for singularities in quotient and remainder:")
        for t in qe, re:
            check_singularities(t, s, sigma, sp.exp(s))


def test_reduce_matrix_shifts():
    a = 4 * s**3 + 2 * s**2 + 1 * s + 7
    b = 1 * s**2 - 10 * s - 5
    c = 3 * s - 1
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
    mrg, lg = reduce_matrix_shifts(m, s, sigma)
    # sp.pprint(mrg, wrap_line=False)
    # sp.pprint(lg, wrap_line=False)

    # check trafo
    l = lg.applyfunc(lambda x: x.as_expr())
    mr = mrg.applyfunc(lambda x: x.as_expr())
    err = l @ m - mr
    err.simplify()
    assert err == sp.Matrix.zeros(*m.shape)

    # check result properties
    # 1) all elements should have same degrees in each column
    cd = get_leading_column_degrees(mr, sigma)
    for i in range(len(tau_m)):
        assert cd[i] == tau_m[i]
    # 2) leading coefficient matrices wrt s and sigma must commute and have full rank
    assert check_derivative_condition(mr, s)
    assert check_prediction_condition(mr, sigma)


# def test_gcdexs():
#     for c in [
#         case_1,
#         # case_2,  ## this one will not occur in our systems
#         case_3,
#         # case_4,
#               ]:
#         f, g = c()
#
#         # degree comp needs actual values, no symbols
#         f = f.subs(subs_dict)
#         g = g.subs(subs_dict)
#
#         s, t, h = gcdexs(f, g, sigma, max_iterations=10)
#         print(f"s: {s}")
#         print(f"t: {t}")
#         print(f"h(gcd): {h}")
#
#         err = s*f + t*g - h
#         err = err.simplify()
#         print(f"Error: {err}")
#         assert err == 0
#
#
# def test_bezouts():
#     for c in [
#         case_1,
#         # case_2,
#         case_3,
#         # case_4,
#     ]:
#         a, b = c()
#
#         # degree comp needs actual values, no symbols
#         a = a.subs(subs_dict)
#         b = b.subs(subs_dict)
#
#         p, q = bezouts(a, b, p_gen=s, s_gen=sigma)
#         print(f"p: {p}")
#         print(f"q: {q}")
