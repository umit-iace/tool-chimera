import random as rd

import pytest
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from chimera import QuasiPolynomialElement, QuasiPolynomialRing
from chimera.utils import (
    check_prediction_condition,
    check_delay_condition,
    generate_integer_transform,
    get_term,
    get_leading_column_degrees,
    get_leading_term_matrix,
    get_lccm,
    is_col_reduced,
    factor_diag_pre,
    factor_diag_post,
)
from chimera.poly import column_reduced_form


def _build_poly(p_gen, p_order, c_max):
    l_deg = rd.randint(0, p_order)
    t_deg = rd.randint(0, l_deg)
    cffs = [rd.randint(1, c_max) if i <= l_deg - t_deg else 0 for i in range(l_deg + 1)]
    poly = sp.Poly.from_list(cffs, p_gen)
    lc = cffs[0]
    ld = l_deg
    tc = cffs[l_deg - t_deg]
    td = t_deg
    return poly, (lc, ld), (tc, td)


def _build_rational(r_gen, p_order, c_max):
    npoly, (nlc, nld), (ntc, ntd) = _build_poly(r_gen, p_order, c_max)
    dpoly, (dlc, dld), (dtc, dtd) = _build_poly(r_gen, p_order, c_max)
    rat = (npoly / dpoly).cancel()
    lc = sp.Rational(nlc, dlc)
    ld = nld - dld
    tc = sp.Rational(ntc, dtc)
    td = ntd - dtd
    return rat, (lc, ld), (tc, td)


def _build_qpoly(c_gen, q_gen, c_order, cc_max):
    refs = {}
    qd = {}
    iqd = {}
    for i in range(rd.randint(1, 10)):
        exp = rd.randint(-100, 100)
        if exp in qd:
            continue
        coeff, *terms = _build_rational(c_gen, c_order, cc_max)
        qd[exp] = coeff
        iqd[coeff] = exp
        refs[coeff] = terms

    ldc = max(t[0][1] for t in refs.values())
    lcc = sum([t[0][0] * q_gen ** iqd[c] for c, t in refs.items() if t[0][1] == ldc])

    tdc = min(t[1][1] for t in refs.values())
    tcc = sum([t[1][0] * q_gen ** iqd[c] for c, t in refs.items() if t[1][1] == tdc])

    ldq = max([e for e in qd.keys()])
    lcq = qd[ldq]
    tdq = min([e for e in qd.keys()])
    tcq = qd[tdq]

    return qd, ((lcc, ldc), (tcc, tdc)), ((lcq, ldq), (tcq, tdq))


def test_get_term_poly():
    s = sp.symbols("s")
    for k in range(100):
        poly, (lc, ld), (tc, td) = _build_poly(s, 100, 100)
        (_lc, _ld), (_tc, _td) = get_term(poly, s)
        assert lc == _lc
        assert ld == _ld
        assert tc == _tc
        assert td == _td


def test_get_term_rational():
    s = sp.symbols("s")
    for k in range(100):
        rat, (lc, ld), (tc, td) = _build_rational(s, 100, 100)
        (_lc, _ld), (_tc, _td) = get_term(rat, s)
        assert lc == _lc
        assert ld == _ld
        assert tc == _tc
        assert td == _td


def test_get_term_qpoly():
    s, sigma = sp.symbols("s sigma")
    cases = []

    if 1:
        # failure for negative leading coeff degrees
        c = (
            sigma ** 61 * (16 * s ** 2 + 67 * s + 92) / (
                        14 * s ** 5 + 91 * s ** 4 + 43 * s ** 3 + 70 * s ** 2) + sigma ** 24 * (38 * s + 42) / (
                        50 * s ** 3 + 9 * s ** 2 + 84 * s),
            ((19 * sigma ** 24 / 25, -2), (46 * sigma ** 61 / 35, -2)),
            (((16 * s ** 2 + 67 * s + 92) / (14 * s ** 5 + 91 * s ** 4 + 43 * s ** 3 + 70 * s ** 2), 61),
             ((38 * s + 42) / (50 * s ** 3 + 9 * s ** 2 + 84 * s), 24)),
        )
        cases.append(c)

    for k in range(100):
        cases.append(_build_qpoly(s, sigma, 10, 100))

    for c in cases:
        qp, ((lcc, ldc), (tcc, tdc)), ((lcq, ldq), (tcq, tdq)) = c
        print("-" * 50)
        print("(")
        print(f"{c[0]},")
        print(f"{c[1]},")
        print(f"{c[2]},")
        print(")")
        print("-" * 50)

        coeff_dom = sp.EX.frac_field(s)
        qp_ring = QuasiPolynomialRing(coeff_dom, sigma)
        qp = qp_ring(qp)
        tests = [qp.as_expr(), qp]

        for _qp in tests:
            # exponents
            (_lc, _ld), (_tc, _td) = get_term(_qp, sigma)
            assert ldq == _ld
            assert tdq == _td
            assert (lcq - _lc.as_expr()).simplify() == 0
            assert (tcq - _tc.as_expr()).simplify() == 0

            # coefficients
            (_lc, _ld), (_tc, _td) = get_term(_qp, s)
            assert ldc == _ld
            assert tdc == _td
            assert (lcc - _lc.as_expr()).simplify() == 0
            assert (tcc - _tc.as_expr()).simplify() == 0


def test_get_lccm():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    m = sp.Matrix([[s, s**2], [1 / (2 - 2 * s), (s - 2) / (s + 2)]])
    lccm = get_lccm(m, s)
    print(lccm)
    assert lccm[0, 0] == 1
    assert lccm[0, 1] == 1
    assert lccm[1, 1] == 0
    assert lccm[1, 1] == 0


def test_get_leading_terms():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    m = sp.Matrix([[2 * s**3 + 2 * s + 5, s**2], [1 / (2 - 2 * s), (s - 2) / (s + 2)]])
    c_ans = sp.Matrix([[2, 1], [-sp.Rational(1, 2), 1]])
    d_ans = sp.Matrix([[3, 2], [-1, 0]])
    c, d = get_leading_term_matrix(m, s)
    assert c == c_ans
    assert d == d_ans

    m = sp.Matrix(
        [
            [2 * sigma ** sp.sqrt(3) + 2 * sigma ** (-sp.sqrt(3)) + 5, sigma**sp.pi],
            [
                (1 - sigma ** (2 * sp.pi)) / (2 * sigma**sp.pi),
                (sigma ** sp.sqrt(3) - 2) / (sigma ** sp.sqrt(3)),
            ],
        ]
    )
    coeff_dom = sp.EX.frac_field(s)
    qp_ring = QuasiPolynomialRing(coeff_dom, sigma)
    m = DomainMatrix.from_list(m.tolist(), domain=qp_ring)

    ans = sp.Matrix([[2, 1], [-sp.Rational(1, 2), 1]])
    assert get_leading_term_matrix(m, sigma)[0] == ans


def test_conditions():
    e, d, s = sp.symbols("epsilon delta s")

    # not sorted
    m = sp.Matrix([[e**3, e**2], [1, 1]])
    assert check_prediction_condition(m, e, s) == False
    # not reduced in s and e
    m = sp.Matrix([[e * s**2, s * e**2], [e, s**2]])
    assert check_prediction_condition(m, e, s) == False
    # valid
    m = sp.Matrix([[e * s**2, s * e], [e * s, e**2 * s**2]])
    assert check_prediction_condition(m, e, s) == True

    # valid
    m = sp.Matrix([[d**2, 1], [d, d]])
    assert check_delay_condition(m, d, s) == True


def test_get_leading_column_degrees():
    s = sp.symbols("s")
    m = sp.Matrix(
        [
            [s**3 + 1, s**4 - 3 * s**3 + 5 * s**2 + 7, s**5 + 17],
            [0, s**2 + 1, 1],
            [s**2, s**2, s**2],
        ]
    )
    r = [3, 4, 5]
    c_degs = get_leading_column_degrees(m, s)
    assert r == c_degs


@pytest.mark.skip("Takes too long")
def test_column_reduced_form():
    s, alpha, beta, m = sp.symbols("s alpha beta m")
    exp = sp.exp
    Matrix = sp.Matrix
    if 1:
        mat = Matrix(
            [
                [
                    s * (4 * m**2 - m * s - 2) / m**2,
                    (
                        m * s
                        - (-4 * m**2 + m * s + 2)
                        * exp(2 * (alpha - beta) * (2 * m**2 - 1) / m)
                    )
                    * exp(-2 * (alpha - beta) * (2 * m**2 - 1) / m)
                    / (4 * m**2 * (2 * m**2 - 1)),
                ],
                [0, -1 / (2 * m**2)],
            ]
        )
    else:
        mat = sp.Matrix(
            [
                [s**3 + 1, s**4 - 3 * s**3 + 5 * s**2 + 7, s**5 + 17],
                [0, s**2 + 1, 1],
                [s**2, s**2, s**2],
            ]
        )
    mat_cr, u = column_reduced_form(mat, s)
    # works as expected
    err = mat @ u - mat_cr
    err.simplify()
    assert err == sp.zeros(*mat.shape)
    # is unimoldular
    assert not u.det().free_symbols
    # yields the desired outcome
    assert is_col_reduced(mat_cr, s)


@pytest.mark.skip("Takes too long")
def test_generate_transform():
    # tests
    Arnf = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [-1, -2, -1, 3, 1],
            [0, 0, 0, 0, 1],
            [1, 3, -3, 1, -1],
        ]
    )
    Brnf = np.array([[0, 0], [0, 0], [1, 1], [0, 0], [0, 1]])

    print("Starting with system in MIMO-CCF:")
    print(Arnf)
    print(Brnf)

    T, Tinv = generate_integer_transform(Arnf.shape[0])
    print(T)

    A = Tinv @ Arnf @ T
    B = Tinv @ Brnf
    print("Resulting 'original' system:")
    print(A)
    print(B)

    b1 = B[:, 0:1]
    b2 = B[:, 1:]
    S = (
        np.round(
            np.hstack((B, A @ B, A @ A @ B, A @ A @ A @ B, A @ A @ A @ A @ B)).astype(
                np.int64
            )
            * 1e5
        )
        / 1e5
    )
    Sn = np.round(np.hstack((b1, A @ b1, A @ A @ b1, b2, A @ b2)) * 1e5) / 1e5

    Sinv = np.round(np.linalg.inv(Sn) * 1e5) / 1e5
    A = np.round(A, 8)
    B = np.round(B, 8)

    q11 = Sinv[2, :]
    q12 = Sinv[2, :] @ A
    q13 = Sinv[2, :] @ A @ A
    q21 = Sinv[4, :]
    q22 = Sinv[4, :] @ A
    Q = np.vstack((q11, q12, q13, q21, q22))


def test_factor_diag():
    h_mat = sp.MatrixSymbol("H", 5, 5).as_explicit()
    del_mat = sp.Matrix.diag(sp.symbols("d_{:5}"))

    # test it
    r = del_mat @ h_mat
    f = factor_diag_pre(r, del_mat)
    assert del_mat @ f - r == sp.Matrix.zeros(*h_mat.shape)

    f = factor_diag_post(r, del_mat)
    assert f @ del_mat - r == sp.Matrix.zeros(*h_mat.shape)


def test_is_col_reduced():
    s = sp.Symbol("s")
    m = sp.Matrix([[0, -1, 3], [0, -4, -2], [0, -1, 3]])
    assert is_col_reduced(m, s) == False
