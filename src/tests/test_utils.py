import pytest
import numpy as np
import sympy as sp
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.rings import PolyElement
from sympy.polys.domains import EX

from chimera import GenPoly
from chimera.utils import (
    check_prediction_condition,
    check_delay_condition,
    generate_integer_transform,
    get_coeffs,
    get_leading_column_degrees,
    get_leading_term_matrix,
    get_lccm,
    is_col_reduced, factor_diag_pre, factor_diag_post,
)
from chimera.poly import column_reduced_form
from chimera.genpoly import construct_fraction_field


def check_dicts(d1, d2):
    for k, v in d1.items():
        assert k in d2
        v2 = d2[k]
        if isinstance(v2, (PolyElement, FracElement)):
            v2 = v2.as_expr()
        assert v - v2 == 0

def test_get_coeff():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    sig_dict = {a: 4 * s + 2, b: 3 * s, -2 * a: 5 * s**2, 2 * b: 3 * s}
    f = GenPoly.from_dict(sig_dict, sigma, s).as_expr()
    c = get_coeffs(f, sigma)
    check_dicts(sig_dict, c)

    f = sigma ** (a) + 5 * s * sigma ** (-a)
    check_dicts(get_coeffs(f, sigma), {a: 1, -a: 5 * s})

    f = (1 / sigma) ** (a) + 5 * s * (1 / sigma) ** (-a)
    check_dicts(get_coeffs(f, sigma), {-a: 1, a: 5 * s})

    f = (sigma) ** (-b) * a * (1 / sigma) ** (-a)
    check_dicts(get_coeffs(f, sigma), {a - b: a})

    f = sp.exp(s * a) + 2 * sp.exp(s * -a)
    check_dicts(get_coeffs(f, sp.exp(s)), {a: 1, -a: 2})

    c1, c2, c3, d1, d2, d3, tau_a, tau_b = sp.symbols(
        "c1, c2, c3, d1 d2 d3 tau_a tau_b"
    )
    f = (
        -d1
        * (-c1 * s / tau_a - c2 * s + s**2 / tau_a)
        * sp.exp(s) ** tau_a
        / (
            tau_b
            * (
                c1 * d2 * s / tau_a
                - c1 * s**2 / (tau_a * tau_b)
                + c2 * d2 * s
                - c2 * s**2 / tau_b
                - d2 * s**2 / tau_a
                + s**3 / (tau_a * tau_b)
            )
            * sp.exp(s) ** tau_b
        )
    )
    cffs = get_coeffs(f, sp.exp(s))
    assert [*cffs.keys()] == [tau_a - tau_b]

    f = (
        d1
        * s
        * (
            -(sigma**tau_b)
            * (sigma**tau_a - sigma**tau_b * sp.exp(d2 * tau_b) ** (tau_a - tau_b))
            * (
                sigma ** (2 * tau_b) * (d2 * tau_b + s) * (d3 * tau_b - 1)
                - (d2 * tau_b - s) * (d3 * tau_b + 1)
            )
            - sigma ** (tau_a + tau_b)
            * (d2 * tau_b - s)
            * (d3 * tau_b + sigma ** (2 * tau_b) * (d3 * tau_b - 1) + 1)
        )
    )
    cffs = get_coeffs(f, sigma)
    assert set(cffs.keys()) == set([2 * tau_b, 4 * tau_b, tau_a + 3 * tau_b])

    f = 1 / (2 * c2 * c3 * sigma - 2 * c2 * sigma)
    cffs = get_coeffs(f, sigma)


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
    if 0:
        Rp, _ = sp.ring(s.name, EX)
        mat = DomainMatrix(mat.tolist(), mat.shape, domain=Rp)

    mat_cr, u = column_reduced_form(mat, s)
    # works as expected
    err = mat @ u - mat_cr
    err.simplify()
    assert err == sp.zeros(*mat.shape)
    # is unimoldular
    assert not u.det().free_symbols
    # yields the desired outcome
    assert is_col_reduced(mat_cr, s)


@pytest.mark.skip("Not related any more")
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
    h_mat = sp.MatrixSymbol("H", 5 ,5).as_explicit()
    del_mat = sp.Matrix.diag(sp.symbols("d_{:5}"))

    # test it
    r = del_mat @ h_mat
    f = factor_diag_pre(r, del_mat)
    assert del_mat @ f - r == sp.Matrix.zeros(*h_mat.shape)

    f = factor_diag_post(r, del_mat)
    assert f @ del_mat - r == sp.Matrix.zeros(*h_mat.shape)
