import sympy as sp

from pyform import get_leading_term_matrix
from pyform.utils import get_coeffs, get_matrix_coeffs, Shift, get_col_lc_mat
from pyform.signomial import Signomial


def test_get_coeff():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    sig_dict = {a: 4*s+2, b: 3*s, -2*a: 5*s**2, b+b: 3*s}
    f = Signomial.from_dict(sig_dict, sigma).to_expr()
    c = get_coeffs(f, sigma)
    assert sig_dict == c

    f = sigma**(a) + 5*s*sigma**(-a)
    assert get_coeffs(f, sigma) == {a: 1, -a: 5*s}

    f = (1/sigma)**(a) + 5*s*(1/sigma)**(-a)
    assert get_coeffs(f, sigma) == {-a: 1, a: 5*s}

    f = (sigma)**(-b) * a * (1/sigma)**(-a)
    assert get_coeffs(f, sigma) == {a-b: a}

    f = sp.exp(s*a) + 2*sp.exp(s*-a)
    assert get_coeffs(f, sp.exp(s)) == {a: 1, -a: 2}

    c1, c2, c3, d1, d2, d3, tau_a, tau_b = sp.symbols("c1, c2, c3, d1 d2 d3 tau_a tau_b")
    f = -d1 * (-c1 * s / tau_a - c2 * s + s ** 2 / tau_a) * sp.exp(s) ** tau_a / (tau_b * (
                c1 * d2 * s / tau_a - c1 * s ** 2 / (
                    tau_a * tau_b) + c2 * d2 * s - c2 * s ** 2 / tau_b - d2 * s ** 2 / tau_a + s ** 3 / (
                            tau_a * tau_b)) * sp.exp(s) ** tau_b)
    cffs = get_coeffs(f, sp.exp(s))
    assert [*cffs.keys()] == [tau_a-tau_b]

    f = d1 * s * (-sigma ** tau_b * (sigma ** tau_a - sigma ** tau_b * sp.exp(d2 * tau_b) ** (tau_a - tau_b)) * (
                sigma ** (2 * tau_b) * (d2 * tau_b + s) * (d3 * tau_b - 1) - (d2 * tau_b - s) * (
                    d3 * tau_b + 1)) - sigma ** (tau_a + tau_b) * (d2 * tau_b - s) * (
                              d3 * tau_b + sigma ** (2 * tau_b) * (d3 * tau_b - 1) + 1))
    cffs = get_coeffs(f, sigma)
    assert [*cffs.keys()] == [2*tau_b, 4*tau_b, tau_a + 3*tau_b]

    f = 1/(2*c2*c3*sigma - 2*c2*sigma)
    cffs = get_coeffs(f, sigma)


def test_shift():
    s = Shift(16)
    print(s)

def test_get_lccm():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    m = sp.Matrix([[s, s**2], [1/(2-2*s), (s-2)/(s+2)]])
    lccm = get_col_lc_mat(m, s)
    print(lccm)
    assert lccm[0,0] == 1
    assert lccm[0,1] == 1
    assert lccm[1,1] == 0
    assert lccm[1,1] == 0

def test_get_leading_terms():
    a, b, s = sp.symbols("a b s")
    sigma = sp.symbols("sigma", positive=True)

    m = sp.Matrix([[2*s**3 + 2*s + 5, s**2], [1/(2-2*s), (s-2)/(s+2)]])
    ans = sp.Matrix([[2, 1], [-sp.Rational(1,2), 1]])
    assert get_leading_term_matrix(m, s)[1] == ans

    m = sp.Matrix([[2 * sigma**sp.sqrt(3) + 2 * sigma**(-sp.sqrt(3)) + 5, sigma ** sp.pi],
                   [(1 - sigma**(2*sp.pi)) / (2 * sigma**sp.pi), (sigma**sp.sqrt(3) - 2) / (sigma**sp.sqrt(3))]])
    ans = sp.Matrix([[2, 1], [-sp.Rational(1,2), 1]])
    assert get_leading_term_matrix(m, sigma)[1] == ans
