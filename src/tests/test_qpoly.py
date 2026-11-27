import pytest

import sympy as sp
from sympy import ZZ, QQ, RR, EX
from sympy.polys.fields import FracElement
from sympy.polys.rings import PolyElement

from chimera.qpoly import QuasiPolynomialElement, QuasiPolynomialRing, CachedPoly, hermite_interpolator

# generators
sigma = sp.Symbol("sigma", positive=True)
s = sp.Symbol("s")

# fields and rings
cd = EX.poly_ring(s)
qr = QuasiPolynomialRing(cd, sigma)

# some symbols
a = sp.sqrt(2)
b = sp.pi

def case_0():
    # positive integer exponents
    f = sp.Integer(0)
    g = sp.Integer(666)
    print("Case 0:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_1():
    # positive integer exponents
    f = sigma**3 + 5 + sigma**2 * 10
    g = sigma**2 + 10 * sigma + 6
    print("Case 1:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_2():
    # positive rational exponents
    f = sigma ** sp.Rational(16, 5) + 5 + sigma ** sp.Rational(14, 5) * 10
    g = sigma ** (2 + sp.Rational(1, 3)) + 10 * sigma + 6
    print("Case 2:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_3():
    # negative integer exponents and polynomial coefficients
    f = sp.Poly(s**2 + s, s) / (2 * sigma) + sigma ** (-2)
    g = sp.Poly(s / 2, s)
    print("Case 3:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_4():
    # compound exponents
    f = (
        (s**2 + 2 * s + 1) * sigma ** (a + b) + (s**3) * sigma**-b
    ) * sigma**a
    g = (s**2 + 2 * s + 1) * sigma**-a + (s**3) * sigma**b
    print("Case 4:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_5():
    # rational coefficients and real exponents
    f = (
        (s**2 + 2 * s + 1) / (s**3 - s) * sigma ** (a + b)
        + (s**3) * sigma**-b
    ) * sigma**a
    g = (
        (s**2 + 2 * s + 1) * sigma**-a
        + (s**3 + 1 * s) / (s**2 + 2) * sigma**b
    )
    print("Case 5:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_6():
    f = (3 * s + 2 * s**2) * sigma ** sp.sqrt(5) + (s**2 + 2 * s) * sigma + 2 * s + 1
    g = (3 * s**4 + 2 * s**2) * sigma ** sp.sqrt(2) + (s**2 + 2 * s) * sigma + 2 * s + 1
    print("Case 6:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_7():
    expr = sigma**(-3 - sp.sqrt(5) + 2*sp.sqrt(7))*(-36*s**2/5 + 72*s/5 - 36/5)/(9*s)
    return expr,


def test_normalize_denominator():
    numer = {
        1: cd(2 * s**2 + 2*s + 2),
        2: cd(4 * s ** 5 + 4 * s**3 + 4 * s**2),
    }
    denom = cd(2*(s-2))
    n,d = QuasiPolynomialElement.normalize_denominator(cd, numer, denom)
    assert d.LC == 1
    assert n[1].LC == 1
    assert n[2].LC == 2


def test_normalize_fraction():
    # no elimination
    numer = {1: cd(2 * (s ** 2 + 1) * (s + 3))}
    denom = cd(2*(s-2))
    n,d = QuasiPolynomialElement.normalize_fraction(cd, numer, denom)

    assert n[1].degree() == 3
    assert n[1].LC == 1
    assert d.LC == 1

    # elimination
    numer = {2: cd(2 * (s + 2) * (s - 2))}
    denom = cd(2*(s-2))
    n,d = QuasiPolynomialElement.normalize_fraction(cd, numer, denom)

    assert n[2].degree() == 1
    assert n[2].LC == 1
    assert d.LC == 1


def test_from_expr():
    for c in [case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7]:
        for expr in c():
            q = QuasiPolynomialElement.from_expr(qr, expr)
            err = expr - q.as_expr()
            err = err.simplify()
            assert err == 0


def test_dict_from_expr():
    for c in [case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7]:
        for expr in c():
            s_dict = QuasiPolynomialElement.dict_from_expr(expr, qr)
            print(s_dict)
            test_expr = QuasiPolynomialElement.dict_to_expr(s_dict, *qr.symbols)
            print(test_expr)
            err = expr - test_expr
            err = err.simplify()
            assert err == 0


def test_add():
    f = QuasiPolynomialElement.from_expr(qr, 0)
    f += 1
    assert f == 1

    for c in [case_0, case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        fg = QuasiPolynomialElement.from_expr(qr, f)
        gg = QuasiPolynomialElement.from_expr(qr, g)
        rg = fg + gg
        err = f + g - rg.as_expr()
        assert err.simplify() == 0


def test_neg():
    for c in [case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7]:
        for expr in c():
            a = QuasiPolynomialElement.from_expr(qr, expr)
            am = -a
            assert am.as_expr().expand() == -a.as_expr().expand()


def test_sub():
    for c in [ case_0, case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        r = f - g

        fg = QuasiPolynomialElement.from_expr(qr, f)
        gg = QuasiPolynomialElement.from_expr(qr, g)
        rg = fg - gg

        err = r - rg.as_expr()
        assert err.simplify() == 0


def test_mul():
    for c in [case_0, case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        fg = QuasiPolynomialElement.from_expr(qr, f)
        gg = QuasiPolynomialElement.from_expr(qr, g)
        rg = fg * gg
        err = f * g - rg.as_expr()
        assert err.simplify() == 0


def test_div():
    for c in [ case_0, case_1, case_2, case_3, case_4, case_5, case_6]:
        f, g = c()
        fg = QuasiPolynomialElement.from_expr(qr, f)
        gg = QuasiPolynomialElement.from_expr(qr, g)
        glm = gg.LM()
        rg = fg / glm
        err = fg - glm * rg
        assert err.as_expr().simplify() == 0

        quo, rem = fg.__divmod__(gg)
        err = f - quo.as_expr() * g - rem.as_expr()
        assert err.simplify() == 0


def test_as_numer_denom():
    for c in [case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7]:
        for expr in c():
            xp = QuasiPolynomialElement.from_expr(qr, expr)
            num, den = xp.as_numer_denom()
            assert isinstance(num, QuasiPolynomialElement)
            assert num.denom == cd.one
            assert cd.of_type(den)
            err = num.as_expr() / den.as_expr() - expr
            assert err.simplify() == 0


@pytest.mark.skip(reason="Not implemented yet")
def test_diff():
    for c in [
        case_0, case_1, case_2, case_3, case_4,
        # case_5,
        case_6
    ]:
        for expr in c():
            expr_deriv = expr.subs(sigma, sp.exp(s))
            qp_deriv = QuasiPolynomialElement.from_expr(qr, expr)
            for k in range(5):
                expr_deriv = expr_deriv.diff(s)
                qp_deriv = qp_deriv.diff(s)

                err = expr_deriv - qp_deriv.as_expr().subs(sigma, sp.exp(s))
                assert err.simplify() == 0


def test_cached_poly():
    R = sp.PolynomialRing(sp.QQ, "x")
    x = R.symbols[0]
    p = R(4*x**3+2*x**2+3*x+17)
    sp.pprint(p)

    cp = CachedPoly(p)
    assert cp == p
    assert p.factor_list() == cp.factor_list()

    h = hash(cp)

    # manipulating the original poly should not matter
    p *= R(x)
    assert h == hash(cp)

    # manipulating the poly invalidates the factor list
    cp *= R(x)
    assert h != hash(cp)
    assert cp._factors is None
    # the entries should be updated on the next call
    assert p.factor_list() == cp.factor_list()


def test_hermite_interpolator():
    f, _ = case_6()
    roots = {
        1: 1,
        2: 1,
        3: 1,
    }
    fq = QuasiPolynomialElement.from_expr(qr, f)
    hp = hermite_interpolator(fq, roots)
    h = hp.as_expr()
    sp.pprint(hp, wrap_line=False)
    for r,k in roots.items():
        for j in range(k-1):
            assert j == 0
            assert f.diff(s,j).subs(s,r) == hp.diff(s).subs(s,r)

    sp.plot(f.subs(sigma, sp.exp(s)), (s, 0,5))
    sp.plot(h, (s, 0, 5))
