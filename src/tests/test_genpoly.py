import sympy as sp

from chimera import GenPoly

# generators
sigma = sp.Symbol("sigma", positive=True)
s = sp.Symbol("s")

# some symbols
a, b = sp.symbols("a b")
param_values = {a: sp.sqrt(2), b: sp.pi}


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
    # negative non-integer exponents and polynomial coefficients
    f = sp.Poly(s**2 + s, s) / (2 * sigma) + sigma ** (-2)
    g = sp.Poly(s / 2, s)
    print("Case 3:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_4():
    # compound exponents
    f = (
        sp.Poly(s**2 + 2 * s + 1, s) * sigma ** (a + b) + sp.Poly(s**3, s) * sigma**-b
    ) * sigma**a
    g = sp.Poly(s**2 + 2 * s + 1, s) * sigma**-a + sp.Poly(s**3, s) * sigma**b
    print("Case 4:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_5():
    # rational coefficients and real exponents
    f = (
        sp.Poly(s**2 + 2 * s + 1, s) / sp.Poly(s**3 - s, s) * sigma ** (a + b)
        + sp.Poly(s**3, s) * sigma**-b
    ) * sigma**a
    g = (
        sp.Poly(s**2 + 2 * s + 1, s) * sigma**-a
        + sp.Poly(s**3 + 1 * s, s) / sp.Poly(s**2 + 2, s) * sigma**b
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


def test_dict_from_expr():
    for c in [case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        for expr in f, g:
            s_dict = GenPoly.dict_from_expr(expr, sigma, s)
            print(s_dict)
            test_expr = GenPoly.dict_to_expr(s_dict, sigma)
            print(test_expr)
            err = expr - test_expr
            err = err.simplify()
            assert err == 0


def test_add():
    f = GenPoly(0, sigma)
    assert f + 1 == 1

    for c in [case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        fg = GenPoly(f, sigma, s)
        gg = GenPoly(g, sigma, s)
        rg = fg + gg
        err = f + g - rg.as_expr()
        assert err.simplify() == 0


def test_neg():
    for c in [case_1, case_2, case_3, case_4, case_5]:
        for expr in c():
            g_expr = GenPoly(expr, sigma, s)
            err = expr + (-g_expr).as_expr()
            assert err.simplify() == 0


def test_sub():
    for c in [case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        fg = GenPoly(f, sigma, s)
        gg = GenPoly(g, sigma, s)
        rg = fg - gg
        err = f - g - rg.as_expr()
        assert err.simplify() == 0


def test_mul():
    for c in [case_1, case_2, case_3, case_4, case_5]:
        f, g = c()
        fg = GenPoly(f, sigma, s)
        gg = GenPoly(g, sigma, s)
        rg = fg * gg
        err = f * g - rg.as_expr()
        assert err.simplify() == 0


def test_div():
    for c in [case_1, case_2, case_3, case_4, case_5, case_6]:
        f, g = c()
        fg = GenPoly(f, sigma, s)
        gg = GenPoly(g, sigma, s)
        rg = fg / gg.LM()
        err = fg - gg.LM() * rg
        assert err.as_expr().simplify() == 0


def test_as_numer_denom():
    for c in [case_1, case_2, case_3, case_4, case_5, case_6]:
        for expr in c():
            xp = GenPoly(expr, sigma, s)
            num, den = xp.as_numer_denom()
            print(f"Numerator: {num}, Denominator: {den}")
            err = num.as_expr() / den.as_expr() - expr
            assert err.simplify() == 0


def test_side_effects():
    for c in [case_1, case_2, case_3, case_4, case_5, case_6]:
        for expr in c():
            a = GenPoly(expr, sigma, s)
            b = a.as_expr()
            b += 5
            err = a - b
            assert err.as_expr() == -5
