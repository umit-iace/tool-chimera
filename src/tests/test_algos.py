import sympy as sp

from pyform import Signomial, sdiv, gcdexs, bezouts

# generators
r = sp.Symbol("rho", positive=True)
s = sp.Symbol("s")

# some symbols
r = sp.Symbol("rho", positive=True)
a, b = sp.symbols("a b")
subs_dict = {a: sp.Rational(2,3), b: sp.Rational(1,8)}


def case_1():
    # positive integer exponents
    f = r**3 + 5 + r**2 * 10
    g = r**2 + 10*r + 6
    print("Case 1:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_2():
    # positive non-integer exponents
    f = r**sp.Rational(16, 5) + 5 + r**sp.Rational(14, 5) * 10
    g = r**(2 + sp.Rational(1, 3)) + 10*r + 6
    print("Case 2:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def case_3():
    # negative non-integer exponents and other symbols like s
    f = (s**2 + s)/(2*r) + r**(-2)
    g = s/2
    print("Case 3:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g

def case_4():
    # compound exponent

    f = ((s**2 + 2*s +1) * r**(a+b) + s**3 * r**-b) * r**a
    g = (s**2 + 2*s +1) * r**-a + s**3 * r**b
    print("Case 4:")
    print(f"f: {f}")
    print(f"g: {g}")
    return f, g


def test_dict_from_expr():
    for c in [
        case_1,
        case_2,
        case_3,
        case_4
    ]:
        f, g = c()
        for expr in f,g:
            s_dict = Signomial.dict_from_expr(expr, r)
            print(s_dict)
            test_expr = Signomial.dict_to_expr(s_dict, r)
            print(test_expr)
            err = expr - test_expr
            err = err.simplify()
            assert err == 0


def test_sdiv():
    for c in [
        case_1,
        case_2,
        case_3,
        case_4
    ]:
        f, g = c()

        # degree comp needs actual values, no symbols
        f = f.subs(subs_dict)
        g = g.subs(subs_dict)

        q, rem = sdiv(f, g, r)
        print(f"q: {q}")
        print(f"r: {rem}")

        rem = f - q.to_expr() * g - rem.to_expr()
        rem = rem.expand()
        print(f"error: {rem}")
        assert rem == 0


def test_gcdexs():
    for c in [
        case_1,
        # case_2,  ## this one will not occur in our systems
        case_3,
        # case_4,
              ]:
        f, g = c()

        # degree comp needs actual values, no symbols
        f = f.subs(subs_dict)
        g = g.subs(subs_dict)

        s, t, h = gcdexs(f, g, r, max_iterations=10)
        print(f"s: {s}")
        print(f"t: {t}")
        print(f"h(gcd): {h}")

        err = s*f + t*g - h
        err = err.simplify()
        print(f"Error: {err}")
        assert err == 0


def test_bezouts():
    for c in [
        case_1,
        # case_2,
        case_3,
        # case_4,
    ]:
        a, b = c()

        # degree comp needs actual values, no symbols
        a = a.subs(subs_dict)
        b = b.subs(subs_dict)

        p, q = bezouts(a, b, p_gen=s, s_gen=r)
        print(f"p: {p}")
        print(f"q: {q}")
