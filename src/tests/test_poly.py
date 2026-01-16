import sympy as sp

from chimera.poly import hermite_form


def test_hermite_form():
    s = sp.Symbol("s")
    m = sp.Matrix(
        [
            [2 * s**2 + 1 * s + 5, s + 2],
            [1, s**3],
        ]
    )
    w = hermite_form(m, s)
    sp.pprint(w)
