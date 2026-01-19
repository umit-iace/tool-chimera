import sympy as sp

from chimera.poly import row_hermite_form, is_in_row_hermite_form


def test_row_hermite_form():
    s = sp.Symbol("s")
    a_mat = sp.Matrix(
        [
            # [2 * s**2 + 1 * s + 5, s + 2],
            # [1, s**3],
            [s**2, s + 2],
            [s**3, 2 * s + 2],
        ]
    )
    h_mat, u_mat = row_hermite_form(a_mat, s)
    sp.pprint(h_mat)
    sp.pprint(u_mat)

    # check properties
    assert is_in_row_hermite_form(h_mat, s)

    # check trafo
    err = h_mat - u_mat @ a_mat
    err.simplify()
    assert err == sp.Matrix.zeros(*err.shape)
    assert s not in u_mat.det().free_symbols


def test_is_in_row_hermite_form():
    x = sp.symbols("x")
    # A simple HNF example
    H = sp.Matrix(
        [
            [1, 0, 0],
            [0, x + 1, 0],
            [
                0,
                0,
                x**2 + x + 1,
            ],  # monic pivots, strictly increasing pivot columns 0,1,2
        ]
    )
    assert is_in_row_hermite_form(H, x) == True

    # Not HNF: pivot not monic
    B = sp.Matrix([[2, 0], [0, x]])
    assert is_in_row_hermite_form(B, x) == False

    # Not HNF: above-pivot entry has degree >= pivot degree
    C = sp.Matrix([[0, x**2], [0, x]])
    assert is_in_row_hermite_form(C, x) == False
