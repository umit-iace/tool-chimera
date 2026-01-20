from random import randint
import sympy as sp

from chimera.poly import row_hermite_form, is_in_row_hermite_form, echelon_form
from poly import is_in_echelon_form


def test_echelon_form():
    s = sp.Symbol("s")
    m = 3
    n = 3
    a_mat = sp.Matrix.zeros(m, n)
    for r in range(m):
        for c in range(n):
            deg = randint(0, 5)
            cffs = [randint(-5, 5) for d in range(deg + 1)]
            a_mat[r, c] = sp.Poly.from_list(cffs, s).as_expr()

    print("Initial matrix:")
    sp.pprint(a_mat)

    print("Left trafo:")
    e_mat, u_mat = echelon_form(a_mat, s, mode="pre")
    sp.pprint(e_mat)
    sp.pprint(u_mat)

    # check properties
    assert is_in_echelon_form(e_mat)
    # assert is_in_row_hermite_form(e_mat, s)

    # check left trafo
    err = e_mat - u_mat @ a_mat
    err.simplify()
    assert err == sp.Matrix.zeros(*err.shape)
    assert s not in u_mat.det().free_symbols

    print("Right trafo:")
    e_mat, u_mat = echelon_form(a_mat, s, mode="post")
    sp.pprint(e_mat)
    sp.pprint(u_mat)

    # check properties
    assert is_in_echelon_form(e_mat)

    # check right trafo
    err = e_mat - a_mat @ u_mat
    err.simplify()
    assert err == sp.Matrix.zeros(*err.shape)
    assert s not in u_mat.det().free_symbols


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
    sp.pprint(a_mat)
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
