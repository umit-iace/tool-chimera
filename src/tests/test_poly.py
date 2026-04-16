import pytest
import sympy as sp

from chimera.poly import (
    row_hermite_form,
    is_in_row_hermite_form,
    echelon_form,
    is_in_echelon_form,
    column_reduced_form,
    is_col_reduced,
    random_poly_mat,
)


def test_echelon_form():
    a, b, c, s = sp.symbols("a b c s")
    a_mat = random_poly_mat(3, 3, 3, -5, 5, s)
    # add some symbols
    a_mat[0, 0] += a
    a_mat[1, 1] += b
    a_mat[2, 2] += c

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


def test_is_in_echelon_form():
    m = sp.Matrix.eye(4)
    assert is_in_echelon_form(m)

    m[1, 2] = 1
    assert is_in_echelon_form(m)

    m[3, 1] = 1
    assert not is_in_echelon_form(m)


@pytest.mark.skip("Not done yet")
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


def test_column_reduced_form():
    s = sp.Symbol("s")
    good = []
    bad = []
    mat = sp.Matrix(
        [
            [
                4 * s**2 + 2 * s - 2,
                s - 4,
                5 * s**3 - 2 * s**2 - 4 * s + 3,
                -3 * s**4 - 5 * s**3 - 2 * s**2 - 5 * s - 1,
                4,
                -5 * s**3 - 3 * s**2 + 4,
            ],
            [
                -4,
                5 * s**4 - 2 * s**3 + 5 * s**2 - 4 * s - 4,
                -3 * s - 3,
                4,
                -4 * s - 1,
                3 * s - 5,
            ],
            [
                -4 * s**4 + 5 * s**3 + 5 * s**2 - 2 * s - 3,
                -4 * s - 1,
                2,
                -(s**3) + 4 * s**2 - 4 * s - 2,
                5 * s**5 - 3 * s**3 + s**2 - s + 3,
                -5 * s**3 - 5 * s**2 + s + 4,
            ],
            [4, 3, 4, 5 * s**2 - 2 * s - 5, 5 * s + 3, 4],
            [
                -1,
                -3 * s**3 - s**2 - s - 5,
                -2 * s - 5,
                3 - 5 * s,
                -3 * s**4 + s**3 + 5 * s**2 + s + 5,
                -5 * s**3 + s**2 + 3 * s,
            ],
        ]
    )
    assert not is_col_reduced(mat, s)
    good.append(mat)

    mat = sp.Matrix([
        [3, -3 * s - 4, -1],
        [-2, -1, -4],
        [3, -3 * s - 4, -1]])
    assert not is_col_reduced(mat, s)
    bad.append(mat)

    for k in range(10):
        while True:
            mat = random_poly_mat(5, 6, 5, -5, 5, s)
            if not is_col_reduced(mat, s):
                break
        if mat.rank() == 5:
            good.append(mat)
        else:
            bad.append(mat)

    def _red_col(m):
        print(m)
        red, r = column_reduced_form(m, s)
        assert is_col_reduced(red, s)

    print("Good list:")
    for b in good:
        _red_col(b)

    print("Bad list:")
    for b in bad:
        with pytest.raises(ValueError):
            _red_col(b)
