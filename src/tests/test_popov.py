from sympy import QQ, RR, symbols, pprint
from sympy.polys.matrices import DomainMatrix

from chimera.popov import row_popov_form, is_row_popov_form, col_popov_form, is_col_popov_form

z = symbols("z")
R = QQ.poly_ring(z)


def test_row_popov_form_canonical():

    A = DomainMatrix.from_list([
        [z**2 + z + 1,  z**2 + 1,      z],
        [z**2 + 1,      z**2 + z + 2,  z + 1],
        [z + 1,         1,             z**2 + 2],
    ], R)

    P, U, pivots = row_popov_form(
        A,
        canonical=True,
        return_pivots=True,
    )

    print("Canonical row Popov form:")
    pprint(P.to_Matrix())

    print("\nTransformation matrix U:")
    pprint(U.to_Matrix())

    print("\nZero-based pivot columns:")
    pprint(pivots)

    assert P == U.matmul(A)
    assert(is_row_popov_form(P, canonical=True))


def test_col_popov_form_canonical():

    A = DomainMatrix.from_list([
        [z**2 + z + 1,  z**2 + 1,      z],
        [z**2 + 1,      z**2 + z + 2,  z + 1],
        [z + 1,         1,             z**2 + 2],
    ], R)

    P, U, pivots = col_popov_form(
        A,
        canonical=True,
        return_pivots=True,
    )

    print("Canonical row Popov form:")
    pprint(P.to_Matrix())

    print("\nTransformation matrix U:")
    pprint(U.to_Matrix())

    print("\nZero-based pivot columns:")
    pprint(pivots)

    assert P == A.matmul(U)
    assert(is_col_popov_form(P, canonical=True))
