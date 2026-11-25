from sympy import QQ, symbols, pprint
from sympy.polys.matrices import DomainMatrix

from chimera.hermite import column_hermite_form, is_column_hermite_form

z = symbols("z")
R = QQ.poly_ring(z)


def test_col_hermite_form():
    A = DomainMatrix.from_list([
        [z**2 + 1,     z + 1,       1],
        [z**3 + z,     z**2 + 2,    z],
    ], R)

    H, V = column_hermite_form(A)

    print("H:")
    pprint(H.to_Matrix())

    print("\nV:")
    pprint(V.to_Matrix())

    print("\nCheck H = A*V:")
    assert H == A.matmul(V)

    print("\nIs H in column Hermite form?")
    assert is_column_hermite_form(H)