import sympy as sp

def hermite_form(mat: sp.Matrix, s_gen: sp.Symbol):
    """
    Compute the row hermite form of a polynomial matrix M

    This will compute an unimodular matrix U such that

    .math::
        B = M U

    where B is an upper triangular matrix.

    Parameters
    ----------
    mat: Polynomial matrix
    s_gen: Generator of the polynomial

    Returns
    -------
    Tuple of matrices B and U
    """
    pass
