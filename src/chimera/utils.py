"""
Utility functions
"""
from random import randint

import numpy as np
import sympy as sp
from sympy import random_poly
from sympy.core.random import seed

from .genpoly import GenPoly

__all__ = ["check_singularities", "get_lccm", "is_col_reduced"]

# seed using the 'cat ran over the keyboard' method
rng_seed = 56802470204575498132489694523498
rng = np.random.default_rng(rng_seed)
seed(rng_seed)


def generate_integer_transform(dim: int, max_entry: int = 4) -> np.ndarray:
    """
    Generate an integer-valued transformation into lumped MIMO controller form
    """
    found = False
    i = 0
    while not found:
        T = rng.integers(low=-max_entry, high=max_entry, size=(dim, dim), endpoint=True)
        det = np.linalg.det(T)
        if np.abs(det) == 1:
            Tinv = np.linalg.inv(T)
            if np.max(np.abs(Tinv * det)) < max_entry + 1:
                found = True
        i += 1

    print(f"Integer Transformation found after {i} iterations")
    return T, Tinv


def get_coeffs(expr: sp.Expr, gen: sp.Symbol) -> dict:
    """
    Helper function to return the coefficients of different powers
    of *gen* in *expr*.
    """
    s = GenPoly(expr, gen)
    c = s.as_dict()
    return c


def get_matrix_coeffs(
    mat: sp.Matrix, gen: sp.Symbol, exponents_only=False
) -> list[list[dict]]:
    coeffs = []
    for j in range(mat.rows):
        r = []
        for i in range(mat.cols):
            c = get_coeffs(mat[j, i], gen)
            if exponents_only:
                r.append(c.keys())
            else:
                r.append(c)
        coeffs.append(r)
    return coeffs


def show_coeffs(expr: sp.Expr, gen: sp.Symbol):
    c = get_coeffs(expr, gen)
    for k, v in c.items():
        print(k, v)


def show_mat_coeffs(mat: sp.Matrix, gen: sp.Symbol):
    mc = get_matrix_coeffs(mat, gen)
    for i, row in enumerate(mc):
        for j, c in enumerate(row):
            print(f"Element: {j, i}")
            for k, v in c.items():
                print(k, v)


def get_leading_term_matrix(mat, sym, subs=None):
    (lc, ld), _ = get_matrix_terms(mat, sym, subs)
    return lc, ld


def get_trailing_term_matrix(mat, sym, subs=None):
    _, (tc, td) = get_matrix_terms(mat, sym, subs)
    return tc, td


def get_matrix_terms(mat, sym, subs: dict = None):
    if subs is not None:
        mat = mat.subs(subs).expand()
    l_degs = sp.MutableDenseMatrix.zeros(*mat.shape)
    l_cffs = sp.MutableDenseMatrix.zeros(*mat.shape)
    t_degs = sp.MutableDenseMatrix.zeros(*mat.shape)
    t_cffs = sp.MutableDenseMatrix.zeros(*mat.shape)
    for idx, e in mat.iter_items():
        (lc, ld), (tc, td) = get_term(e, sym)
        l_cffs[idx] = lc
        l_degs[idx] = ld
        t_cffs[idx] = tc
        t_degs[idx] = td
    return (l_cffs, l_degs), (t_cffs, t_degs)


def get_leading_column_degrees(
    mat: sp.Matrix, sym: sp.Symbol, subs: dict = None
) -> list:
    """
    Extract the highest degrees of each column
    """
    (coeffs, degs), _ = get_matrix_terms(mat, sym, subs)
    col_degs = [max(degs[:, c]) for c in range(mat.shape[1])]
    return col_degs


def get_trailing_column_degrees(
    mat: sp.Matrix, sym: sp.Symbol, subs: dict = None
) -> list:
    """
    Extract the smallest degrees of each column
    """
    _, (coeffs, degs) = get_matrix_terms(mat, sym, subs)
    col_degs = [min(degs[:, c]) for c in range(mat.shape[1])]
    return col_degs


def get_column_degrees(mat: sp.Matrix, sym: sp.Symbol, subs: dict = None) -> list:
    """
    Extract the highest and smallest degrees of each column
    """
    (l_coeffs, l_degs), (t_coeffs, t_degs) = get_matrix_terms(mat, sym, subs)
    min_degs = [min(t_degs[:, c]) for c in range(mat.shape[1])]
    max_degs = [max(l_degs[:, c]) for c in range(mat.shape[1])]
    return max_degs, min_degs


def get_term(
    e: sp.Expr, sym: sp.Symbol
) -> tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]:
    """
    Extract the leading coefficient of some expression

    At first classical polynomials with integer exponents are tried,
    if those don't work rational expressions are tested.
    Finally, the expression is treated as generalized polynomial with
    real exponents.

    Args:
        e (sp.Expr): Expression to parse
        sym (sp.Symbol): Generator whose leading term is sought

    Returns:
        Tuple of coefficient and exponent for the selected mode.
    """

    def _handle_gpoly(_gp: GenPoly):
        lc = _gp.LC()
        ld = _gp.degmax()
        tc = _gp.TC()
        td = _gp.degmin()
        return (lc, ld), (tc, td)

    def _handle_poly(_p: sp.Poly):
        lc = _p.LC()
        ld = _p.degree()
        tc = _p.TC()
        td = sp.Poly(_p.EM(), _p.gens).degree()
        return (lc, ld), (tc, td)

    def _handle_rational(_np, _dp):
        lc = _np.LC() / _dp.LC()
        ld = _np.degree() - _dp.degree()
        tc = _np.EC() / _dp.EC()
        tdn = sp.Poly(_np.EM(), _np.gens).degree()
        tdd = sp.Poly(_dp.EM(), _dp.gens).degree()
        if tdn == tdd == sp.S.NegativeInfinity:
            td = 0
        return (lc, ld), (tc, td)

    def _handle_coefficients(_p, sym):
        ld = td = 0
        for c in _p.coefficients():
            (_lc, _ld), (_tc, _td) = get_term(c, sym)
            if _ld >= ld:
                ld = _ld
                lc = _lc
            if _td <= td:
                td = _td
                tc = _tc
        return (lc, ld), (tc, td)

    # maybe it is a generalized polynomial?
    if isinstance(e, GenPoly):
        if e.gen == sym:
            return _handle_gpoly(e)
        else:
            return _handle_coefficients(e, sym)

    # maybe it is a polynomial expression?
    if e.is_Poly:
        if e.gen == sym:
            return _handle_poly(e)
        else:
            return _handle_coefficients(e, sym)
    if e.is_polynomial(sym):
        p = sp.Poly(e, sym)
        return _handle_poly(p)

    # maybe it is a rational expression?
    num, den = e.as_numer_denom()
    if num.is_polynomial(sym) and den.is_polynomial(sym):
        np, dp = [sp.Poly(expr, sym) for expr in (num, den)]
        return _handle_rational(np, dp)

    # last resort, try to parse as generalized polynomial
    p = GenPoly(e, sym)
    return _handle_gpoly(p)


def get_lccm(mat: sp.Matrix, sym: sp.Symbol, subs: dict = None) -> sp.Matrix:
    """
    Compute the leading column coefficient matrix

    Args:
        mat (sp.Matrix): Matrix to analyse
        sym (sp.Symbol): Generator to extract the coefficients of
        subs (dict): Parameter dictionary to use.

    Returns:
        The Leading column coefficient matrix.

    """
    (coeffs, degs), _ = get_matrix_terms(mat, sym, subs)
    col_degs = [max(degs[:, c]) for c in range(mat.shape[1])]

    r = sp.Matrix.zeros(*mat.shape)
    for c_idx in range(mat.shape[1]):
        for r_idx in range(mat.shape[0]):
            if degs[r_idx, c_idx] == col_degs[c_idx]:
                r[r_idx, c_idx] = coeffs[r_idx, c_idx]
    return r


def get_lctm(mat: sp.Matrix, sym: sp.Symbol, subs: dict = None) -> sp.Matrix:
    """
    Compute the leading column term matrix

    Args:
        mat (sp.Matrix): Matrix to analyse
        sym (sp.Symbol): Generator to extract the coefficients of
        subs (dict): Parameter dictionary to use.

    Returns:
        The Leading column term matrix.

    """
    (coeffs, degs), _ = get_matrix_terms(mat, sym, subs, "leading")
    col_degs = [max(degs[:, c]) for c in range(mat.shape[1])]

    r = sp.Matrix.zeros(*mat.shape)
    for c_idx in range(mat.shape[1]):
        for r_idx in range(mat.shape[0]):
            if degs[r_idx, c_idx] == col_degs[c_idx]:
                r[r_idx, c_idx] = coeffs[r_idx, c_idx] * sym ** degs[r_idx, c_idx]
    return r


def is_col_reduced(mat: sp.Matrix, sym: sp.Symbol, subs: dict = None):
    """
    Check if mat is column reduced, thus, if lccm(mat,sym) has full rank
    """
    lcm = get_lccm(mat, sym, subs)
    return lcm.rank() == min(mat.shape)


def check_derivative_condition(mat, diff_sym):
    return is_col_reduced(mat, diff_sym)


def check_prediction_condition(mat, pred_sym, diff_sym):
    cds = get_leading_column_degrees(mat, pred_sym)
    if sorted(cds) != cds:
        return False

    # Test structural identity
    lcms = get_lccm(mat, diff_sym)
    lcmse = get_lccm(lcms, pred_sym)
    lcme = get_lccm(mat, pred_sym)
    lcmes = get_lccm(lcme, diff_sym)
    if lcmse != lcmes:
        return False

    return True


def check_delay_condition(mat, delay_sym, diff_sym):
    if not is_col_reduced(mat, delay_sym):
        return False

    # Test structural identity
    lccm_s = get_lccm(mat, diff_sym)
    lccm_sd = get_lccm(lccm_s, delay_sym)
    lccm_d = get_lccm(mat, delay_sym)
    lccm_ds = get_lccm(lccm_d, diff_sym)
    if lccm_sd != lccm_ds:
        return False

    return True


def check_all_conds(mat, pred_sym, delay_sym, diff_sym):
    return (
        check_derivative_condition(mat, diff_sym)
        and check_prediction_condition(mat, pred_sym, diff_sym)
        and check_delay_condition(mat, delay_sym, diff_sym)
    )


def check_equal(expr1, expr2, n=10, positive=False, strictly_positive=False):
    """
    Compare to symbolic expressions by substituting random values in them

    Motivated by a Stackoverflow answer.
    """
    # determine over what range to generate random numbers
    sample_min = -1
    sample_max = 1
    if positive:
        sample_min = 0
        sample_max = 1
    if strictly_positive:
        sample_min = 1
        sample_max = 2

    # regroup all free symbols from both expressions
    free_symbols = set(expr1.free_symbols) | set(expr2.free_symbols)

    # brute force equality testing n-times
    for i in range(n):
        test_values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        s_dict = {s: v for s, v in zip(free_symbols, test_values)}
        res = (expr1 - expr2).evalf(subs=s_dict)
        if not np.isclose(res, 0):
            return False

    return True


def check_singularities(
    expr: sp.Expr,
    rat_sym: sp.Symbol,
    gp_sym: sp.Symbol,
    gp_expr: sp.Expr,
    verbose: bool = False,
):
    """
    Check if denominator zeroes are numerator zeroes as well

    Args:
        expr: Expression to check
        rat_sym: Generator of rational expressions
        gp_sym: Generator of generalized polynomials
        gp_expr: Expression to use for *gp_sym*
        verbose: Be verbose

    Returns:
        False if expr contains singularities, True otherwise.
    """
    if not isinstance(expr, GenPoly):
        expr = GenPoly(expr, gp_sym)
    num, den = expr.as_numer_denom()
    poles = sp.roots(den, rat_sym)
    for p, m in poles.items():
        if verbose:
            print(f"Testing pole at s={p}:")
        for expr in num, den:
            res = expr.as_expr().subs(gp_sym, gp_expr).evalf(subs={rat_sym: p})
            if not np.isclose(complex(res), 0):
                return False
    else:
        return True


def random_poly_mat(m, n, order, c_min, c_max, p_gen):
    mat = sp.Matrix([random_poly(p_gen, randint(0,order), c_min, c_max) for i in range(m*n)]).reshape(m,n)
    return mat


def zero_lower_left(mat):
    res = sp.MutableDenseMatrix(mat)
    for r in range(mat.shape[0]):
        for c in range(0, r):
            res[r,c] = 0
    return res


def zero_upper_right(mat):
    res = sp.MutableDenseMatrix(mat)
    for r in range(mat.shape[0]):
        for c in range(r+1, mat.shape[1]):
            res[r,c] = 0
    return res


def factor_diag_pre(mat, diag_mat):
    assert  diag_mat.shape[1] == mat.shape[0]
    res = sp.Matrix.zeros(*mat.shape)
    for r in range(mat.shape[0]):
        res[r, :] = mat[r, :] / diag_mat[r, r]
    return res


def factor_diag_post(mat, diag_mat):
    assert mat.shape[1] == diag_mat.shape[0]
    res = sp.Matrix.zeros(*mat.shape)
    for c in range(mat.shape[1]):
        res[:, c] = mat[:, c] / diag_mat[c, c]
    return res
