"""
Utility functions
"""

import random as rd
from typing import Callable

import numpy as np
import sympy as sp
from sympy.core.random import seed, shuffle
from sympy.core.function import AppliedUndef
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.expressiondomain import EX
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.rings import PolyElement
from sympy.utilities.iterables import ordered_partitions
from sympy.matrices import Matrix

from .qpoly import QPoly, QuasiPolynomialElement

__all__ = [
    "check_singularities",
    "get_lccm",
    "get_matrix_terms",
    "is_col_reduced",
    "inverse_laplace_transform",
    "simplify_conv_integrals",
    "substitute_non_state_terms",
    "build_hcf_state",
    "substitute_hcf_state",
]

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


def get_leading_term_matrix(mat, sym, subs=None):
    (lc, ld), _ = get_matrix_terms(mat, sym, subs)
    return lc, ld


def get_leading_row_degrees(mat, sym):
    (lc, ld), (tc, td) = get_matrix_terms(mat, sym)
    degs = ld - td
    lr_degs = [max(degs[r, :]) for r in range(mat.shape[0])]
    return lr_degs


def get_trailing_term_matrix(mat, sym, subs=None):
    _, (tc, td) = get_matrix_terms(mat, sym, subs)
    return tc, td


def get_matrix_terms(mat, sym, subs: dict = None):
    if subs is not None:
        mat = mat.subs(subs).expand()

    # l_degs = sp.MutableDenseMatrix.zeros(*mat.shape)
    # l_cffs = sp.MutableDenseMatrix.zeros(*mat.shape)
    # t_degs = sp.MutableDenseMatrix.zeros(*mat.shape)
    # t_cffs = sp.MutableDenseMatrix.zeros(*mat.shape)

    l_degs = []
    l_cffs = []
    t_degs = []
    t_cffs = []
    for e in mat.flat():
        (lc, ld), (tc, td) = get_term(e, sym)
        l_cffs.append(lc)
        l_degs.append(ld)
        t_cffs.append(tc)
        t_degs.append(td)

    l_degs = domain_matrix_from_flat_list(l_degs, mat.shape)
    l_cffs = domain_matrix_from_flat_list(l_cffs, mat.shape)
    t_degs = domain_matrix_from_flat_list(t_degs, mat.shape)
    t_cffs = domain_matrix_from_flat_list(t_cffs, mat.shape)

    return (l_cffs, l_degs), (t_cffs, t_degs)


def domain_matrix_from_flat_list(elements, shape):
    e = elements[0]
    if isinstance(e, DomainElement):
        if isinstance(e, (PolyElement, QuasiPolynomialElement)):
            domain = e.ring
        elif isinstance(e, FracElement):
            base_dom = e.field.domain
            syms = e.field.symbols
            domain = base_dom.frac_field(*syms)
        elif EX.of_type(e):
            domain = EX
        else:
            domain = e.domain
        dm = DomainMatrix.from_list_flat(elements, shape, domain)
    else:
        # construct nested list
        rows, cols = shape
        elems_nested = [elements[i * cols: (i + 1) * cols] for i in range(rows)]
        dm = DomainMatrix.from_list_sympy(rows, cols, elems_nested, extension=True)

    m = dm.to_Matrix()
    if not m.free_symbols:
        return m

    return dm


def get_leading_column_degrees(
        mat: sp.Matrix, sym: sp.Symbol, subs: dict = None
) -> list:
    """
    Extract the highest degrees of each column
    """
    (coeffs, degs), _ = get_matrix_terms(mat, sym, subs)
    col_degs = _leading_col_degs(degs)
    return col_degs


def _leading_col_degs(degs: DomainMatrix | Matrix) -> list[DomainElement | sp.Expr]:
    return _col_degs(degs, max)


def _trailing_col_degs(degs: DomainMatrix | Matrix) -> list[DomainElement | sp.Expr]:
    return _col_degs(degs, min)


def _leading_row_degs(degs: DomainMatrix | Matrix) -> list[DomainElement | sp.Expr]:
    return _col_degs(degs.transpose(), max)


def _trailing_row_degs(degs: DomainMatrix | Matrix) -> list[DomainElement | sp.Expr]:
    return _col_degs(degs.transpose(), min)


def _col_degs(degs: DomainMatrix | Matrix, cmp: Callable) -> list[DomainElement | sp.Expr]:
    c_degs = []
    for c in range(degs.shape[1]):
        cd = []
        for r in range(degs.shape[0]):
            d = degs[r, c]
            cd.append(d.element if isinstance(d, DomainScalar) else d)
        c_degs.append(cmp(cd))
    return c_degs


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


def get_row_degrees(mat: sp.Matrix, sym: sp.Symbol) -> list:
    """
    Extract the highest and smallest degrees of each row
    """
    (_, l_degs), (_, t_degs) = get_matrix_terms(mat, sym)
    r_degs = [max(l_degs[r, :] - t_degs[r, :]) for r in range(mat.shape[0])]
    return r_degs


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

    def _handle_qpoly(_qp: QuasiPolynomialElement):
        lc = _qp.LC()
        ld = _qp.degmax()
        tc = _qp.TC()
        td = _qp.degmin()
        return (lc, ld), (tc, td)

    def _handle_fracelement(_fe):
        syms = _fe.field.symbols
        if sym not in syms:
            return (_fe, 0), (_fe, 0)
        return _handle_rational(_fe.numer, _fe.denom)

    def _handle_rational(_np, _dp):
        if isinstance(_np, PolyElement):
            (lcn, ldn), (tcn, tdn) = _handle_polyelement(_np)
            (lcd, ldd), (tcd, tdd) = _handle_polyelement(_dp)
        else:
            (lcn, ldn), (tcn, tdn) = _handle_poly(_np)
            (lcd, ldd), (tcd, tdd) = _handle_poly(_dp)

        lc = lcn / lcd
        ld = ldn - ldd
        if tdn == sp.S.NegativeInfinity:
            tdn = 0
        if tdd == sp.S.NegativeInfinity:
            tdd = 0
        tc = tcn / tcd
        if hasattr(tc, "cancel"):
            tc = tc.cancel()
        td = tdn - tdd

        return (lc, ld), (tc, td)

    def _handle_polyelement(_pe):
        lc = _pe.LC
        ld = _pe.degree()
        td = _pe.tail_degree()
        if td == sp.S.NegativeInfinity:
            tc = 0
        else:
            tc = _pe.coeff(_pe.ring(_pe.ring.gens[0] ** td))
        return (lc, ld), (tc, td)

    def _handle_poly(_p: sp.Poly):
        lc = _p.LC()
        ld = _p.degree()
        td = sp.Poly(_p.EM().as_expr(), _p.gen).degree()
        tc = _p.coeff_monomial(_p.gen ** td)
        return (lc, ld), (tc, td)

    def _handle_qpoly_coefficients(_p, sym):
        ld = sp.S.NegativeInfinity
        td = sp.S.Infinity
        lc = tc = _p.ring.zero
        for e, c in _p.items():
            (_lc, _ld), (_tc, _td) = _handle_fracelement(c)
            _lc = QuasiPolynomialElement.from_dict({e: _lc}, _p.ring)
            if _ld == ld:
                lc += _lc
            elif _ld > ld:
                ld = _ld
                lc = _lc

            _tc = QuasiPolynomialElement.from_dict({e: _tc}, _p.ring)
            if _td == td:
                tc += _tc
            elif _td < td:
                td = _td
                tc = _tc

        return (lc, ld), (tc, td)

    # maybe it is a quasipolynomial?
    if isinstance(e, QuasiPolynomialElement):
        if e.gen == sym:
            return _handle_qpoly(e)
        else:
            return _handle_qpoly_coefficients(e, sym)

    # maybe it is a FracElement?
    if isinstance(e, FracElement):
        return _handle_fracelement(e)

    # maybe it is a PolyElement?
    if isinstance(e, PolyElement):
        return _handle_polyelement(e)

    # maybe it is an Expression?
    if EX.of_type(e):
        raise NotImplementedError

    # maybe it is a polynomial expression?
    if e.is_Poly:
        if e.gen == sym:
            return _handle_poly(e)
        # else:
        #     return _handle_qpoly_coefficients(e, sym)
    if e.is_polynomial(sym):
        p = sp.Poly(e, sym)
        return _handle_poly(p)

    # maybe it is a rational expression?
    if e.is_rational_function(sym):
        num, den = e.as_numer_denom()
        np, dp = [sp.Poly(expr, sym) for expr in (num, den)]
        return _handle_rational(np, dp)

    raise ValueError
    # last resort, try to parse as quasipolynomial
    p = QuasiPolynomialElement.from_expr(e, sym)
    return _handle_qpoly(p)


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
    col_degs = [
        max([degs[i, c] for i in range(mat.shape[0])]) for c in range(mat.shape[1])
    ]

    if isinstance(coeffs, DomainMatrix):
        r = DomainMatrix.zeros(mat.shape, coeffs.domain)
    else:
        r = Matrix.zeros(*mat.shape)
    for c_idx in range(mat.shape[1]):
        for r_idx in range(mat.shape[0]):
            if degs[r_idx, c_idx] == col_degs[c_idx]:
                c = coeffs[r_idx, c_idx]
                if isinstance(c, DomainScalar):
                    c = c.element
                r[r_idx, c_idx] = c
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
        gp_sym: Generator of generalised polynomials
        gp_expr: Expression to use for *gp_sym*
        verbose: Be verbose

    Returns:
        False if expr contains singularities, True otherwise.
    """
    if not isinstance(expr, QPoly):
        expr = QPoly(expr, gp_sym)
    num, den = expr.as_numer_denom()
    p_den = sp.Poly(den.as_expr(), rat_sym)
    poles = sp.roots(p_den, rat_sym)
    if verbose:
        print(f"{len(poles.items())} poles to check.")
    for p, m in poles.items():
        if verbose:
            print(f"Testing pole at s={p} with multiplicity {m}:")
        res = num.as_expr().subs(gp_sym, gp_expr).evalf(subs={rat_sym: p})
        if not np.isclose(complex(res), 0):
            if verbose:
                print(f"Failure! Numerator yields residual {res}")
            return False
    else:
        return True


def zero_lower_left(mat):
    res = sp.MutableDenseMatrix(mat)
    for r in range(mat.shape[0]):
        for c in range(0, r):
            res[r, c] = 0
    return res


def zero_upper_right(mat):
    res = sp.MutableDenseMatrix(mat)
    for r in range(mat.shape[0]):
        for c in range(r + 1, mat.shape[1]):
            res[r, c] = 0
    return res


def factor_diag_pre(mat, diag_mat):
    assert diag_mat.shape[1] == mat.shape[0]
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


def simplify_conv_integrals(expr):
    if isinstance(expr, sp.Integral):
        int_kernel = expr.args[0]
        int_var, lb, ub = expr.args[1]
        if lb == -sp.oo and ub == sp.oo:
            res = 0
            for r in _simplify_conv(int_kernel, int_var, lb, ub):
                res += expr.func(r[0], r[1])
            return res
        else:
            return expr

    # traverse the tree
    new_args = tuple(simplify_conv_integrals(arg) for arg in expr.args)
    if new_args != expr.args:
        try:
            expr = expr.func(*new_args).doit()
        except Exception as e:
            # Fallback: keep original node if reconstruction fails
            print(e)
            print(expr)
            print(expr.args)
            print(new_args)
            pass

    return expr


def _simplify_conv(kernel, var, lower, upper):
    results = []

    # identify applied undefined functions that are being convolved
    c_funcs = kernel.atoms(AppliedUndef)

    # treat each function separately
    for func in c_funcs:
        factor = kernel.expand().coeff(func)
        res = _simplify_conv_factor(factor, var, lower, upper)
        for r in res:
            results.append((r[0] * func, r[1]))

    return results


def _simplify_conv_factor(factor, var, lower, upper):
    res = []

    dir_factors = 0
    hea_factors = {}
    rem_factors = 0
    for term in sp.Add.make_args(factor.expand()):
        if term.has(sp.DiracDelta) and not term.has(sp.Heaviside):
            dir_factors += term
        elif term.has(sp.Heaviside):
            f = term.atoms(sp.Heaviside)
            assert len(f) == 1
            f = list(f)[0]
            c = (term / f).simplify()
            if c in hea_factors:
                hea_factors[c] += [f]
            else:
                if -c in hea_factors:
                    hea_factors[-c] += [-f]
                else:
                    hea_factors[c] = [f]
        else:
            rem_factors += term

    # just re-add the diracs, sympy will simplify them
    res.append((dir_factors, (var, lower, upper)))

    # continue with heavyside expressions
    for c, funcs in hea_factors.items():
        # order funcs by arg and correct the signs such that
        # expr = factor * (H(lb) - H(ub))

        # extract and order the bounds
        bounds = []
        for f in funcs:
            if not isinstance(f, sp.Heaviside):
                # ignore the minus sign
                if isinstance(f, sp.Mul):
                    for a in f.args:
                        if isinstance(a, sp.Heaviside):
                            f = a
                            break
                else:
                    raise NotImplementedError
            bounds.append(f.args[0].subs(var, 0))
        s_funcs, s_bounds = zip(*sorted(zip(funcs, bounds), key=lambda x: x[1]))

        # check for minus sign at the wrong place
        if not isinstance(s_funcs[0], sp.Heaviside):
            s_funcs = [-1 * f for f in s_funcs]
            c *= -1

        # append new factor with reduced bounds
        res.append((c, (var, *bounds)))

    # add the remainder
    res.append((rem_factors, (var, lower, upper)))
    return res


def substitute_non_state_terms(expr, var, order, lower, upper):
    """
    Substitute terms that will not be part of the controller form state

    var: Variable to look at
    order: Derivative order of the distributed state member
    lower:  Lower bound of the shift interval
    upper:  Upper bound of the shift interval
    """
    if var.name in str(expr.atoms(sp.core.function.AppliedUndef)):
        if isinstance(expr, sp.Integral):
            # distributed case
            return _substitute_non_state_distributed_term(expr, var, order)
        if isinstance(expr, (sp.core.function.AppliedUndef, sp.Subs)):
            # lumped case
            return _substitute_non_state_lumped_term(expr, var, order, lower, upper)

    new_args = tuple(
        substitute_non_state_terms(a, var, order, lower, upper) for a in expr.args
    )
    if new_args != expr.args:
        return expr.func(*new_args)
    return expr


def _substitute_non_state_lumped_term(expr, var, order, lower, upper):
    time_sym = var.args[0]
    if isinstance(expr, sp.core.function.AppliedUndef):
        func_term = expr
        der_order = 0
    elif isinstance(expr, sp.Subs):
        der_term = expr.args[0]
        if not isinstance(der_term, sp.Derivative):
            return expr
        func_term = der_term.args[0].subs(zip(*expr.args[1:]))
        assert len(der_term.args) == 2
        der_order = der_term.args[1][1]
    else:
        raise NotImplementedError(expr)

    # highest derivatives are allowed
    if der_order == order:
        return expr

    # extract the shift amplitude
    time_arg = func_term.args[0]
    shift_degree = time_arg - time_sym

    # lumped lower derivatives only allowed with minimum shift (lower shift boundary)
    if shift_degree == lower:
        return expr

    # substitute lumped term
    # print(f"Found incompatible term {expr}")
    return _substitute_lower_derivative_lumped_delay(
        var, shift_degree, der_order, order, (lower, upper)
    )


def _substitute_lower_derivative_lumped_delay(
        var, shift_degree, der_order, max_der_order, bounds
):
    time_sym = var.args[0]
    theta, sigma = sp.symbols("theta sigma")
    tau = shift_degree
    tau_lo, tau_hi = bounds
    i = der_order
    n = max_der_order

    res = 0
    for k in range(i, n):
        der = var.subs(time_sym, theta).diff(theta, k).subs(theta, time_sym - tau_lo)
        s1 = (tau + tau_lo) ** (k - i) / sp.factorial(k - i) * der

        int_kernel = (
                (tau - sigma) ** (n - i - 1)
                / sp.factorial(n - i - 1)
                * var.subs(time_sym, theta).diff(theta, n).subs(theta, time_sym + sigma)
        )

        s2 = sp.Integral(int_kernel, (sigma, -tau_lo, tau_hi))
        res += s1 + s2

    # print(f"Replacing with {res}")
    return res


def _substitute_non_state_distributed_term(expr, var, order):
    time_sym = var.args[0]
    int_arg, int_bounds = expr.args

    conv_kernel, func_part = int_arg.as_independent(time_sym)
    if isinstance(func_part, sp.core.function.AppliedUndef):
        # func_term = func_part
        der_order = 0
    elif isinstance(func_part, sp.Subs):
        der_term = func_part.args[0]
        if not isinstance(der_term, sp.Derivative):
            return expr
        # func_term = der_term.args[0].subs(zip(*func_part.args[1:]))
        assert len(der_term.args) == 2
        der_order = der_term.args[1][1]
    else:
        raise NotImplementedError(expr)

    # check the derivative order
    if der_order == order:
        # nothing to do
        return expr

    # substitute the integral
    # print(f"Found incompatible term {expr}")
    return _substitute_lower_derivative_distributed_delay(
        var, der_order, order, conv_kernel, int_bounds
    )


def _substitute_lower_derivative_distributed_delay(
        var, der_order, max_der_order, kernel, bounds
):
    theta, sigma = sp.symbols("theta sigma")
    time_sym = var.args[0]

    tau, tau_lo, tau_hi = bounds
    i = der_order
    n = max_der_order

    res = 0
    for k in range(i, n):
        int1 = (tau_hi - tau) ** (k - i) * kernel
        i1 = sp.Integral(int1, (tau, tau_lo, tau_hi))
        i1 *= (
                (-1)
                * +(k - i)
                / sp.factorial(k - i)
                * var.subs(time_sym, theta).diff(theta, k).subs(theta, time_sym + tau_hi)
        )

        int2 = (tau - sigma) ** (n - i - 1) * kernel.subs(tau, sigma)
        i2 = sp.Integral(int2, (sigma, tau_lo, tau))
        int3 = (
                i2
                * (-1) ** (n - 1)
                / sp.factorial(n - i - 1)
                * var.subs(time_sym, theta).diff(theta, n).subs(theta, time_sym + tau)
        )
        i3 = sp.Integral(int3, (tau, tau_lo, tau_hi))
        res += i1 + i3

    # print(f"Replacing with {res}")
    return res


def inverse_laplace_transform(expr: QuasiPolynomialElement, diff_sym, shift_sym, ana_sym):
    """
    Helper function to help sympy compute the inverse laplace transform
    """
    if not isinstance(expr, QuasiPolynomialElement):
        raise DeprecationWarning("This is no longer supported")
        # extract the normalised common polynomial denominator
        nrs, drs = expr.together(shift_sym).as_numer_denom()
        drs_qp = QPoly(drs, shift_sym)
        assert len(drs_qp.exponents()) == 1
        d_lc = shift_sym ** drs_qp.degmax() * sp.Poly(drs_qp.LC(), diff_sym).LC()
        nn = (nrs / d_lc).simplify()
        dn = (drs / d_lc).simplify()

        # verify
        err = (nn / dn - expr).simplify()
        assert err == 0

        # collect the shifts
        nn_qp = QPoly(nn, shift_sym)
    else:
        # extract the normalised common polynomial denominator
        nn_qp, dn_p = expr.as_numer_denom()
        dn = dn_p.as_expr()

    # perform partial fraction decomposition of the common poles
    d_pfd = (1 / dn).apart()

    res = 0
    for term in sp.Add.make_args(d_pfd):
        for exp, coeff in nn_qp.to_dict().items():
            """
            Sadly, sympy's inverse laplace transform  does not know how the handle 
            shifts, so instead of doing:
                s_expr = term * coeff * sp.exp(s * exp)
            we add the manually.
            """
            s_expr = term * coeff.as_expr()
            t_expr = sp.inverse_laplace_transform(s_expr, diff_sym, ana_sym).subs(
                ana_sym, ana_sym + exp
            )
            res += t_expr

    return res


def build_hcf_state(y_vec, derivative_orders, shift_intervals, temp_var, spat_var):
    """
    Compute the hyperbolic MIMO controller form state
    including the corresponding substitution maps to use it
    """
    y_eta_map = {}
    eta_y_map = {}
    hcf_vec = []
    for i in range(len(y_vec)):
        for j in range(1, derivative_orders[i] + 2):
            if j <= derivative_orders[i]:
                # lumped part
                eta = sp.Function(f"\\eta^O_{{{ {i + 1}, {j} }}}")(temp_var)
                y = y_vec[i].diff(temp_var, j - 1)
            else:
                # distributed part
                eta = sp.Function(f"\\eta^P_{{{i + 1}}}")(spat_var, temp_var)
                y = (
                    y_vec[i]
                    .diff(temp_var, j - 1)
                    .subs(temp_var, temp_var + shift_intervals[i][0] + spat_var)
                )

            eta_y_map[eta] = y
            y_eta_map[y] = eta
            hcf_vec.append(eta)

    return sp.Matrix(hcf_vec), y_eta_map, eta_y_map


def substitute_hcf_state(expr, mapping, temp_var, spat_var):
    """
    Substitute the hyperbolic MIMO controller form state into a FDE in the
    system input and flat output

    Basically this just helps sympy to recognise derivatives of functions with different
    argument names.
    """
    var_map = {}
    for var in mapping.keys():
        if isinstance(var, sp.Subs):
            var_map[(var.args[0].args[0].name, var.args[0].args[1][1])] = var.args[2][0]
        if isinstance(var, sp.Derivative):
            var_map[(var.args[0].name, var.args[1][1])] = var.args[0].args[0]
        if isinstance(var, AppliedUndef):
            var_map[(var.name, 0)] = var.args[0]

    def rec(_expr):
        func = None
        if isinstance(_expr, AppliedUndef):
            func = _expr
            der_order = 0
        if isinstance(_expr, sp.Subs):
            der = _expr.args[0]
            if isinstance(der, sp.Derivative):
                func = der.args[0].subs(zip(*_expr.args[1:]))
                der_order = der.args[1][1]

        if func is not None:
            orig_arg = func.args[0]
            subs_arg = var_map.get((func.name, der_order), None)
            if subs_arg is None:
                return _expr

            ref_func = (
                func.func(temp_var).diff(temp_var, der_order).subs(temp_var, subs_arg)
            )
            rep_func = mapping[ref_func]
            if len(rep_func.args) == 1:
                arg_map = {temp_var: orig_arg}
            elif len(rep_func.args) == 2:
                d = sp.Dummy()
                orig_arg = orig_arg.subs(spat_var, d)
                # resolve the argument shifts
                tb = sp.solve(subs_arg.subs(spat_var, 0), temp_var)[0]
                spat_expr = orig_arg.subs(temp_var, tb).subs(d, spat_var)
                arg_map = {spat_var: spat_expr}
            else:
                raise NotImplementedError

            return rep_func.subs(arg_map)

        new_args = tuple(rec(a) for a in _expr.args)
        if new_args != _expr.args:
            return _expr.func(*new_args)
        return _expr

    return rec(expr)


def random_partition_m_parts(n, m):
    """Generate a random partition of n into exactly m parts."""
    if m < 1:
        raise ValueError("m must be greater than 1")
    # pad the input as ordered partitions generates no zero entries
    n += m

    # form all possible partitions, choose one and shuffle it
    all_partitions = [list(p) for p in ordered_partitions(n, m)]
    part = rd.choice(all_partitions)
    shuffle(part)

    # remove padding
    part[:] = [p - 1 for p in part]

    return part


def show_matrix_degrees(m: DomainMatrix):
    sp.pprint(
        sp.Matrix(
            m.applyfunc(lambda x: (x.degmin().evalf(), x.degmax().evalf())).to_list()
        ),
        wrap_line=False,
    )


def show_matrix_degree(m: DomainMatrix):
    sp.pprint(
        sp.Matrix(m.applyfunc(lambda x: x.degree().as_expr().evalf()).to_list()),
        wrap_line=False,
    )
