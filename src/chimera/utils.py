"""
Utility functions
"""
from random import randint

import numpy as np
import sympy as sp
from sympy import random_poly
from sympy.core.random import seed
from sympy.core.function import AppliedUndef

from .genpoly import GenPoly

__all__ = ["check_singularities", "get_lccm", "get_matrix_terms", "is_col_reduced", "inverse_laplace_transform",
           "simplify_conv_integrals", "substitute_non_state_terms", "build_hcf_state", "substitute_hcf_state"]

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
        if tdn == sp.S.NegativeInfinity:
            tdn = 0
        if tdd == sp.S.NegativeInfinity:
            tdd = 0
        td = tdn - tdd
        return (lc, ld), (tc, td)

    def _handle_coefficients(_p, sym):
        ld = 0
        td = sp.S.Infinity
        for c in _p.coefficients():
            (_lc, _ld), (_tc, _td) = get_term(c, sym)
            if _ld >= ld:
                ld = _ld
                lc = _lc
            if _td <= td:
                td = _td
                tc = _tc
        return (lc, ld), (tc, td)

    # maybe it is a generalised polynomial?
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
    mat = sp.Matrix([random_poly(p_gen, randint(0, order), c_min, c_max) for i in range(m * n)]).reshape(m, n)
    return mat


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

    # # collect expressions concerning them
    # col_kernel = kernel.collect(c_funcs)
    # display(col_kernel)

    # treat each function separately
    for func in c_funcs:
        factor = kernel.coeff(func)
        res = _simplify_conv_factor(factor, var, lower, upper)
        for r in res:
            results.append((r[0] * func, r[1]))

    return results


def _simplify_conv_factor(factor, var, lower, upper):
    res = []

    # display(factor)

    # start with isolated dirichlet deltas
    dir_factors = 0
    hea_factors = {}
    rem_factors = 0
    for term in sp.Add.make_args(factor.expand()):
        if term.has(sp.DiracDelta) and not term.has(sp.Heaviside):
            dir_factors += term
        elif term.has(sp.Heaviside):
            # display(term)
            f = term.atoms(sp.Heaviside)
            assert len(f) == 1
            f = list(f)[0]
            # display(f)
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

    # just re-add the Diracs, sympy will simplify them
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

    new_args = tuple(substitute_non_state_terms(a, var, order, lower, upper) for a in expr.args)
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

    # extract the shift amplitude
    time_arg = func_term.args[0]
    shift_degree = time_arg - time_sym

    # highest derivatives are allowed
    if der_order == order:
        # with all but the maximum shifts
        if shift_degree < upper:
            return expr
        else:
            print("Warning, ignoring possibly incompatible term")
            return expr

    # lumped lower derivatives only allowed with minimum shift (lower shift boundary)
    if shift_degree == lower:
        return expr

    # substitute lumped term
    # print(f"Found incompatible term {expr}")
    return _substitute_lower_derivative_lumped_delay(var, shift_degree, der_order, order, (lower, upper))


def _substitute_lower_derivative_lumped_delay(var, shift_degree, der_order, max_der_order, bounds):
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

        int_kernel = (tau - sigma) ** (n - i - 1) / sp.factorial(n - i - 1) * var.subs(time_sym, theta).diff(theta,
                                                                                                             n).subs(
            theta,
            time_sym + sigma)

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
    return _substitute_lower_derivative_distributed_delay(var, der_order, order, conv_kernel, int_bounds)


def _substitute_lower_derivative_distributed_delay(var, der_order, max_der_order, kernel, bounds):
    theta, sigma = sp.symbols("theta sigma")
    time_sym = var.args[0]

    tau, tau_lo, tau_hi = bounds
    i = der_order
    n = max_der_order

    res = 0
    for k in range(i, n):
        int1 = (tau_hi - tau) ** (k - i) * kernel
        i1 = sp.Integral(int1, (tau, tau_lo, tau_hi))
        i1 *= (-1) * +(k - i) / sp.factorial(k - i) * var.subs(time_sym, theta).diff(theta, k).subs(theta,
                                                                                                    time_sym + tau_hi)

        int2 = (tau - sigma) ** (n - i - 1) * kernel.subs(tau, sigma)
        i2 = sp.Integral(int2, (sigma, tau_lo, tau))
        int3 = i2 * (-1) ** (n - 1) / sp.factorial(n - i - 1) * var.subs(time_sym, theta).diff(theta, n).subs(theta,
                                                                                                              time_sym + tau)
        i3 = sp.Integral(int3, (tau, tau_lo, tau_hi))
        res += i1 + i3

    # print(f"Replacing with {res}")
    return res


def inverse_laplace_transform(expr, diff_sym, shift_sym, ana_sym):
    # extract the normalised common polynomial denominator
    nrs, drs = expr.together(shift_sym).as_numer_denom()
    drs_qp = GenPoly(drs, shift_sym)
    assert len(drs_qp.exponents()) == 1
    d_lc = shift_sym ** drs_qp.degmax() * sp.Poly(drs_qp.LC(), diff_sym).LC()
    nn = (nrs / d_lc).simplify()
    dn = (drs / d_lc).simplify()

    # verify
    err = (nn / dn - expr).simplify()
    assert err == 0

    # collect the shifts
    nn_qp = GenPoly(nn, shift_sym)

    # perform partial fraction decomposition of the common poles
    d_pfd = (1 / dn).apart()
    # display(d_pfd)

    res = 0
    for term in sp.Add.make_args(d_pfd):
        for exp, coeff in nn_qp.as_dict().items():
            if 0:
                # Laplace transform of sympy does not know how the handle shifts
                s_expr = term * coeff * sp.exp(s * exp)
            else:
                # so we perform the shift manually
                s_expr = term * coeff
                t_expr = sp.inverse_laplace_transform(s_expr, diff_sym, ana_sym
                                                      ).subs(ana_sym, ana_sym + exp)
            # display(s_expr)
            # display(t_expr)
            res += t_expr

    return res


def build_hcf_state(y_vec, derivative_orders, shift_intervals, temp_var, spat_var):
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
                y = y_vec[i].diff(temp_var, j - 1).subs(temp_var, temp_var + shift_intervals[i][0] + spat_var)

            eta_y_map[eta] = y
            y_eta_map[y] = eta
            hcf_vec.append(eta)

    return sp.Matrix(hcf_vec), y_eta_map, eta_y_map


def substitute_hcf_state(expr, mapping, temp_var, spat_var):
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
            subs_arg = var_map[(func.name, der_order)]
            ref_func = func.func(temp_var).diff(temp_var, der_order).subs(temp_var, subs_arg)
            rep_func = mapping[ref_func]
            if len(rep_func.args) == 1:
                arg_map = {temp_var: orig_arg}
            elif len(rep_func.args) == 2:
                d = sp.Dummy()
                orig_arg = orig_arg.subs(spat_var, d)
                spat_expr = sp.solve(orig_arg.subs(temp_var, subs_arg) - temp_var, spat_var)[0]
                arg_map = {spat_var: spat_expr.subs(d, spat_var)}
            else:
                raise NotImplementedError

            return rep_func.subs(arg_map)

        new_args = tuple(rec(a) for a in _expr.args)
        if new_args != _expr.args:
            return _expr.func(*new_args)
        return _expr

    return rec(expr)
