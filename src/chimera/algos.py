"""
Custom algorithms for the project
"""

from typing import Union, Tuple, Any

import sympy as sp

from .poly import echelon_form, column_reduced_form
from .genpoly import GenPoly
from .utils import (
    get_matrix_terms,
    get_leading_term_matrix,
    factor_diag_post,
    is_col_reduced,
)

__all__ = ["gpld", "make_entire", "gplde", "eliminate_delays"]


def gcdexs(
    f: sp.Expr, g: sp.Expr, gp_gen, tol=1, max_iterations=100
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Extension of sympys gcdex that will work with generalized polynomials

    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``
    """
    h = [f, g]
    s = [1, 0]
    t = [0, 1]
    m = [0, 1]
    n = [1, 0]

    idx = 1
    while idx < max_iterations + 1:
        q, r = gplde(h[idx - 1], h[idx], gp_gen)
        q = q.as_expr()

        if r == 0:
            # this is really unlikely to happen, since exponents must be commensurable for that
            # print("Remainder is zero, algorithm terminated.")
            break

        sn = s[idx - 1] - q * s[idx]
        tn = t[idx - 1] - q * t[idx]

        # estimates
        mn = q * m[idx] + m[idx - 1]
        nn = q * n[idx] + n[idx - 1]
        err = sp.N(1 / nn**2)
        # err = sp.N(1 / nn**2)#, subs=s_dict)

        if 0:
            print(f"Step {idx}:")
            print("q, r, s, t, m, n, err:")
            print(q)
            print(r)
            print(sn)
            print(tn)
            print(mn)
            print(nn)
            print(err)
            print("#" * 40)

        # check breaking criterion
        try:
            br = bool(err <= tol)
        except TypeError:
            br = False
        if br:
            # approximation is sufficient
            break

        # update the stack
        h.append(r)
        s.append(sn)
        t.append(tn)
        m.append(mn)
        n.append(nn)
        idx += 1

    else:
        raise ArithmeticError(
            f"Maximum number of iterations reached but "
            f"approximation error is still {err}"
        )

    return s[-1], t[-1], h[-1]


def bezouts(
    p: sp.Expr,
    q: sp.Expr,
    p_gen: sp.Symbol,
    s_gen: sp.Symbol,
    s_gen_def: sp.Expr = None,
):
    """
    Solve multinomial bezout identity :math:`a p + b q = 1`,
    containing polynomials in *p_gen* and signomials in *s_gen*.

    Returns Coefficients *a* and *b* to satisfy the equation.
    """
    if s_gen_def is None:
        s_gen_def = sp.exp(p_gen)

    if s_gen not in p.free_symbols | q.free_symbols:
        # no signomials, shortcut possible
        a, b, h = sp.gcdex(p, q, gens=p_gen)
        assert h == 1
        return a, b

    """
    Step 1: Naively solve bezout identity $ap + bq = 1$ in domain `F[s_gen][p_gen]` 
    (Signomials in *s_gen* with coeffs that are Polys in *p_gen*).
    This will return a,b as rational functions in *p_gen*.
    """
    a, b, h = gcdexs(p, q, s_gen)
    # print("Solutions for a, b and h:")
    # print(a, b, h)

    # verify the solution
    sol = a * p + b * q - h
    sol = sol.expand().simplify()
    assert sol == 0

    """
    Step 2: Multiply equation with the lcm of the lcms of a and b,
    eliminating all denominators in the equation thus making a and b
    polynomials.
    """
    lcm_a = GenPoly(a, s_gen).get_coeff_denom_lcm()
    lcm_b = GenPoly(b, s_gen).get_coeff_denom_lcm()
    lcm = sp.lcm(lcm_a, lcm_b)
    # print("LCM of bezout coefficients:")
    # print(lcm)

    at = a * lcm
    at = at.simplify()
    bt = b * lcm
    bt = bt.simplify()
    ht = h * lcm
    ht = ht.simplify()
    # print("New coefficients a, b, h:")
    # print(at, bt, ht)

    # verify the solution
    sol = at * p + bt * q - ht
    sol = sol.simplify()
    assert sol == 0

    if 0:
        """
        Step 3: What does this step actually do?
        Because currently it does nothing.
        """
        n1, d1 = (p / ht).as_numer_denom()
        n2, d2 = (q / ht).as_numer_denom()

        gcd = sp.gcd(d1, d2)
        # print(gcd)
        pt = n1 * d2 / gcd
        pt = pt.simplify()
        qt = n2 * d1 / gcd
        qt = qt.simplify()
        ht = d1 * d2 / gcd
        ht = ht.simplify()
    else:
        pt = p
        qt = q

    # print("New coefficients pt, qt ,ht:")
    # print(pt, qt, ht)

    # verify the solution
    sol = at * pt + bt * qt - ht
    sol = sol.expand()
    sol = sol.simplify()
    assert sol == 0

    """
    Step 4: Iteratively reduce the RHS by including factors from ht into a and b
    """
    # TODO: When pn and qb are zero, choose the less convoluted one to keep the expressions small
    lc = sp.Poly(ht, p_gen).LC()
    a_s = at / lc
    b_s = bt / lc
    h_s = ht / lc

    rts = sp.roots(ht, p_gen)
    for r, m in rts.items():
        # print(f"\nProcessing root at s={r}:")
        for o in range(m):
            # print(f"Processing Multiplicity {o + 1}/{m}:")
            f = p_gen - r
            qb = qt.subs(s_gen, s_gen_def).subs(p_gen, r)
            pb = pt.subs(s_gen, s_gen_def).subs(p_gen, r)
            if qb == pb == 0:
                # common zero -> there is nothing to be done
                # print("Common Root")
                continue
            elif pb != 0:
                # print("Root is no part of pt, using that")
                bb = b_s.subs(s_gen, s_gen_def).subs(p_gen, r)
                alpha = bb / pb
            elif qb != 0:
                # print("Root is no part of qt, using that")
                ab = a_s.subs(s_gen, s_gen_def).subs(p_gen, r)
                alpha = ab / qb
            else:
                raise NotImplementedError("This should not happen")

            a_s = (a_s + alpha * qt) / f
            b_s = (b_s - alpha * pt) / f
            h_s = h_s / f

            if 0:
                print("Symplifying")
                a_s = a_s.simplify()
                b_s = b_s.simplify()
                h_s = h_s.simplify()

            if 0:
                print("New coefficients as,bs and hs:")
                print(a_s, b_s, h_s)
                break

    # print(a_s)
    # print(b_s)
    # print(h_s)
    # print("Finished.")

    # verify the solution
    if h_s.simplify() != 1:
        raise AssertionError(f"GCD of {h_s} is not equal to one")
    sol = a_s * pt + b_s * qt - 1
    sol = sol.expand()
    sol = sol.simplify()
    assert sol == 0

    # verify the solution
    sol = a_s * p + b_s * q - 1
    sol = sol.expand().simplify()
    assert sol == 0

    return a_s, b_s


def hermite_form(mat: sp.Matrix, p_gen: sp.Symbol, s_gen: sp.Symbol):
    """
    Compute the column hermite form of a given matrix M via a right multiplication with
    an unimodular matrix U

    .math::
        M U = S

    Returns: Tuple of Transformation and result matrices U and S
    """
    u_mat = sp.Matrix.eye(mat.cols)
    ui_mat = sp.Matrix.eye(mat.cols)
    mt = mat
    r_dim = mat.rows
    for i in range(r_dim):
        # print(f"Processing row {i}")
        """
        Step 1:

        Normalize the diagonal element and zero another entry
        """
        # print("Step 1")
        a = mt[i, i]
        for k in range(i, mt.cols):
            # print(f"Trying to use k={k}")
            if mt[i, k] == 0 or k == i:
                # and p_gen not in mt[i,k].free_symbols
                continue
            b = mt[i, k]
            try:
                p, q = bezouts(a, b, p_gen=p_gen, s_gen=s_gen)
                break
            except AssertionError:
                continue
        else:
            raise ValueError("No element found to continue reduction")

        _u = sp.Matrix.eye(*u_mat.shape)
        _u[i, i] = p
        _u[k, i] = q
        _u[i, k] = -b
        _u[k, k] = a
        u_mat = u_mat @ _u
        u_mat.simplify()
        ui_mat = _u.inv() @ ui_mat
        ui_mat.simplify()
        mt = mt @ _u
        mt.simplify()
        # sp.pprint(_u)
        # sp.pprint(mt)

        """
        Step 2:
        use diagonal entry to eliminate all other entries
        """
        # print("Step 2")
        _u = sp.Matrix.eye(*u_mat.shape)
        for j in range(mt.cols):
            if j == i or mt[i, j] == 0:
                # those columns are already done
                continue
            _u[i, j] = -mt[i, j]
        u_mat = u_mat @ _u
        u_mat.simplify()
        ui_mat = _u.inv() @ ui_mat
        ui_mat.simplify()
        mt = mt @ _u
        mt.simplify()
        # sp.pprint(_u)
        # sp.pprint(mt)

        ud = u_mat.det().simplify()
        assert ud == 1

    u_mat.simplify()

    return mt, u_mat, ui_mat


def gpld(
    x: Union[GenPoly],
    y: Union[GenPoly],
    verbose: bool = False,
) -> tuple[GenPoly, GenPoly]:
    """
    Perform generalized polynomial long division

    This function will compute generalized polynomials  q and r,
    such that x = qy +r with deg(r) < deg(y).

    Note:
        No care is taken of the coefficients and it is assumed that they
        support division.

    Args:
        x (GenPoly): Expression to be divided
        y (GenPoly): Expression to divide by
        verbose (bool): If True, be verbose about what steps are taken

    Returns:
        Tuple (q,r) of generalized quotient polynomial and remainder.
    """
    if not isinstance(x, GenPoly) or not isinstance(y, GenPoly):
        raise ValueError("Only arguments of type GenPoly supported")
    assert x.gen == y.gen
    assert y != 0

    y_dmax = y.degmax()
    y_dmin = y.degmin()

    q = GenPoly(0, x.gen)
    r = x
    while True:
        r_dmax = r.degmax()
        r_dmin = r.degmin()

        degmax_correct = sp.assumptions.ask(r_dmax <= y_dmax)
        degmin_correct = sp.assumptions.ask(r_dmin >= y_dmin)

        if degmax_correct is None or degmin_correct is None:
            raise ValueError(
                "Degree order cannot be decided. Make sure that"
                "the given expressions exponents to not contain"
                "any symbols."
            )

        if verbose:
            print()
            print(f"Current pred degree: {r_dmax} < {y_dmax}? {degmax_correct}")
            print(f"Current delay degree: {r_dmin} > {y_dmin}? {degmin_correct}")

        if not degmax_correct:
            # reduce predictions
            q_kd = {(r_dmax - y_dmax): r.LC() / y.LC()}
        elif not degmin_correct:
            # reduce delays
            q_kd = {(r_dmin - y_dmin): r.TC() / y.TC()}
        else:
            break
        q_k = GenPoly.from_dict(q_kd, y.gen)

        r = r - q_k * y
        q += q_k

    return q, r


def make_entire(
    expr: GenPoly,
    rat_sym: sp.Symbol,
    gen_expr: sp.Expr = None,
) -> GenPoly:
    """
    Convert a generalised polynomial in *sig_gen* with rational coefficients
    in *sym* into a version without singularities

    As the result is a whole function it can therefore be interpreted as the
    Laplace transform of a shift operator with compact support.

    Args:
        expr (GenPoly): Expression to work on
        rat_sym (sp.Symbol): Free variable of the rational coefficients
        gen_expr (sp.Expr): Expression in sym, that will be substituted for *gp_gen*

    Returns:
        GenPoly: The resulting expression without singularities (entire function in rat_sym)
    """
    if gen_expr is None:
        gen_expr = sp.exp(rat_sym)
    if rat_sym not in gen_expr.free_symbols:
        raise ValueError(
            "The expression to substitute for the GenPoly generator must "
            "contain the free variable of the rational coefficients."
        )

    # extract the common denominator
    num, den = expr.as_numer_denom()

    # factorise the denominator, extract leading coefficient for normalisation
    p_den = sp.Poly(den, rat_sym)
    lc = p_den.LC()
    rts = sp.roots(p_den, rat_sym)

    # start without denominator and iteratively add zeroes
    entire_gpoly = GenPoly(num / lc, gen_expr)
    for r, m in rts.items():
        for k in range(m):
            gp_at_r = sp.limit(entire_gpoly.as_expr(), rat_sym, r)
            entire_gpoly = (entire_gpoly - gp_at_r) / (rat_sym - r)

    # add offset term to original expression
    p = entire_gpoly.as_dict().get(0, sp.S.Zero)
    return expr + p


def gplde(
    x: Union[GenPoly],
    y: Union[GenPoly],
    c_gen: sp.Symbol,
    verbose: bool = False,
) -> tuple[GenPoly, GenPoly]:
    """
    Perform  modified generalized polynomial long division

    This function will compute generalized polynomials q and r,
    such that x = qy +r with deg(r) = deg(y).

    Note:
        Care is taken, such that the resulting q and r remain whole functions

    Args:
        x (GenPoly): Expression to be divided
        y (GenPoly): Expression to divide by
        c_gen (sp.Symbol): Generator of the coefficient polynomials
        verbose (bool): If True, be verbose about what steps are taken

    Returns:
        Tuple (q,r) of generalized quotient polynomial and remainder.
    """
    q, r = gpld(x, y, verbose)
    if q.is_zero:
        return q, r
    qe = make_entire(q, c_gen)
    p = qe.as_dict().get(0, sp.S.Zero)
    re = r - p * y
    return qe, re


def get_column_reduction_transform(mat, col, p_gen, gp_gen=None):
    l_mat = sp.Matrix.eye(mat.shape[0])
    for i in range(col + 1, mat.shape[0]):
        x = GenPoly(mat[i, col], gp_gen)
        y = GenPoly(mat[col, col], gp_gen)
        # x = mat[i, col]
        # y = mat[col, col]
        qe, re = gplde(x, y, p_gen)
        l_mat[i, col] = -qe.as_expr()
        # l_mat[i, col] = -qe
    return l_mat


def reduce_matrix_shifts(mat: sp.Matrix, p_gen: sp.Symbol, gp_gen: sp.Symbol):
    # gp_mat = mat.applyfunc(lambda x: GenPoly(x, gp_gen))
    gp_mat = mat.copy()
    check_canonical_form(gp_mat, p_gen, gp_gen)

    l_mat = sp.Matrix.eye(mat.shape[0])
    for i in range(1, mat.shape[0]):
        print(f"Reducing row {i}")
        for j in range(mat.shape[0] - 1):
            print(f"Reducing col {j}")
            lc = get_column_reduction_transform(gp_mat, j, p_gen, gp_gen)
            gp_mat = lc @ gp_mat
            l_mat = lc @ l_mat
            if 0:
                # gp_mat = gp_mat.applyfunc(sp.simplify)
                gp_mat = gp_mat.applyfunc(lambda x: GenPoly(x, gp_gen))
                # gp_mat = gp_mat.applyfunc(lambda x: GenPoly(x, gp_gen).as_expr())
            # l_mat = (lc @ l_mat).applyfunc(lambda x: GenPoly(x, gp_gen))

    return gp_mat, l_mat


def canonical_form(
    mat: sp.Matrix, p_gen: sp.Symbol, gp_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Apply row and colum permutations to bring mat into canonical form
    """
    m, l_mat = sort_shift_intervals(mat, gp_gen)
    m, r_mat = move_col_degrees_on_diagonal(m, p_gen)
    return l_mat, m, r_mat


def sort_shift_intervals(
        mat: sp.Matrix, gp_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Compute a permutation that sorts the leading row degrees w.r.t shifts ascendingly
    """
    # extract row degrees
    (_, ld), (_, td) = get_matrix_terms(mat, gp_gen)
    cd = ld - td
    r_degs = [max(cd[r, :]) for r in range(cd.rows)]

    # sort them
    r_pair = list(enumerate(r_degs))
    r_pair.sort(key=lambda x: x[1])
    r_order, _ = zip(*r_pair)

    # build permutation matrix
    perm = sp.combinatorics.Permutation(r_order)
    l_mat = sp.PermutationMatrix(perm).as_explicit()

    return l_mat @ mat, l_mat


def move_col_degrees_on_diagonal(
    mat: sp.Matrix, p_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Compute a permutation that moves the highest derivatives from each column on the main diagonal
    """
    # extract column degrees
    cffs, degs = get_leading_term_matrix(mat, p_gen)
    c_degs = [max(degs[:, c]) for c in range(degs.cols)]

    # check compatibilities
    r_compat = [set() for c in range(degs.cols)]
    for r in range(degs.rows):
        for c in range(degs.cols):
            if degs[r, c] == c_degs[c]:
                r_compat[c].add(r)
    if any([not s for s in r_compat]):
        raise ValueError(
            "Properties not obtainable by reordering, Matrix is probably not column reduced."
        )

    # sort entries first that only fit a few rows
    col_compat = list(zip(list(range(degs.cols)), r_compat))
    col_compat.sort(key=lambda x: len(x[1]))

    # select columns
    new_cols = [None] * mat.cols
    for r in range(mat.rows):
        for idx, (c, adm_rs) in enumerate(col_compat):
            if r in adm_rs:
                new_cols[r] = c
                break
        else:
            raise ValueError
        col_compat.pop(idx)

    # build permutation matrix
    perm = sp.combinatorics.Permutation(new_cols)
    r_mat = sp.PermutationMatrix(perm).as_explicit().T

    return mat @ r_mat, r_mat


def check_canonical_form(mat: sp.Matrix, p_gen: sp.Symbol, gp_gen: sp.Symbol):
    """
    Check if a matrix m is in canonical form

    This includes two checks:
    - the highest derivative degrees in each column must occur on the main diagonal
    - the shift diameters are increasing for each row
    """
    # 1: highest derivatives on the main diagonal
    check_col_degrees_on_diagonal(mat, p_gen)

    # 2: shift degrees are increasing row-wise
    (l_cffs, l_degs), (t_cffs, t_degs) = get_matrix_terms(mat, gp_gen)
    c_degs = l_degs - t_degs
    for j in range(mat.rows - 1):
        diff = c_degs[j + 1, :] - c_degs[j, :]
        if not all(diff[0] == d for d in diff):
            raise ValueError(
                f"Given matrix does not exhibit the same shift differences for all columns between row {j} and {j + 1}."
            )
        if diff[0] < 0:
            raise ValueError(
                f"Given matrix does not exhibit increasing shift intervals in between row {j} and {j + 1}."
            )


def check_col_degrees_on_diagonal(mat: sp.Matrix, p_gen: sp.Symbol) -> Any:
    cffs, degs = get_leading_term_matrix(mat, p_gen)
    for j in range(degs.shape[1]):
        if degs[j, j] != max(degs[:, j]):
            raise ValueError(
                f"Given matrix does not have highest degrees w.r.t {p_gen} on the main diagonal "
                f"in column {j}."
            )
    return True


def eliminate_delays_ur_post(
    mat: sp.Matrix, p_gen: sp.Symbol, s_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Eliminate all delays right of the main diagonal
    """
    r_mat = sp.Matrix.eye(mat.shape[1])
    for r in range(mat.shape[0]):
        pivot = GenPoly(mat[r, r], s_gen)
        for c in range(r + 1, mat.shape[1]):
            elem = GenPoly(mat[r, c], s_gen)
            quo, rem = gplde(elem, pivot, p_gen)
            r_mat[r, c] = -quo.as_expr()
    res = mat @ r_mat
    res.simplify()
    return res, r_mat


def eliminate_delays(
    h_mat: sp.Matrix, del_mat: sp.Matrix, p_gen: sp.Symbol, s_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Compute transformation that will eliminate all delays in a given system

    Parameters
    ----------
    h_mat
    del_mat
    p_gen
    s_gen

    Returns
    -------
    ht_mat, r_mat, ri_mat
    """
    # print("Step 0: Original matrices")
    # print(h_mat)
    # print(del_mat)

    # print("Step 1a: Echelon form")
    h1, r1i = echelon_form(h_mat, p_gen, mode="post")
    # print(h1)
    # print(r1i)

    # print("Step 1b: Factor out delay matrix")
    h1p = factor_diag_post(del_mat @ h1, del_mat)
    # print(h1p)

    # print("Step 2: Eliminate delays")
    h2, r2i = eliminate_delays_ur_post(h1p, p_gen, s_gen)
    # print(h2)
    # print(r2i)

    # print("Step 3: Col reduced form")
    ht, r3i = column_reduced_form(h2, p_gen)
    # print(ht)
    # print(r3i)

    # print("Result: Transform for y:")
    ri_mat = r1i @ del_mat.inv() @ r2i @ r3i
    r_mat = r3i.inv() @ r2i.inv() @ del_mat @ r1i.inv()
    # print(r_mat)

    # check consistency
    # print("Error:")
    err = del_mat @ h_mat - ht @ r_mat
    err = err.applyfunc(sp.simplify)
    # print(err)
    assert err == sp.Matrix.zeros(*err.shape)

    return ht, r_mat, ri_mat
