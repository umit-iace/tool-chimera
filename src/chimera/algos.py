"""
Custom algorithms for the project
"""

import logging
from typing import Union, Tuple, Any

import sympy as sp
import numpy as np

from .poly import echelon_form, column_reduced_form
from .qpoly import QPoly
from .utils import (
    get_matrix_terms,
    get_row_degrees,
    get_leading_term_matrix,
    get_leading_row_degrees,
    get_lccm,
    factor_diag_post,
)

__all__ = ["qplde", "make_entire", "qpld_single", "qplde_single", "eliminate_delays", "reduce_matrix_shifts"]

logger = logging.getLogger(__name__)


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
        q, r = qplde(h[idx - 1], h[idx], gp_gen)
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
    lcm_a = QPoly(a, s_gen).get_coeff_denom_lcm()
    lcm_b = QPoly(b, s_gen).get_coeff_denom_lcm()
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


def qplde(mat: sp.Matrix, row: int, c_gen: sp.Symbol) -> tuple[sp.Matrix, sp.Matrix]:
    """
    Perform modified quasipolynomial long division

    This function will compute vectors of quasipolynomials q and r
    such that the one row of the given matrix (called x in the following)
    will be reduced by means of the elements above it.

    Note:
        Care is taken, such that the resulting components in
        q and r remain whole functions.

    Args:
        mat (Matrix of QPolys): Matrix to be "divided"
        row (int): Row of the matrix to be divided
        c_gen (sp.Symbol): Generator of the coefficient polynomials

    Returns:
        Tuple (q,r) of quotient quasipolynomial and remainder vectors.
    """
    for elem in mat.iter_values():
        assert isinstance(elem, QPoly)
        assert elem.gen == mat[0, 0].gen

    x = mat[row, :row].T
    y = mat.diagonal()[:row, :row].T
    logger.debug(
        f"Degrees of divisors: {[(q.degmin().evalf(), q.degmax().evalf()) for q in y]}"
    )

    q = np.array([[QPoly(0, _x.gen)] for _x in x])
    r = np.array(x.tolist())
    while True:
        logger.debug(
            f"Degrees of remainder: {[(q.degmin().evalf(), q.degmax().evalf()) for q in r[:, 0]]}"
        )

        # check column compatibilities
        compat_max = []
        compat_min = []
        for _y in y:
            dmax = []
            dmin = []
            for _r in r[:,0]:
                max_c = sp.assumptions.ask(_r.degmax() > _y.degmax())
                dmax.append(max_c)
                min_c = sp.assumptions.ask(_r.degmin() < _y.degmin())
                dmin.append(min_c)
                if max_c is None or min_c is None:
                    raise ValueError(
                        "Degree order cannot be decided. Make sure that"
                        "the given expressions exponents to not contain"
                        "any symbols."
                    )
            compat_max.append(all(dmax))
            compat_min.append(all(dmin))

        al = []
        bl = []
        cl = []
        rl = []
        for i in range(row):
            cl.append([mat[j, i] for j in range(row)])
            if compat_max[i]:
                al.append(
                    [mat[j, i].LM().as_expr() for j in range(row) if compat_max[j]]
                )
                bl.append(r[i,0].LM().as_expr())
            elif compat_min[i]:
                al.append(
                    [mat[j, i].TM().as_expr() for j in range(row) if compat_min[j]]
                )
                bl.append(r[i,0].TM().as_expr())
            else:
                continue
            rl.append([1 if i == j else 0 for j in range(row)])
        a_mat = sp.Matrix(al)
        b_vec = sp.Matrix(bl)
        rej_mat = sp.Matrix(rl).T
        c_mat = np.array(cl)

        if rej_mat.shape == (0, 0):
            logger.debug("Remainder small enough")
            # no further reduction necessary
            break

        q_r = rej_mat @ a_mat.inv() @ b_vec
        q_r = q_r.applyfunc(lambda x: QPoly(x, mat[0, 0].gen))
        q_e = q_r.applyfunc(lambda x: make_entire(x, c_gen)[0])
        # work around domain matrix mayhem
        q_k = np.array(q_e.tolist())
        r = r - c_mat @ q_k
        q = q + q_k

    return q, r


def make_entire(
    expr: QPoly,
    rat_sym: sp.Symbol,
    gen_expr: sp.Expr = None,
    verbose: bool = False,
) -> Tuple[QPoly, sp.Expr]:
    """
    Convert a generalised polynomial in *sig_gen* with rational coefficients
    in *sym* into a version without singularities

    As the result is a whole function it can therefore be interpreted as the
    Laplace transform of a shift operator with compact support.

    Args:
        expr (QPoly): Expression to work on
        rat_sym (sp.Symbol): Free variable of the rational coefficients
        gen_expr (sp.Expr): Expression in sym, that will be substituted for *gp_gen*

    Returns:
        GenPoly, Expr: Tuple of the resulting expression without singularities (entire function in rat_sym)
        as well as the correcting factor.
    """
    if expr.is_zero:
        return expr, sp.S.Zero
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
    p_den = sp.Poly(den.as_expr(), rat_sym)
    lc = p_den.LC()
    rts = sp.roots(p_den, rat_sym)

    # start without denominator and iteratively add zeroes
    eq = QPoly(num / lc, *expr.gens)
    for r, m in rts.items():
        logger.debug(f"Lifting root at {rat_sym}={r} with multiplicity {m}")

        # compute derivatives
        eq_k_ds = [eq.as_expr()]
        for i in range(1, m):
            eq_k_ds.append(eq_k_ds[-1].diff(rat_sym))

        # compute the limit at s = r
        eq_at_r = sp.Add(
            *[
                eq_k_ds[i].subs(rat_sym, r) * (rat_sym - r) ** i / sp.factorial(i)
                for i in range(m)
            ]
        )
        # include multiple roots in one step
        eq = (eq - eq_at_r) / (rat_sym - r) ** m

    # add offset term to original expression
    p = eq.as_dict().get(0, sp.S.Zero)
    return expr + p, p


def qplde_single(x, y, c_gen):
    q, r = qpld_single(x, y)
    if q.is_zero:
        return q, r
    qe, p = make_entire(q, c_gen)
    re = r - p * y
    return qe, re


def qpld_single(
    x: Union[QPoly],
    y: Union[QPoly],
    verbose: bool = False,
) -> tuple[QPoly, QPoly]:
    """
    Perform quasipolynomial long division

    This function will compute quasipolynomials  q and r,
    such that x = qy +r with deg(r) < deg(y).

    Note:
        No care is taken of the coefficients and it is assumed that they
        support division.

    Args:
        x (QPoly): Expression to be divided
        y (QPoly): Expression to divide by
        verbose (bool): If True, be verbose about what steps are taken

    Returns:
        Tuple (q,r) of quotient quasipolynomial and remainder.
    """
    if not isinstance(x, QPoly) or not isinstance(y, QPoly):
        raise ValueError("Only arguments of type GenPoly supported")
    assert x.gen == y.gen
    assert y != 0

    y_dmax = y.degmax()
    y_dmin = y.degmin()

    q = QPoly(0, x.gen)
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
        q_k = QPoly.from_dict(q_kd, y.gen)

        r = r - q_k * y
        q += q_k

    return q, r


def get_column_reduction_transform(mat, col, p_gen, q_gen=None):
    if col is not None:
        l_mat = sp.Matrix.eye(mat.shape[0])
        # reduce single column
        y = QPoly(mat[col, col], q_gen)
        for i in range(col + 1, mat.shape[0]):
            x = QPoly(mat[i, col], q_gen)
            qe, re = qplde_single(x, y, p_gen)
            l_mat[i, col] = -qe.as_expr()
    else:
        l_mat = np.eye(mat.shape[0], dtype=QPoly)
        # reduce multiple columns at once
        if not isinstance(mat[0,0], QPoly):
            mq = mat.applyfunc(lambda x: QPoly(x, q_gen))
        else:
            mq = mat.copy()
        for i in range(1, mat.shape[0]):
            qe, _ = qplde(mq, i, p_gen)
            l_mat[i, :i] = -qe.T
        return l_mat


def reduce_matrix_shifts(
    mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    check_canonical_form(mat, p_gen, q_gen)

    if 0:
        red_mat = mat.copy()
        l_mat = sp.Matrix.eye(mat.shape[0])
        row_degs = sorted(set(get_row_degrees(mat, q_gen)))
        diffs = [row_degs[i + 1] - row_degs[i] for i in range(len(row_degs) - 1)]
        it_cnt = sp.ceiling((max(row_degs) - min(row_degs)) / min(diffs))
        for i in range(it_cnt):
            print(f"Iteration {i}:")
            if len(row_degs) == mat.shape[0]:
                # unique shifts
                for j in range(mat.shape[1] - 1):
                    print(f"Reducing col {j}")
                    lc = get_column_reduction_transform(red_mat, j, p_gen, q_gen)
                    l_mat = lc @ l_mat
                    red_mat = lc @ red_mat
            if len(row_degs) < mat.shape[0]:
                # equivalent shifts
                lc = get_column_reduction_transform(red_mat, None, p_gen, q_gen)
                l_mat = lc @ l_mat
                red_mat = lc @ red_mat
            row_degs = sorted(set(get_row_degrees(red_mat, q_gen)))
            sp.pprint(red_mat.applyfunc(lambda x: QPoly(x, q_gen).degree().evalf()))
    else:
        # make sure we work with qpolys
        if not isinstance(mat[0,0], QPoly):
            mq = mat.applyfunc(lambda x: QPoly(x, q_gen))
        l_mat = get_column_reduction_transform(mq, None, p_gen, q_gen)

        # convert to numpy array to circumvent domain matrix problems
        red_mat = np.array(mq.tolist())
        red_mat = l_mat @ red_mat

        # convert back to sympy matrices
        red_mat = sp.Matrix(red_mat.tolist())
        l_mat = sp.Matrix(l_mat.tolist())

    return red_mat, l_mat


def canonical_form(
    mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Apply row and colum permutations to bring mat into canonical form

    In Detail:
    * Sort rows such that the shift intervals are ascending
    * Sort columns such that the highest poly degrees appear on the main diagonal
    * Homogenise the leading row delays
    """
    m, l_mat = sort_shift_intervals(mat, q_gen)
    m, hom_trafo = homogenise_leading_shifts(m, q_gen, mode="delay")
    m, r_mat = move_col_degrees_on_diagonal(m, p_gen, q_gen)
    return hom_trafo @ l_mat, m, r_mat


def sort_shift_intervals(
    mat: sp.Matrix, q_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Compute a permutation that sorts the leading row degrees w.r.t shifts ascendingly
    """
    # extract row degrees
    (_, ld), (_, td) = get_matrix_terms(mat, q_gen)
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
    mat: sp.Matrix,
    p_gen: sp.Symbol,
    q_gen: sp.Symbol,
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Compute a permutation that moves the highest derivatives from each column on the main diagonal
    """
    (qlc, qld), (qtc, qtd) = get_matrix_terms(mat, q_gen)
    qt_row_degs = [min(qtd[r, :]) for r in range(mat.shape[0])]
    # ql_row_degs = [max(qld[r, :]) for r in range(mat.shape[0])]

    # make sure that we work with homogenised delays
    assert all([lrd == qt_row_degs[0] for lrd in qt_row_degs])

    # extract column degrees but only look at coefficients of largest predictions
    plc, pld = get_leading_term_matrix(qlc, p_gen)
    p_col_degs = [max(pld[:, c]) for c in range(pld.cols)]

    # check compatibilities
    r_compat = [set() for c in range(pld.cols)]
    for r in range(pld.rows):
        for c in range(pld.cols):
            # element must have col degree
            if pld[r, c] != p_col_degs[c]:
                continue
            # element must leading row delayd degree
            if r < pld.rows - 1 and qtd[r,c] != qt_row_degs[r]:
                continue
            r_compat[c].add(r)
    if any([not s for s in r_compat]):
        raise ValueError(
            "Properties not obtainable by reordering, Matrix is probably not column reduced."
        )

    col_compat = list(zip(list(range(pld.cols)), r_compat))

    # sort entries first that only fit a few rows
    col_compat.sort(key=lambda x: len(x[1]))

    # select columns
    new_cols = [None] * mat.cols
    for i in range(mat.cols):
        c, adm_rs = col_compat[i]
        for r in adm_rs:
            if new_cols[r] is None:
                new_cols[r] = c
                # remove row for compatibility list and sort again
                for i in range(len(col_compat)):
                    col_compat[i][1].discard(r)
                col_compat.sort(key=lambda x: len(x[1]))
                break
        else:
            raise ValueError

    # build permutation matrix
    perm = sp.combinatorics.Permutation(new_cols)
    r_mat = sp.PermutationMatrix(perm).as_explicit().T

    return mat @ r_mat, r_mat


def homogenise_leading_shifts(mat, q_gen, mode):
    (_, ld), (_, td) = get_matrix_terms(mat, q_gen)
    if mode == "delay":
        r_degs = [min(td[r, :]) for r in range(mat.shape[0])]
    elif mode == "pred":
        r_degs = [max(ld[r, :]) for r in range(mat.shape[0])]
    else:
        raise ValueError(f"Unknown mode {mode}")

    mean = sum(r_degs) / len(r_degs)
    l_mat = sp.Matrix.diag([q_gen ** (mean - rd) for rd in r_degs])
    mh = l_mat @ mat
    mh.simplify()
    return mh, l_mat


def check_canonical_form(mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol):
    """
    Check if a matrix m is in canonical form

    This includes two checks:
    - the highest derivative degrees in each column must occur on the main diagonal
    - the shift diameters are increasing for each row
    """
    # 1: highest derivatives on the main diagonal
    check_col_degrees_on_diagonal(mat, p_gen, q_gen)

    # 2: shifts
    (l_cffs, l_degs), (t_cffs, t_degs) = get_matrix_terms(mat, q_gen)
    s_degs = l_degs - t_degs
    row_degs = [max(s_degs[r, :]) for r in range(mat.shape[0])]
    row_degs_m = [min(t_degs[r, :]) for r in range(mat.shape[0])]
    # row_degs_p = [max(l_degs[r,:]) for r in range(mat.shape[0])]

    # 2a: leading delays of each row must be the same
    if not all(row_degs_m[0] == rd for rd in row_degs_m):
        raise ValueError("Leading row delays are not equal. Homogenise first.")

    # 2b: leading shifts of each row must be ascending
    if not sorted(row_degs):
        raise ValueError("Leading row shift intervals must be ascending.")


def check_col_degrees_on_diagonal(
    mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol = None
) -> Any:
    if q_gen is not None:
        q_cffs, q_degs = get_leading_term_matrix(mat, q_gen)
        lccm = get_lccm(q_cffs, p_gen)
    else:
        lccm = get_lccm(mat, p_gen)

    if not all([lccm[i, i] != 0 for i in range(lccm.shape[0])]):
        raise ValueError(
            f"Given matrix does not have highest degrees w.r.t {p_gen} on the main diagonal."
        )
    return True


def eliminate_delays_ur_post(
    mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Eliminate all delays right of the main diagonal
    """
    r_mat = sp.Matrix.eye(mat.shape[1])
    for r in range(mat.shape[0]):
        pivot = QPoly(mat[r, r], q_gen)
        for c in range(r + 1, mat.shape[1]):
            elem = QPoly(mat[r, c], q_gen)
            quo, rem = qplde(elem, pivot, p_gen)
            r_mat[r, c] = -quo.as_expr()
    res = mat @ r_mat
    res.simplify()
    return res, r_mat


def eliminate_delays(
    h_mat: sp.Matrix, del_mat: sp.Matrix, p_gen: sp.Symbol, q_gen: sp.Symbol
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Compute transformation that will eliminate all delays in a given system

    Parameters
    ----------
    h_mat
    del_mat
    p_gen
    q_gen

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
    h2, r2i = eliminate_delays_ur_post(h1p, p_gen, q_gen)
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
