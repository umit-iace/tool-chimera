"""
Custom algorithms for the project
"""

from copy import copy
from typing import Union

import sympy as sp

__all__ = ["gpld", "make_entire", "gplde", "gcdexs", "bezouts"]

from .genpoly import GenPoly


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
        q = q.to_expr()

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
    # display(a, b, h)

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
    # display(lcm)

    at = a * lcm
    at = at.simplify()
    bt = b * lcm
    bt = bt.simplify()
    ht = h * lcm
    ht = ht.simplify()
    # print("New coefficients a, b, h:")
    # display(at, bt, ht)

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
        # display(gcd)
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
    # display(pt, qt, ht)

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
                # Test: b_s should not have a pole at r
                display(b_s)
                bsn, bsd = b_s.subs(pred, sp.exp(s)).as_numer_denom()
                display(bsn, bsd)
                display(bsn.subs(s, r), bsd.subs(s, r))
                assert bsn.subs(s, r) == bsd.subs(s, r) == 0

            if 0:
                print("New coefficients as,bs and hs:")
                display(a_s)
                display(b_s)
                display(h_s)
                break

    # display(a_s)
    # display(b_s)
    # display(h_s)
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
    x: Union[sp.Expr, GenPoly],
    y: Union[sp.Expr, GenPoly],
    gen: Union[sp.Symbol, sp.Expr],
    verbose: bool = False,
) -> tuple[sp.Expr, sp.Expr]:
    """
    Perform generalized polynomial long division

    This function will compute generalized polynomials  q and r,
    such that x = qy +r with deg(r) < deg(y).

    Note:
        No care is taken of the coefficients and it is assumed that they
        support division.

    Args:
        x (sp.Expr): Expression to be divided
        y (sp.Expr): Expression to divide by
        gen (sp.Symbol,sp.Expr): Generator of the generalized polynomials
        verbose (bool): If True, be verbose about what steps are taken

    Returns:
        Tuple (q,r) of generalised quotient polynomial and remainder.
    """
    assert y != 0
    if not isinstance(y, GenPoly):
        y = GenPoly(y, gen)
    else:
        if isinstance(x, GenPoly):
            assert x.gen == y.gen
            r = x.to_expr()
        else:
            r = copy(x)

    y_dmax = y.degmax()
    y_dmin = y.degmin()

    q = sp.sympify(0)
    r = copy(x)
    while True:
        r = GenPoly(r.simplify(), y.gen)
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
            q_k = r.LC() / y.LC() * y.gen ** (r_dmax - y_dmax)
        elif not degmin_correct:
            # reduce delays
            q_k = r.TC() / y.TC() * y.gen ** (r_dmin - y_dmin)
        else:
            break

        r = r.to_expr() - q_k * y.to_expr()
        q += q_k

    return q.simplify(), r.to_expr()


def make_entire(
    expr: sp.Expr,
    rat_sym: sp.Symbol,
    gp_gen: Union[sp.Symbol, sp.Expr],
    gen_expr: sp.Expr,
):
    """
    Convert a generalized polynomial in *sig_gen* with rational coefficients
    in *sym* into a version without singularities

    As the result is a whole function it can therefore be interpreted as the
    Laplace transform of a shift operator with compact support.

    Args:
        expr (sp.Expr): Expression to work on
        rat_sym (sp.Symbol): Free variable of the rational coefficients
        gp_gen (sp.Symbol): Generator of the generalized polynomial
        gen_expr (sp.Expr): Expression in sym, that will be substituted for *gp_gen*

    Returns:
        sp.Expr: The resulting expression without singularities (entire function in rat_sym)
    """
    # extract the common denominator
    if not expr.has(gp_gen):
        raise ValueError("Given expression does not contain any shifts")
    e_sig = GenPoly(expr, gp_gen)
    exps = e_sig.exponents()
    nums, den = e_sig.as_numers_denom()

    # factorize the denominator, extract leading coefficient
    p_den = sp.Poly(den, rat_sym)
    lc = p_den.LC()
    rts = sp.roots(p_den, rat_sym)

    # start without denominator and iteratively add zeroes
    if rat_sym not in gen_expr.free_symbols:
        raise ValueError(
            "The expression to substitute for the GenPoly generator must"
            "contain the free variable of the rational coefficients."
        )
    entire_expr = sp.Add(*[num / lc * gen_expr**exp for num, exp in zip(nums, exps)])
    for r, m in rts.items():
        wes = entire_expr.subs(rat_sym, r)
        entire_expr = (entire_expr - wes) / (rat_sym - r) ** m

    # re-Subs the generator expression
    w = sp.Wild("w", exclude=[rat_sym])
    r = entire_expr.replace(gen_expr**w, gp_gen**w)

    return r


def gplde(
    x: Union[sp.Expr, GenPoly],
    y: Union[sp.Expr, GenPoly],
    c_gen: sp.Symbol,
    gp_gen: sp.Symbol,
    gen_expr: sp.Expr,
    verbose: bool = False,
) -> tuple[sp.Expr, sp.Expr]:
    """
    Perform  modified generalized polynomial long division

    This function will compute generalized polynomials q and r,
    such that x = qy +r with deg(r) = deg(y).

    Note:
        Care is taken, such that the resulting q and r remain whole functions

    Args:
        x (sp.Expr): Expression to be divided
        y (sp.Expr): Expression to divide by
        gp_gen (sp.Symbol): Generator of the generalized polynomials
        c_gen (sp.Symbol): Generator of the coefficient polynomials
        gen_expr (sp.Expr): Expression in sym, that will be substituted for *gp_gen*
        verbose (bool): If True, be verbose about what steps are taken

    Returns:
        Tuple (q,r) of generalized quotient polynomial and remainder.
    """
    q, r = gpld(x, y, gp_gen, verbose)
    if q == 0:
        return q, r
    qe = make_entire(q, c_gen, gp_gen, gen_expr)
    re = x - qe * y
    return qe, re


def get_column_reduction_transform(mat, col, p_gen, gp_gen, gp_expr):
    l_mat = sp.Matrix.eye(mat.shape[0])
    for i in range(col + 1, mat.shape[0]):
        qe, re = gplde(mat[i, col], mat[col, col], p_gen, gp_gen, gp_expr)
        l_mat[i, col] = -qe
    return l_mat


def reduce_matrix_shifts(mat: sp.Matrix, p_gen: sp.Symbol, gp_gen: sp.Symbol):
    bmat = copy(mat)
    l_mat = sp.Matrix.eye(mat.shape[0])
    for i in range(2, mat.shape[0]):
        for j in range(1, mat.shape[0] - 1):
            lc = get_column_reduction_transform(bmat, j, p_gen, gp_gen, sp.exp(p_gen))
            bmat = lc @ bmat
            l_mat = lc @ l_mat

    return bmat, l_mat
