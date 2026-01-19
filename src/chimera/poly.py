import sympy as sp

from sympy import Matrix
from sympy.polys.polytools import Poly, gcdex
from sympy.polys.domains import QQ


def hermite_form1(mat: sp.Matrix, s_gen: sp.Symbol):
    """
    Compute the row hermite form W of a polynomial matrix A

    This will compute an unimodular matrix U such that

    .math::
        W = A U

    where W is in hermite form.

    Parameters
    ----------
    mat: Polynomial matrix
    s_gen: Generator of the polynomial

    Returns
    -------
    Tuple of matrices B and U
    """
    a_mat = mat.copy()
    w_mat = sp.Matrix.zeros(*a_mat.shape)
    # 1 Init
    m, n = a_mat.shape
    i = m - 1
    j = n - 1
    k = n - 1
    if m <= n:
        l = 1 - 1
    else:
        l = m - n + 1 - 1

    while True:
        # 2 Check zero
        if j > 0:
            j -= 1
            if a_mat[i, j] == 0:
                continue

            # 3 Euclidian step
            u, v, d = sp.gcdex(a_mat[i, k], a_mat[i, j])
            b_vec = u * a_mat[:, k] + v * a_mat[:, j]
            a_mat[:, j] = (a_mat[i, k] / d) * a_mat[:, j] - (a_mat[i, j] / d) * a_mat[
                :, k
            ]
            a_mat[:, k] = b_vec
        else:  # j == 1 - 1
            # 4 Final reductions
            b = a_mat[i, k]
            if b < 0:
                a_mat[:, k] = -a_mat[:, k]
                b = -b
            if b == 0:
                k += 1
                # 5 Finished?
                if i == l:
                    for j in range(n - k + 1):
                        w_mat[:, j] = a_mat[:, j] = a_mat[:, j + k - 1]
                    break
                else:
                    i -= 1
                    k -= 1
                    j = k
            else:
                for r in range(k + 1, n):
                    q = a_mat[i, r] / b
                    a_mat[:, r] = a_mat[:, r] - q * a_mat[:, k]

    return w_mat


def hermite_form2(M: sp.Matrix, s_gen: sp.Symbol):
    """
    Copied form sympy
    """
    # We work one row at a time, starting from the bottom row, and working our
    # way up.
    m, n = M.shape
    A = M.copy()
    # Our goal is to put pivot entries in the rightmost columns.
    # Invariant: Before processing each row, k should be the index of the
    # leftmost column in which we have so far put a pivot.
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            # This case can arise when n < m and we've already found n pivots.
            # We don't need to consider any more rows, because this is already
            # the maximum possible number of pivots.
            break
        k -= 1
        # k now points to the column in which we want to put a pivot.
        # We want zeros in all entries to the left of the pivot column.
        for j in range(k - 1, -1, -1):
            if A[i, j] != 0:
                # Replace cols j, k by lin combs of these cols such that, in row i,
                # col j has 0, while col k has the gcd of their row i entries. Note
                # that this ensures a nonzero entry in col k.
                u, v, d = sp.gcdex(A[i, k], A[i, j])
                r, s = A[i, k] / d, A[i, j] / d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i, k]
        # Do not want the pivot entry to be negative.
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # The pivot entry will be 0 iff the row was 0 from the pivot col all the
        # way to the left. In this case, we are still working on the same pivot
        # col for the next row. Therefore:
        if b == 0:
            k += 1
        # If the pivot entry is nonzero, then we want to reduce all entries to its
        # right in the sense of the division algorithm, i.e. make them all remainders
        # w.r.t. the pivot as divisor.
        else:
            for j in range(k + 1, n):
                q = A[i, j] / b
                add_columns(A, j, k, 1, -q, 0, 1)
    # Finally, the HNF consists of those columns of A in which we succeeded in making
    # a nonzero pivot.
    W = sp.Matrix.zeros(*A.shape)
    for j in range(n - k + 1):
        W[:, j] = A[:, j + k - 1]

    return W


def add_columns(m, i, j, a, b, c, d):
    # replace m[:, i] by a*m[:, i] + b*m[:, j]
    # and m[:, j] by c*m[:, i] + d*m[:, j]
    for k in range(m.shape[0]):
        e = m[k, i]
        m[k, i] = a * e + b * m[k, j]
        m[k, j] = c * e + d * m[k, j]


def hermite_form(A, x, domain=QQ):
    """
    Made up by Chat gpt

    Compute the row Hermite normal form H of a matrix A with entries in F[x],
    where F is a field (default: QQ), and a unimodular U over F[x] such that:
        H = U * A

    Parameters
    ----------
    A : sympy.Matrix
        Matrix with entries that are polynomials (or expressions) in x.
    x : sympy.Symbol
        The polynomial indeterminate.
    domain : sympy Domain, optional
        Base field domain for coefficients (default QQ). Must be a field.

    Returns
    -------
    H : sympy.Matrix  (entries are polynomial expressions in x)
    U : sympy.Matrix  (unimodular over F[x])

    Notes
    -----
    - The algorithm uses:
        1) row swaps,
        2) row addition by polynomial multiples,
        3) 2x2 unimodular transformations based on extended GCD to clear a column.
      These preserve unimodularity over F[x].
    - Pivots are normalized to be monic (allowed because domain is a field).
    """
    m, n = A.shape

    # Convert to polynomial entries in the chosen domain
    def to_poly(a):
        if isinstance(a, Poly):
            if a.gens != (x,):
                a = Poly(a.as_expr(), x, domain=domain)
            else:
                a = a.set_domain(domain)
            return a
        return Poly(a, x, domain=domain)

    P = [[to_poly(A[i, j]) for j in range(n)] for i in range(m)]
    # U starts as identity over F[x]
    U = [
        [
            Poly(1, x, domain=domain) if i == j else Poly(0, x, domain=domain)
            for j in range(m)
        ]
        for i in range(m)
    ]

    def row_swap(M, i, j):
        M[i], M[j] = M[j], M[i]

    def row_lincomb(M, i, j, coeff):  # row_i <- row_i + coeff*row_j
        for col in range(len(M[i])):
            M[i][col] = M[i][col] + coeff * M[j][col]

    def row_scale_constant(M, i, c):  # scale by nonzero constant in the field
        if c == 0:
            raise ValueError("Scaling by zero is not allowed.")
        for col in range(len(M[i])):
            M[i][col] = (M[i][col] * c).set_domain(domain)

    r = 0  # current pivot row
    for c in range(n):
        if r >= m:
            break
        # Find a row with nonzero entry in column c at or below r
        pivot_row = None
        for i in range(r, m):
            if not P[i][c].is_zero:
                pivot_row = i
                break
        if pivot_row is None:
            continue

        # Swap to bring pivot candidate to row r
        if pivot_row != r:
            row_swap(P, r, pivot_row)
            row_swap(U, r, pivot_row)

        # Clear entries below in column c using 2x2 unimodular transformations
        i = r + 1
        while i < m:
            if P[i][c].is_zero:
                i += 1
                continue

            a = P[r][c]
            b = P[i][c]
            # Extended GCD: u*a + v*b = g
            u, v, g = gcdex(a, b, x, domain=domain)
            # Normalize g to be monic
            # lc(g) is a nonzero constant in the field
            lc_g = g.LC()
            u = to_poly(u / lc_g)
            v = to_poly(v / lc_g)
            g = to_poly(g / lc_g)

            # Compute a/g and b/g (exact division in the polynomial ring)
            ag = a.quo(g)
            bg = b.quo(g)

            # 2x2 unimodular block M = [[u, v], [ -b/g, a/g ]]
            # Apply to the pair of rows (r, i) on both P and U
            def apply_2x2_rows(Mmat, r_row, i_row, u_, v_, neg_bg, ag_):
                # new_r = u_*old_r + v_*old_i
                # new_i = neg_bg*old_r + ag_*old_i
                old_r = [entry for entry in Mmat[r_row]]
                old_i = [entry for entry in Mmat[i_row]]
                new_r = [u_ * rr + v_ * ii for rr, ii in zip(old_r, old_i)]
                new_i = [neg_bg * rr + ag_ * ii for rr, ii in zip(old_r, old_i)]
                Mmat[r_row] = new_r
                Mmat[i_row] = new_i

            apply_2x2_rows(P, r, i, u, v, -bg, ag)
            apply_2x2_rows(U, r, i, u, v, -bg, ag)

            # Now P[i][c] should be zero by construction
            i += 1

        # Normalize pivot to be monic: scale row r by 1/LC(pivot) (constant)
        pivot = P[r][c]
        if pivot.is_zero:
            # Should not happen here, but guard anyway
            continue
        lc = pivot.LC()  # constant in the base field
        if lc != 1:
            inv_lc = domain.convert(1) / lc
            row_scale_constant(P, r, inv_lc)
            row_scale_constant(U, r, inv_lc)

        # Reduce entries above the pivot in column c: make them zero using division
        pivot = P[r][c]  # monic now
        for i in range(r):
            if P[i][c].is_zero:
                continue
            q, rem = P[i][c].div(pivot)  # P[i][c] = q*pivot + rem
            # row_i <- row_i - q * row_r, so entry becomes 'rem'
            if not q.is_zero:
                # Apply to P
                for j in range(n):
                    P[i][j] = P[i][j] - q * P[r][j]
                # Apply to U
                for j in range(m):
                    U[i][j] = U[i][j] - q * U[r][j]
            # Because pivot is monic, we can then subtract rem*? Actually we want exact zero:
            # In Hermite normal form (row variant), entries above pivot are reduced to degree < deg(pivot).
            # The above operation already achieves that reduction (rem has deg < deg(pivot)).
            # If you want strict zeros above, do an extra subtraction of rem/pivot times pivot row,
            # but that would require a non-polynomial factor if rem != 0. So we keep 'reduced' form.
            # If you prefer zeros above, iterate column-by-column so that P[i][c] becomes exactly zero
            # via repeated 2x2 steps; here we follow the standard 'reduced' condition.

        r += 1

    # Build sympy Matrices as expressions
    H = Matrix([[p.as_expr() for p in row] for row in P])
    Umat = Matrix([[u.as_expr() for u in row] for row in U])
    return H, Umat


def is_in_row_hermite_form(H, x, domain=QQ):
    """
    Check whether a polynomial matrix H over F[x] (F a field) is in row Hermite normal form.

    Conditions checked:
      1) Zero rows (if any) are at the bottom.
      2) Each nonzero row has a pivot at its leftmost nonzero entry.
      3) Pivot columns strictly increase down the rows.
      4) Each pivot is monic.
      5) Entries to the left of the pivot in its row are zero.
      6) Entries below the pivot in its column are zero.
      7) Entries above the pivot in its column have degree < degree(pivot).

    Parameters
    ----------
    H : sympy.Matrix
        Matrix with polynomial entries (or expressions) in x.
    x : sympy.Symbol
        The polynomial indeterminate.
    domain : sympy Domain, optional
        Base field for coefficients (default QQ).

    Returns
    -------
    bool
        True if H satisfies the row Hermite normal form conditions, False otherwise.
    """
    m, n = H.shape

    # Convert to Poly objects
    def to_poly(a):
        if isinstance(a, Poly):
            if a.gens != (x,):
                a = Poly(a.as_expr(), x, domain=domain)
            else:
                a = a.set_domain(domain)
            return a
        return Poly(a, x, domain=domain)

    P = [[to_poly(H[i, j]) for j in range(n)] for i in range(m)]

    # Helper: polynomial degree (None for zero)
    def poly_deg(p):
        return None if p.is_zero else p.degree()

    pivot_cols = []
    zero_row_started = False

    for i in range(m):
        # Find pivot (leftmost nonzero) in row i
        pivot_j = None
        for j in range(n):
            if not P[i][j].is_zero:
                pivot_j = j
                break

        if pivot_j is None:
            # Zero row
            zero_row_started = True
            continue
        else:
            # If we saw a zero row earlier, nonzero row afterward violates HNF
            if zero_row_started:
                return False

        # Entries to the left of pivot must be zero
        for j in range(pivot_j):
            if not P[i][j].is_zero:
                return False

        pivot = P[i][pivot_j]
        # Pivot must be monic
        if pivot.LC() != domain.convert(1):
            return False

        # Pivot columns strictly increasing
        if pivot_cols and pivot_j <= pivot_cols[-1]:
            return False
        pivot_cols.append(pivot_j)

        # Check below-pivot zeros in the pivot column
        for k in range(i + 1, m):
            if not P[k][pivot_j].is_zero:
                return False

        # Check above-pivot reduction: degree(entry) < degree(pivot)
        dpiv = pivot.degree()
        for k in range(0, i):
            pk = P[k][pivot_j]
            if not pk.is_zero:
                dpk = pk.degree()
                if dpk >= dpiv:
                    return False

    return True
