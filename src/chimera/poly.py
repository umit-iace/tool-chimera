import numpy as np
import sympy as sp
from sympy import Matrix, Basic
from sympy.polys.polytools import Poly
from sympy.polys.domains import QQ, EX
from sympy.polys.matrices import DomainMatrix

from .utils import is_col_reduced, get_lccm, get_leading_column_degrees

__all__ = ["echelon_form", "column_reduced_form"]


def row_hermite_form(A, x, domain=QQ):
    """
    TODO: Currently this does not work
    Compute the row Hermite normal form H of a matrix A with entries in F[x],
    where F is a field (default: QQ), and a unimodular U over F[x] such that:
        H = A * U

    Parameters
    ----------
    A : sp.Matrix
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
    R = domain.poly_ring(x)

    # Convert input to DomainMatrix over R
    if isinstance(A, DomainMatrix):
        P = A if A.domain == R else A.convert_to(R)
    else:
        P = DomainMatrix.from_Matrix(Matrix(A)).convert_to(R)

    # Store transformation steps
    U = DomainMatrix.eye(n, R)

    c = 0  # current pivot col
    for r in range(m):
        if c >= n:
            break
        # Find a row with nonzero entry in c at or below r
        pivot_col = None
        for i in range(r, m):
            if not P[i, c].is_zero():
                pivot_col = i
                break
        if pivot_col is None:
            # all entries in this col are already zero
            continue

        # Swap to bring pivot candidate to col c
        if pivot_col != c:
            P = col_swap(P, c, pivot_col)
            U = col_swap(U, c, pivot_col)

        # Clear entries below the pivot in column c
        i = r + 1
        while i < m:
            if P[i, c].is_zero():
                i += 1
                continue

            # zero entry P[i,c]
            P, U = col_gcd(P, U, r, i, pivot_col)

            # Now P[i,c] should be zero
            i += 1

        # Normalize pivot to be monic: scale row r by 1/LC(pivot) (constant)
        pivot = P[r, c].element
        if pivot.is_zero:
            # Should not happen here, but guard anyway
            continue
        lc = pivot.LC  # constant in the base field
        if lc != 1:
            inv_lc = domain.convert(1) / lc
            P = col_scale_constant(P, r, inv_lc)
            U = col_scale_constant(U, r, inv_lc)

        # Reduce entries left the pivot in row c: make them zero using division
        pivot = P[r, c].element  # monic now
        for i in range(r):
            if P[r, i].is_zero():
                continue
            q, rem = P[r, i].element.div(pivot)  # P[r,i] = q*pivot + rem
            # col_i <- col_i - q * col_c, so entry becomes 'rem'
            if not q.is_zero:
                # Apply to P
                for j in range(m):
                    P[j, i] = P[j, i] - q * P[r, c]
                # Apply to U
                for j in range(n):
                    U[j, i] = U[j, i] - q * U[r, c]
        r += 1

    # return sympy matrices
    H = P.to_Matrix()
    Umat = U.to_Matrix()
    return H, Umat


def echelon_form(A, x, domain=None, mode: str = "pre"):
    """
    Compute the row echelon form of a matrix

    This method will yield an upper right triangular form either using
    pre oder post multiplication with a unimodular matrix.
    """
    M = A.copy()
    n, m = M.shape
    if domain is None:
        if M.free_symbols:
            domain = EX
        else:
            domain = QQ
    R = domain.poly_ring(x)

    # Convert input to DomainMatrix over R
    if isinstance(M, DomainMatrix):
        P = M if M.domain == R else M.convert_to(R)
    else:
        P = DomainMatrix.from_list(M.tolist(), domain=R).to_sparse()

    # Store transformation steps
    U = DomainMatrix.eye(n, R)

    if mode == "pre":
        # Eliminate row entries by pre-multiplication (row operations)
        r = 0
        for c in range(m):
            for i in range(r + 1, n):
                P, U = row_gcd(P, U, r, i, c)
            if not P[r, c].is_zero():
                r += 1
        if 0:
            # satisfy requirement (r2), TBD
            r = 1
            for k in range(m):
                pivot = P[r, k].element
                if not pivot.is_zero:
                    # Make pivots monic
                    scale = 1 / pivot.LC
                    P, U = row_scale_constant(P, U, r, scale)
                    # Scale row with pivot
                    for i in range(r - 1):
                        q = R.quo(P[i, k], P[r, k]).element
                        P, U = row_add(P, U, r, i, -q)
                    r += 1
    elif mode == "post":
        # Eliminate row entries by post-multiplication (col operations)
        c = m - 1
        for r in range(n - 1, -1, -1):
            for i in range(c - 1, -1, -1):
                P, U = col_gcd(P, U, c, i, r)
            if not P[r, c].is_zero():
                c -= 1

    else:
        raise ValueError(f"Mode '{mode}' not supported.")

    P_mat = P.to_Matrix()
    U_mat = U.to_Matrix()
    return P_mat, U_mat


def col_gcd(
    P: DomainMatrix, U: DomainMatrix, c: int, i: int, r: int
) -> tuple[DomainMatrix, DomainMatrix]:
    """
    Use the entry P[r,c] to zero the element P[r,i] using an unimodular right transform

    This will also set P[r,c] to the GCD of P[r,c] and P[r,i].
    """
    R = P.domain
    # pivot element
    a = P[r, c].element
    # element to zero
    b = P[r, i].element
    # Extended GCD: s*a + t*b = g
    s, t, g = R.gcdex(a, b)
    # Normalize g to be monic
    lc_g = g.LC
    s = R.quo(s, lc_g)
    t = R.quo(t, lc_g)
    g = R.quo(g, lc_g)

    # Compute a/g and b/g (exact division in the polynomial ring)
    v = a.quo(g)
    u = -b.quo(g)

    # 2x2 unimodular block T = [[s, u], [t, v]]
    T = DomainMatrix.eye(P.shape[1], R)
    T[r, r] = s
    T[r, i] = u
    T[i, r] = t
    T[i, i] = v
    P = P.matmul(T)
    U = U.matmul(T)
    return P, U


def row_gcd(
    P: DomainMatrix, U: DomainMatrix, r: int, i: int, c: int
) -> tuple[DomainMatrix, DomainMatrix]:
    """
    Use the entry P[r,c] to zero the element P[i,c] using an unimodular left transform

    This will also set P[r,c] to the GCD of P[r,c] and P[i,c].
    """
    R = P.domain
    # pivot element
    a = P[r, c].element
    # element to zero
    b = P[i, c].element
    # Extended GCD: s*a + t*b = g
    s, t, g = R.gcdex(a, b)
    # Normalize g to be monic
    # lc(g) is a nonzero constant in the field
    lc_g = g.LC
    s = R.quo(s, lc_g)
    t = R.quo(t, lc_g)
    g = R.quo(g, lc_g)

    # Compute a/g and b/g (exact division in the polynomial ring)
    v = a.quo(g)
    u = -b.quo(g)

    # 2x2 unimodular block T = [[s, t], [v, u]]
    T = DomainMatrix.eye(P.shape[0], R)
    T[r, r] = s
    T[r, i] = t
    T[i, r] = u
    T[i, i] = v
    P = T.matmul(P)
    U = T.matmul(U)
    return P, U


def row_swap(M: DomainMatrix, i: int, j: int):
    T = DomainMatrix.eye(M.shape[0], M.domain)
    T[i, i] = 0
    T[j, j] = 0
    T[i, j] = 1
    T[j, i] = 1
    return T.matmul(M)


def col_swap(M: DomainMatrix, i: int, j: int):
    T = DomainMatrix.eye(M.shape[1], M.domain)
    T[i, i] = 0
    T[j, j] = 0
    T[i, j] = 1
    T[j, i] = 1
    return M.matmul(T)


def row_add(M: DomainMatrix, U: DomainMatrix, r: int, i: int, c: Basic):
    """
    Add row r to row i after scaling it with c
    """
    T = DomainMatrix.eye(M.shape[0], M.domain)
    T[i, r] = M.domain.convert(c)
    return T.matmul(M), T.matmul(U)


def row_scale_constant(M: DomainMatrix, U: DomainMatrix, i: int, c: Basic):
    # scale by nonzero constant in the field
    if c == 0:
        raise ValueError("Scaling by zero is not allowed.")

    T = DomainMatrix.eye(M.shape[0], M.domain)
    T[i, i] = M.domain.convert(c)
    return T.matmul(M), T.matmul(U)


def col_scale_constant(M: DomainMatrix, i: int, c: Basic):
    # scale by nonzero constant in the field
    if c == 0:
        raise ValueError("Scaling by zero is not allowed.")

    T = DomainMatrix.eye(M.shape[1], M.domain)
    T[i, i] = M.domain.convert(c)
    return M.matmul(T)


def is_in_echelon_form(mat: Matrix):
    m, n = mat.shape
    for r in range(m):
        for c in range(r):
            if not mat[r, c].is_zero:
                return False
    return True


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


def column_reduced_form(
    mat: sp.Matrix, sym: sp.Symbol, debug: bool = False
) -> tuple[sp.Matrix, sp.Matrix]:
    """
    Compute the column reduced form S of a matrix mat by using
    an unimodular right transformation :math:`M U = S`.

    Returns: Tuple(S, U)
    """
    u = sp.eye(mat.cols)
    m = mat.copy()
    while not is_col_reduced(m, sym):
        # sort into decreasing column degrees
        cds = get_leading_column_degrees(m, sym)
        if sorted(cds, reverse=True) != cds:
            s_idxs = np.argsort(cds).tolist()[::-1]
            perm = sp.combinatorics.Permutation(s_idxs)
            p = sp.PermutationMatrix(perm).as_explicit().T
            m = m @ p
            u = u @ p

        # select pivot column index k
        mh = get_lccm(m, sym)
        cds = get_leading_column_degrees(m, sym)
        cfs = None
        for k in range(m.cols):
            # check if mh[k] can be expressed as linear combination of columns mh[k+1:]
            for q in range(1, m.cols-k):
                # create some coefficients
                cs = sp.symbols(f"c_{k + 1}:{k + 1 + q}")
                # pad them for the first k columns and the remaining ones
                csp = (tuple(0 for n in range(k + 1))
                       + cs
                       + tuple(0 for n in range(k +1 +q, m.cols)))
                expr = mh[:, k] - sp.Add(*[c * mh[:, idx] for idx, c in enumerate(csp)])
                # compute basis coefficients
                sol = sp.solve(expr, cs)
                if sol:
                    sol[0] = 0
                    cfs = [sol[c] for c in csp]
                    break
            if cfs is not None:
                break

        # eliminate kth column from mh
        r = sp.Matrix.eye(m.cols)
        for i in range(r.rows):
            if i == k:
                continue
            r[i, k] = -cfs[i] * sym ** (cds[k] - cds[i])
        m = m @ r
        m.simplify()

        # remember the operations
        u = u @ r

    return m, u
