from sympy import Matrix
from sympy.polys.polytools import Poly
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix


def row_hermite_form(A, x, domain=QQ):
    """
    Compute the row Hermite normal form H of a matrix A with entries in F[x],
    where F is a field (default: QQ), and a unimodular U over F[x] such that:
        H = A * U

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
    R = domain.poly_ring(x)

    # Convert input to DomainMatrix over R
    if isinstance(A, DomainMatrix):
        P = A if A.domain == R else A.convert_to(R)
    else:
        P = DomainMatrix.from_Matrix(Matrix(A)).convert_to(R)

    U = DomainMatrix.eye(n, R)

    def col_swap(M, i, j):
        T = DomainMatrix.eye(n, R)
        T[i, i] = 0
        T[i, j] = 1
        T[j, i] = 1
        return M.matmul(T)

    def row_swap(M, i, j):
        T = DomainMatrix.eye(n, R)
        T[i, i] = 0
        T[i, j] = 1
        T[j, i] = 1
        return T.matmul(M)

    def col_scale_constant(M, i, c):  # scale by nonzero constant in the field
        if c == 0:
            raise ValueError("Scaling by zero is not allowed.")

        T = DomainMatrix.eye(n, R)
        T[i, i] = R.convert(c)
        return M.matmul(T)

    def row_scale_constant(M, i, c):  # scale by nonzero constant in the field
        if c == 0:
            raise ValueError("Scaling by zero is not allowed.")

        T = DomainMatrix.eye(n, R)
        T[i, i] = R.convert(c)
        return T.matmul(M)

    r = 0  # current pivot row
    for c in range(n):
        if r >= m:
            break
        # Find a row with nonzero entry in column c at or below r
        pivot_row = None
        for i in range(r, m):
            if not P[i, c].is_zero():
                pivot_row = i
                break
        if pivot_row is None:
            continue

        # Swap to bring pivot candidate to row r
        if pivot_row != r:
            P = row_swap(P, r, pivot_row)
            U = row_swap(U, r, pivot_row)

        # Clear entries below in column c using 2x2 unimodular transformations
        i = r + 1
        while i < m:
            if P[i, c].is_zero():
                i += 1
                continue

            a = P[r, c].element
            b = P[i, c].element
            # Extended GCD: u*a + v*b = g
            u, v, g = R.gcdex(a, b)
            # Normalize g to be monic
            # lc(g) is a nonzero constant in the field
            lc_g = g.LC
            u = R.quo(u, lc_g)
            v = R.quo(v, lc_g)
            g = R.quo(g, lc_g)

            # Compute a/g and b/g (exact division in the polynomial ring)
            ag = a.quo(g)
            bg = b.quo(g)

            # 2x2 unimodular block M = [[u, v], [ -b/g, a/g ]]
            # Apply to the pair of rows (r, i) on both P and U
            T = DomainMatrix.eye(n, P.domain)
            T[r, r] = u
            T[r, i] = v
            T[i, r] = -bg
            T[i, i] = ag
            # P = T.matmul(P)
            # U = T.matmul(U)
            P = P.matmul(P)
            U = U.matmul(U)

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
        pivot = P[r,c].element  # monic now
        for i in range(r):
            if P[r,i].is_zero():
                continue
            q, rem = P[r,i].element.div(pivot)  # P[r,i] = q*pivot + rem
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
