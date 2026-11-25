from __future__ import annotations

from sympy.polys.matrices import DomainMatrix


# ---------------------------------------------------------------------------
# Shared DomainMatrix / polynomial-ring helpers
# ---------------------------------------------------------------------------

def _require_univariate_polynomial_ring(A: DomainMatrix):
    """
    Verify that A is over a univariate polynomial ring K[x], where K is
    a field.

    Examples
    --------
    >>> R = QQ.poly_ring(z)
    >>> A = DomainMatrix.from_list([...], R)

    or:

    >>> R = GF(7).poly_ring(z)
    >>> A = DomainMatrix.from_list([...], R)
    """
    R = A.domain

    if not hasattr(R, "ngens") or R.ngens != 1:
        raise ValueError(
            "A must be over a univariate polynomial ring K[x]. "
            "For example: R = QQ.poly_ring(z)."
        )

    if not hasattr(R, "domain"):
        raise ValueError(
            "A.domain must be a polynomial ring such as QQ[z]."
        )

    K = R.domain

    if not K.is_Field:
        raise ValueError(
            "The coefficient domain must be a field. "
            "For example, use QQ.poly_ring(z), not ZZ.poly_ring(z)."
        )

    return R, K


def _domain_matrix_from_rows(rows, ncols, R):
    """
    Construct a DomainMatrix from a dense row list.

    The explicit column count is needed when rows is empty.
    """
    if not rows:
        return DomainMatrix.zeros((0, ncols), R)

    return DomainMatrix.from_list(rows, R)


def _identity_rows(n, R):
    """
    Return the n x n identity matrix as a dense list of rows over R.
    """
    return [
        [R.one if i == j else R.zero for j in range(n)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Column operations
# ---------------------------------------------------------------------------

def _swap_columns(Arows, Vrows, i, j):
    """
    Swap columns i and j in both Arows and Vrows.

    Since H = A*V, every column operation performed on H must also be
    performed on V.
    """
    if i == j:
        return

    for row in Arows:
        row[i], row[j] = row[j], row[i]

    for row in Vrows:
        row[i], row[j] = row[j], row[i]


def _scale_column(Arows, Vrows, j, scalar):
    r"""
    Apply the column operation

    \[
        C_j \leftarrow scalar \cdot C_j
    \]

    to both the current matrix and the transformation matrix.
    """
    for row in Arows:
        row[j] *= scalar

    for row in Vrows:
        row[j] *= scalar


def _column_subtract_multiple(Arows, Vrows, target, source, q):
    r"""
    Perform the column operation

    \[
        C_{\mathrm{target}}
        \leftarrow
        C_{\mathrm{target}}
        -
        q C_{\mathrm{source}}.
    \]

    The same operation is applied to V, preserving

    \[
        H = A V.
    \]
    """
    for row in Arows:
        row[target] -= q * row[source]

    for row in Vrows:
        row[target] -= q * row[source]


def _column_gcd_reduce(Arows, Vrows, pivot_col, other_col, row, R):
    r"""
    Replace two columns so that their entries in a specified row become

    \[
        (a, b) \longmapsto (\gcd(a,b), 0).
    \]

    Let

    \[
        a = A_{row,pivot\_col},
        \qquad
        b = A_{row,other\_col}.
    \]

    If

    \[
        s a + t b = g,
        \qquad
        g = \gcd(a,b),
    \]

    and

    \[
        a = g a',
        \qquad
        b = g b',
    \]

    then the unimodular column transformation

    \[
    \begin{bmatrix}
        C_{\mathrm{pivot}}' &
        C_{\mathrm{other}}'
    \end{bmatrix}
    =
    \begin{bmatrix}
        C_{\mathrm{pivot}} &
        C_{\mathrm{other}}
    \end{bmatrix}
    \begin{bmatrix}
        s & -b' \\
        t & a'
    \end{bmatrix}
    \]

    gives

    \[
        A_{row,\mathrm{pivot}}' = g,
        \qquad
        A_{row,\mathrm{other}}' = 0.
    \]
    """
    a = Arows[row][pivot_col]
    b = Arows[row][other_col]

    if not b:
        return

    # In the Euclidean domain R = K[x]:
    #
    #     s*a + t*b = g
    #
    s, t, g = R.gcdex(a, b)

    a_quotient = R.exquo(a, g)
    b_quotient = R.exquo(b, g)

    # Save the old columns because both new columns depend on both old ones.
    old_A_pivot = [Arows[i][pivot_col] for i in range(len(Arows))]
    old_A_other = [Arows[i][other_col] for i in range(len(Arows))]

    old_V_pivot = [Vrows[i][pivot_col] for i in range(len(Vrows))]
    old_V_other = [Vrows[i][other_col] for i in range(len(Vrows))]

    # New pivot column:
    #
    #     C_pivot <- s*C_pivot + t*C_other
    #
    # New other column:
    #
    #     C_other <- -b' * C_pivot + a' * C_other
    #
    for i in range(len(Arows)):
        Arows[i][pivot_col] = (
            s * old_A_pivot[i] + t * old_A_other[i]
        )

        Arows[i][other_col] = (
            -b_quotient * old_A_pivot[i]
            + a_quotient * old_A_other[i]
        )

    for i in range(len(Vrows)):
        Vrows[i][pivot_col] = (
            s * old_V_pivot[i] + t * old_V_other[i]
        )

        Vrows[i][other_col] = (
            -b_quotient * old_V_pivot[i]
            + a_quotient * old_V_other[i]
        )


# ---------------------------------------------------------------------------
# Column Hermite form
# ---------------------------------------------------------------------------

def column_hermite_form(A: DomainMatrix):
    r"""
    Compute the column Hermite form of a full-row-rank polynomial matrix.

    The input must be a matrix over a univariate polynomial ring \(K[x]\),
    where \(K\) is a field.

    Parameters
    ----------
    A : DomainMatrix
        An \(m \times n\) polynomial matrix over \(K[x]\), with \(m \le n\)
        and full row rank \(m\).

    Returns
    -------
    H, V : tuple[DomainMatrix, DomainMatrix]
        `H` is the column Hermite form of `A`, and `V` is a unimodular
        polynomial matrix satisfying

        \[
            H = A V.
        \]

    Form of H
    ---------
    The result H has the structure

    \[
        H =
        \begin{bmatrix}
            h_{00} & 0      & \cdots & 0      & 0 & \cdots & 0 \\
            h_{10} & h_{11} & \cdots & 0      & 0 & \cdots & 0 \\
            \vdots & \vdots & \ddots & \vdots & \vdots & & \vdots \\
            h_{m-1,0} & h_{m-1,1} & \cdots & h_{m-1,m-1}
                & 0 & \cdots & 0
        \end{bmatrix}.
    \]

    Its leading m x m block is lower triangular, all diagonal entries are
    monic, and the entries below each diagonal element satisfy

    \[
        \deg(h_{ij}) < \deg(h_{ii}),
        \qquad 0 \le j < i < m.
    \]

    Preconditions
    -------------
    A must have full row rank. Equivalently, during elimination every row
    k must have a nonzero entry among columns k through n-1.

    If the matrix is rank deficient, a ValueError is raised.

    Notes
    -----
    The algorithm performs only unimodular polynomial column operations:

    * column swaps;
    * multiplication of a column by a nonzero element of K;
    * replacing a column by itself minus a polynomial multiple of another;
    * extended-gcd based two-column unimodular transformations.

    Therefore the transformation matrix V is unimodular.
    """
    R, K = _require_univariate_polynomial_ring(A)

    m, n = A.shape

    if m > n:
        raise ValueError(
            "A column Hermite form with an m x m lower-triangular pivot "
            "block requires m <= n."
        )

    Hrows = A.to_list()
    Vrows = _identity_rows(n, R)

    # -----------------------------------------------------------------------
    # Stage 1:
    #
    # Produce a lower-triangular matrix with monic diagonal.
    #
    # At stage k, use extended gcd column operations among columns k,...,n-1
    # to make H[k,k] the gcd of row-k entries and make all other entries
    # in row k equal to zero.
    # -----------------------------------------------------------------------
    for k in range(m):
        pivot_col = None

        for j in range(k, n):
            if Hrows[k][j]:
                pivot_col = j
                break

        if pivot_col is None:
            raise ValueError(
                "A must have full row rank for this column Hermite form."
            )

        # Move a nonzero entry into the diagonal position.
        _swap_columns(Hrows, Vrows, k, pivot_col)

        # Replace H[k,k] and H[k,j] by gcd(H[k,k], H[k,j]) and zero.
        for j in range(k + 1, n):
            if Hrows[k][j]:
                _column_gcd_reduce(
                    Arows=Hrows,
                    Vrows=Vrows,
                    pivot_col=k,
                    other_col=j,
                    row=k,
                    R=R,
                )

        # Normalize the diagonal polynomial to monic form.
        pivot = Hrows[k][k]
        leading_coefficient = pivot.LC

        inverse_lc = K.exquo(K.one, leading_coefficient)
        scale = K(inverse_lc)

        _scale_column(Hrows, Vrows, k, scale)

    # -----------------------------------------------------------------------
    # Stage 2:
    #
    # Reduce entries below the diagonal.
    #
    # For i > j, divide H[i,j] by H[i,i]:
    #
    #     H[i,j] = q H[i,i] + r,
    #
    # and then perform:
    #
    #     C_j <- C_j - q C_i.
    #
    # Since H[i,i] is in column i and columns i are zero above row i, this
    # leaves rows 0,...,i-1 unchanged and makes:
    #
    #     H[i,j] = r,
    #
    # with deg(r) < deg(H[i,i]).
    #
    # Process rows from top to bottom. A reduction at row i may alter entries
    # in lower rows, but those lower rows have not yet been reduced.
    # -----------------------------------------------------------------------
    for i in range(1, m):
        diagonal = Hrows[i][i]

        for j in range(i):
            entry = Hrows[i][j]

            if not entry:
                continue

            quotient, _remainder = divmod(entry, diagonal)

            if quotient:
                _column_subtract_multiple(
                    Arows=Hrows,
                    Vrows=Vrows,
                    target=j,
                    source=i,
                    q=quotient,
                )

    H = _domain_matrix_from_rows(Hrows, n, R)
    V = _domain_matrix_from_rows(Vrows, n, R)

    return H, V


def is_column_hermite_form(H: DomainMatrix) -> bool:
    r"""
    Test whether H is in the column Hermite form used by
    `column_hermite_form`.

    The required structure for an m x n matrix, m <= n, is:

    1. The leading m x m block is lower triangular:

       \[
           H_{ij} = 0 \quad \text{for } i < j.
       \]

    2. Every column after the first m columns is zero.

    3. Every diagonal entry H[i,i] is nonzero and monic.

    4. Below-diagonal entries are reduced:

       \[
           \deg(H_{ij}) < \deg(H_{ii}),
           \qquad j < i.
       \]
    """
    _require_univariate_polynomial_ring(H)

    R = H.domain
    rows = H.to_list()

    m, n = H.shape

    if m > n:
        return False

    # Diagonal entries must be nonzero and monic.
    for i in range(m):
        diagonal = rows[i][i]

        if not diagonal:
            return False

        if diagonal.LC != R.domain.one:
            return False

    # The leading m x m block must be lower triangular.
    for i in range(m):
        for j in range(i + 1, m):
            if rows[i][j]:
                return False

    # All columns after the leading m x m block must be zero.
    for i in range(m):
        for j in range(m, n):
            if rows[i][j]:
                return False

    # Entries below the diagonal must be reduced modulo the diagonal entry
    # in their row.
    for i in range(1, m):
        diagonal_degree = rows[i][i].degree()

        for j in range(i):
            entry = rows[i][j]

            # degree(0) = -oo, hence zero automatically satisfies the
            # reduction condition.
            if entry and entry.degree() >= diagonal_degree:
                return False

    return True
