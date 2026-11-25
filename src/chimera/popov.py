from __future__ import annotations

from typing import Optional

from sympy.polys.matrices import DomainMatrix


def _require_univariate_polynomial_ring(A: DomainMatrix):
    """
    Verify that A is a DomainMatrix over a univariate polynomial ring K[x],
    with K a field.

    Supported examples:

        R = QQ.poly_ring(z)
        A = DomainMatrix.from_list([...], R)

    or:

        R = GF(7).poly_ring(z)
        A = DomainMatrix.from_list([...], R)

    Returns
    -------
    R:
        The polynomial ring K[x].
    K:
        The coefficient field K.
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
            "Use, for example, QQ.poly_ring(z), not ZZ.poly_ring(z)."
        )

    return R, K


def _domain_matrix_from_rows(rows, ncols, R):
    """
    Construct a DomainMatrix from dense rows.

    The explicit ncols argument is required when rows is empty.
    """
    if not rows:
        return DomainMatrix.zeros((0, ncols), R)

    return DomainMatrix.from_list(rows, R)


def _identity_rows(n, R):
    """
    Return an n x n identity matrix as a dense list of rows over R.
    """
    return [
        [R.one if i == j else R.zero for j in range(n)]
        for i in range(n)
    ]


def _row_degree(rows, i):
    r"""
    Return the degree of row i.

    The polynomial ring uses the natural convention

    \[
        \deg(0) = -\infty.
    \]

    Hence a zero row has row degree \(-\infty\).
    """
    return max(f.degree() for f in rows[i])


def _is_zero_row(row):
    """
    Return True if all entries in row are zero.
    """
    return all(not f for f in row)


def _pivot_column(rows, i) -> Optional[int]:
    r"""
    Return the pivot column of row i.

    The convention follows the supplied paper:

    * The pivot is the rightmost entry having the maximum row degree.
    * A zero row has no pivot and returns None.

    If row i has degree \(\nu_i\), then its pivot column \(q_i\) is the
    rightmost index satisfying

    \[
        \deg(A_{i,q_i}) = \nu_i.
    \]
    """
    degree = _row_degree(rows, i)

    # A zero row has degree -oo. It must not receive a pivot.
    if degree < 0:
        return None

    for j in range(len(rows[i]) - 1, -1, -1):
        if rows[i][j].degree() == degree:
            return j

    raise RuntimeError("Internal error: nonzero row has no pivot.")


def _remove_zero_rows(Arows, Urows):
    """
    Remove zero rows from Arows and remove the matching rows from Urows.

    The relation

        Arows = Urows * A_input

    is preserved.
    """
    keep = [i for i, row in enumerate(Arows) if not _is_zero_row(row)]

    return (
        [Arows[i] for i in keep],
        [Urows[i] for i in keep],
    )


def _row_subtract_multiple(Arows, Urows, target, source, q):
    r"""
    Perform the polynomial row operation

    \[
        \operatorname{row}_{target}
        \leftarrow
        \operatorname{row}_{target}
        -
        q \operatorname{row}_{source}.
    \]

    The same operation is made on Urows, preserving

    \[
        A_{\mathrm{current}} = U_{\mathrm{current}} A_{\mathrm{input}}.
    \]
    """
    Arows[target] = [
        a - q * b
        for a, b in zip(Arows[target], Arows[source])
    ]

    Urows[target] = [
        a - q * b
        for a, b in zip(Urows[target], Urows[source])
    ]


def _scale_row(Arows, Urows, i, scalar):
    r"""
    Apply

    \[
        \operatorname{row}_i
        \leftarrow
        scalar \cdot \operatorname{row}_i
    \]

    to both the polynomial matrix and the transformation matrix.
    """
    Arows[i] = [scalar * entry for entry in Arows[i]]
    Urows[i] = [scalar * entry for entry in Urows[i]]


def _make_pivots_monic(Arows, Urows, R, K):
    """
    Scale each nonzero row so that its pivot entry is monic.

    If the pivot polynomial is p(x) with leading coefficient lc(p), use

        row <- lc(p)^(-1) * row.
    """
    for i in range(len(Arows)):
        q = _pivot_column(Arows, i)

        if q is None:
            continue

        pivot = Arows[i][q]
        leading_coefficient = pivot.LC

        inverse_lc = K.exquo(K.one, leading_coefficient)
        scalar = K(inverse_lc)

        _scale_row(Arows, Urows, i, scalar)


def _weak_popov_reduce(Arows, Urows, R, K):
    r"""
    Compute weak row Popov form.

    A polynomial matrix is in weak row Popov form when all nonzero rows
    have distinct pivot columns.

    Suppose rows s and t have the same pivot column q. Let their row
    degrees be d_s and d_t, with d_s <= d_t. Then apply

    \[
        \operatorname{row}_t
        \leftarrow
        \operatorname{row}_t
        -
        \frac{\operatorname{lc}(A_{tq})}
             {\operatorname{lc}(A_{sq})}
        x^{d_t-d_s}
        \operatorname{row}_s.
    \]

    This cancels the leading term in column q of row t.

    The process terminates because every reduction lowers the leading
    monomial of one row.
    """
    Arows, Urows = _remove_zero_rows(Arows, Urows)

    while True:
        pivots = [_pivot_column(Arows, i) for i in range(len(Arows))]
        reduction_done = False

        ncols = len(Arows[0]) if Arows else 0

        for q in range(ncols):
            rows_with_same_pivot = [
                i for i, pivot in enumerate(pivots) if pivot == q
            ]

            if len(rows_with_same_pivot) <= 1:
                continue

            # Use lowest-degree row as reducer.
            # The row number is a deterministic tie breaker.
            rows_with_same_pivot.sort(
                key=lambda i: (_row_degree(Arows, i), i)
            )

            source = rows_with_same_pivot[0]
            target = rows_with_same_pivot[-1]

            source_degree = _row_degree(Arows, source)
            target_degree = _row_degree(Arows, target)

            source_lc = Arows[source][q].LC
            target_lc = Arows[target][q].LC

            coefficient = K.exquo(target_lc, source_lc)
            degree_shift = target_degree - source_degree
            multiplier = R({(degree_shift,): coefficient})

            _row_subtract_multiple(
                Arows=Arows,
                Urows=Urows,
                target=target,
                source=source,
                q=multiplier,
            )

            Arows, Urows = _remove_zero_rows(Arows, Urows)

            reduction_done = True
            break

        if not reduction_done:
            break

    _make_pivots_monic(Arows, Urows, R, K)

    return Arows, Urows


def _canonicalize_weak_popov(Arows, Urows):
    r"""
    Convert a weak row Popov form into canonical row Popov form.

    Preconditions
    -------------
    Arows is already in weak row Popov form:

    * every nonzero row has a pivot;
    * pivot columns are distinct;
    * pivots are monic.

    For each pivot row j, with pivot column q_j and pivot polynomial

    \[
        p_j = A_{j,q_j},
    \]

    reduce every other entry in its pivot column modulo p_j.  Thus, for
    each i != j, enforce

    \[
        \deg(A_{i,q_j}) < \deg(p_j).
    \]

    If

    \[
        A_{i,q_j} = Q(z) p_j(z) + R(z),
        \qquad
        \deg R(z) < \deg p_j(z),
    \]

    then apply the row operation

    \[
        \operatorname{row}_i
        \leftarrow
        \operatorname{row}_i
        -
        Q(z)\operatorname{row}_j.
    \]

    This changes only the selected target row i, but it changes all
    columns of that row. Therefore, after reducing with respect to one
    pivot column, the target row may need a later reduction with respect
    to another pivot column.

    No weak-Popov reduction is performed here: this routine assumes that
    the weak Popov pivots remain valid under these degree-reducing row
    operations.
    """
    while True:
        pivots = [
            _pivot_column(Arows, i)
            for i in range(len(Arows))
        ]

        row_degrees = [
            _row_degree(Arows, i)
            for i in range(len(Arows))
        ]

        reduction_done = False

        for source in range(len(Arows)):
            pivot_column = pivots[source]
            pivot_polynomial = Arows[source][pivot_column]
            pivot_degree = row_degrees[source]

            for target in range(len(Arows)):
                if target == source:
                    continue

                entry = Arows[target][pivot_column]

                # The zero polynomial has degree -oo, so it already
                # satisfies deg(0) < pivot_degree.
                if not entry or entry.degree() < pivot_degree:
                    continue

                quotient, _remainder = divmod(
                    entry,
                    pivot_polynomial,
                )

                # Only row `target` changes. Since this is a row
                # operation, every entry in that one row is updated.
                _row_subtract_multiple(
                    Arows=Arows,
                    Urows=Urows,
                    target=target,
                    source=source,
                    q=quotient,
                )

                reduction_done = True
                break

            if reduction_done:
                break

        # Every non-pivot entry in every pivot column is reduced.
        if not reduction_done:
            break

    return Arows, Urows


def row_popov_form(
    A: DomainMatrix,
    *,
    canonical: bool = True,
    return_pivots: bool = False,
):
    r"""
    Compute the weak or canonical row Popov form of A.

    Parameters
    ----------
    A : DomainMatrix
        A matrix over a univariate polynomial ring K[x], where K is a
        field; for example, a DomainMatrix over QQ[z] or GF(p)[z].

    canonical : bool, default=True
        If False, return weak row Popov form.

        If True, first compute weak row Popov form and then canonicalize
        it by reducing non-pivot entries in pivot columns.

    return_pivots : bool, default=False
        If True, also return the zero-based pivot-column indices.

    Returns
    -------
    P, U
        If `return_pivots=False`.

    P, U, pivots
        If `return_pivots=True`.

    The result satisfies

    \[
        P = U A.
    \]

    The pivot convention is that of the paper: the pivot of a row is its
    rightmost entry of maximal degree.
    """
    R, K = _require_univariate_polynomial_ring(A)

    m, n = A.shape

    # Convert once. Entries remain elements of R = K[x].
    Arows = A.to_list()

    # Initially, Arows = Urows * A.
    Urows = _identity_rows(m, R)

    Arows, Urows = _remove_zero_rows(Arows, Urows)

    # The zero matrix has no nonzero Popov rows.
    if not Arows:
        P = DomainMatrix.zeros((0, n), R)
        U = DomainMatrix.zeros((0, m), R)

        if return_pivots:
            return P, U, []

        return P, U

    # Every call computes weak Popov form first.
    Arows, Urows = _weak_popov_reduce(Arows, Urows, R, K)

    # Canonical form is weak Popov form plus pivot-column reductions.
    if canonical:
        Arows, Urows = _canonicalize_weak_popov(Arows, Urows)

    # Sort rows as required in the paper: increasing row degree.
    # The pivot column is merely a deterministic tie-breaker.
    order = sorted(
        range(len(Arows)),
        key=lambda i: (
            _row_degree(Arows, i),
            _pivot_column(Arows, i),
        ),
    )

    Arows = [Arows[i] for i in order]
    Urows = [Urows[i] for i in order]

    pivots = [
        _pivot_column(Arows, i)
        for i in range(len(Arows))
    ]

    P = _domain_matrix_from_rows(Arows, n, R)
    U = _domain_matrix_from_rows(Urows, m, R)

    if return_pivots:
        return P, U, pivots

    return P, U


def col_popov_form(
    A: DomainMatrix,
    *,
    canonical: bool = True,
    return_pivots: bool = False,
):
    ret = row_popov_form(
        A=A.transpose(),
        canonical=canonical,
        return_pivots=return_pivots,
    )
    P = ret[0].transpose()
    U = ret[1].transpose()

    if return_pivots:
        return P, U, ret[2]

    return P, U



def is_row_popov_form(
    P: DomainMatrix,
    *,
    canonical: bool = True,
) -> bool:
    r"""
    Test whether P is in weak or canonical row Popov form.

    The matrix P must be over a univariate polynomial ring K[x], where K
    is a field.

    Parameters
    ----------
    P : DomainMatrix
        Polynomial matrix over K[x].

    canonical : bool, default=True
        If False, test only weak row Popov form.

        If True, test canonical row Popov form.

    Returns
    -------
    bool
        True if P satisfies the requested Popov-form conditions.

    Weak row Popov conditions
    -------------------------
    Let nu_i be the degree of row i, and let q_i be the rightmost column
    attaining degree nu_i in that row.

    The following conditions are checked:

    1. P has no zero rows.

    2. The row degrees are nondecreasing:

       \[
           \nu_1 \leq \nu_2 \leq \cdots.
       \]

    3. Pivot columns are distinct:

       \[
           q_i \ne q_j \quad \text{for } i \ne j.
       \]

    4. Every pivot polynomial P[i, q_i] is monic.

    Canonical row Popov condition
    -----------------------------
    In addition to weak Popov form, for every pivot column q_j:

    \[
        \deg(P_{i,q_j}) < \nu_j
        \qquad \text{for every } i \ne j.
    """
    _require_univariate_polynomial_ring(P)

    rows = P.to_list()

    # The empty matrix has no rows violating the conditions.
    if not rows:
        return True

    # Popov form should contain no zero rows.
    if any(_is_zero_row(row) for row in rows):
        return False

    row_degrees = [
        _row_degree(rows, i)
        for i in range(len(rows))
    ]

    pivots = [
        _pivot_column(rows, i)
        for i in range(len(rows))
    ]

    # A nonzero row must have a pivot.
    if any(q is None for q in pivots):
        return False

    # The convention in the paper orders row degrees increasingly.
    if row_degrees != sorted(row_degrees):
        return False

    # Weak Popov condition: all pivot columns must be distinct.
    if len(set(pivots)) != len(pivots):
        return False

    # Every pivot polynomial must be monic.
    for i, q in enumerate(pivots):
        if rows[i][q].LC != P.domain.one:
            return False

    # Weak Popov form requires no further tests.
    if not canonical:
        return True

    # Canonical Popov condition:
    #
    #     deg(P[i, q_j]) < deg(P[j, q_j])
    #
    # for every i != j.
    #
    # The Domain polynomial representation uses deg(0) = -oo. We can
    # therefore omit zero entries directly: -oo < any pivot degree.
    for source, pivot_column in enumerate(pivots):
        pivot_degree = row_degrees[source]

        for target in range(len(rows)):
            if target == source:
                continue

            entry = rows[target][pivot_column]

            if entry and entry.degree() >= pivot_degree:
                return False

    return True


def is_col_popov_form(
    P: DomainMatrix,
    *,
    canonical: bool = True,
) -> bool:
    return is_row_popov_form(P.transpose(), canonical=canonical)
