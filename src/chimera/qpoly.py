from collections import defaultdict
from functools import reduce
from typing import Union

import sympy as sp
from sympy.core.exprtools import decompose_power_rat
from sympy.core.numbers import S

__all__ = ["QPoly"]


class QPoly(sp.Basic):
    """
    Class to handle signomials, which are polynomials but with rational exponents
    """

    is_commutative = True
    is_QPoly = True

    def __new__(cls, rep, *gens):
        """Create a new generalised polynomial instance out of something useful."""
        if isinstance(rep, QPoly):
            return cls.from_dict(rep.as_dict(), *gens)
        if isinstance(rep, dict):
            return cls.from_dict(rep, *gens)
        elif isinstance(rep, list):
            return cls.from_list(rep, *gens)
        elif isinstance(rep, sp.Add):
            # gp = next((o for o in rep.args if isinstance(rep, GenPoly)), None)
            # if gp is not None:
            if any(isinstance(arg, QPoly) for arg in rep.args):
                return cls(reduce(lambda f, g: f + g, rep.args), *gens)

        # last resort
        rep = sp.sympify(rep, evaluate=type(rep) is not str)
        return cls.from_expr(rep, *gens)

    @classmethod
    def from_qpoly(cls, expr, *gens: sp.Symbol):
        return cls.from_dict(expr.as_dict(), *gens)

    @classmethod
    def from_expr(cls, expr: sp.Expr, *gens: sp.Symbol):
        return cls.from_dict(cls.dict_from_expr(expr, *gens), *gens)

    @classmethod
    def from_list(cls, exp_coeff_list: list[tuple], *gens: sp.Symbol):
        return cls.from_dict({k: v for k, v in exp_coeff_list}, *gens)

    @classmethod
    def from_dict(cls, exp_coeff_map: dict, *gens: sp.Symbol):
        red_map = {deg: cff for deg, cff in exp_coeff_map.items() if cff != 0}
        return cls.new(red_map, *gens)

    @classmethod
    def new(cls, rep, *gens):
        assert isinstance(rep, dict)
        assert isinstance(gens[0], (sp.Symbol, sp.Expr))
        obj = sp.Basic.__new__(cls, rep, *gens)
        obj.rep = rep
        obj.gens = gens
        return obj

    # @property
    # def expr(self):
    #     return self.as_expr()

    # @property
    # def args(self):
    #     return (self.rep,) + self.gens

    def _hashable_content(self):
        return tuple(self.rep.items()) + self.gens

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        return f.add(g)

    def __radd__(f, g):
        return f.add(g)

    def __sub__(f, g):
        return f.sub(g)

    def __rsub__(f, g):
        return f.neg().add(g)

    def __mul__(f, g):
        return f.mul(g)

    def __rmul__(f, g):
        return f.mul(g)

    def __truediv__(f, g):
        return f.div(g)

    def __rtruediv__(f, g):
        return f.mul(g ** (-1))

    def __str__(self):
        return str(f"QPoly({self.exponents()}, {self.gen})")

    def __repr__(self):
        return str(f"QPoly({self.exponents()}, {self.gen})")

    def add_ground(f, g):
        if g == S.Zero:
            return QPoly(f, *f.gens)
        rd = f.as_dict()
        c0 = {0: rd.get(0, S.Zero) + g}
        return QPoly.from_dict(rd | c0, *f.gens)

    def add(f, g):
        g = sp.sympify(g)
        if not f.rep:
            return g
        if not isinstance(g, QPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.add_ground(g)
            else:
                g = QPoly(g, *f.gens)

        # extract exponents
        fe = set(f.exponents())
        ge = set(g.exponents())
        common_exps = fe.intersection(ge)
        f_only_exps = fe.difference(common_exps)
        g_only_exps = ge.difference(common_exps)

        # join coefficients
        fd = f.as_dict()
        gd = g.as_dict()
        rd = {exp: (fd[exp] + gd[exp]).expand().cancel() for exp in common_exps}
        rd.update({exp: fd[exp] for exp in f_only_exps})
        rd.update({exp: gd[exp] for exp in g_only_exps})

        return QPoly.from_dict(rd, f.gen)

    def neg(f):
        rd = {exp: -cff for exp, cff in f.as_dict().items()}
        return QPoly.from_dict(rd, f.gen)

    def sub(f, g):
        return f.add(-g)

    def mul_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff * g

        return QPoly.from_dict(rd, f.gen)

    def mul(f, g):
        g = sp.sympify(g)
        if g == S.Zero:
            return QPoly(0, *f.gens)
        if not isinstance(g, QPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.mul_ground(g)
            g = QPoly(g, f.gen)

        r_dict = {}
        for fd, fc in f.as_dict().items():
            for gd, gc in g.as_dict().items():
                rd = fd + gd
                rc = fc * gc
                if rd in r_dict:
                    r_dict[rd] = (r_dict[rd] + rc).expand().cancel()
                else:
                    r_dict[rd] = rc
        return QPoly.from_dict(r_dict, f.gen)

    def div_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff / g
        return QPoly.from_dict(rd, f.gen)

    def div(f, g):
        g = sp.sympify(g)
        if not isinstance(g, QPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.div_ground(g)

        # fallback
        return f.as_expr() / g.as_expr()

    def as_dict(self) -> dict:
        # TODO copy is a hack to get rid of side effects
        return self.rep.copy()

    def _eval_is_polynomial(self, gens):
        return False

    def as_coeff_Mul(self, *deps, **kwargs):
        return self.as_expr().as_coeff_Mul(*deps, **kwargs)

    def as_coeff_Add(self, *deps, **kwargs):
        return self.as_expr().as_coeff_Add(*deps, **kwargs)

    def as_numer_denom(self):
        """
        Return GenPoly with coefficient numerators as well as their common denominator
        """
        c_denoms = [c.as_numer_denom()[1] for c in self.coefficients()]
        denom = sp.lcm(c_denoms)

        nd = {exp: sp.cancel(cff * denom) for exp, cff in self.as_dict().items()}
        numer = QPoly.from_dict(nd, self.gen)
        return numer, denom

    def as_base_exp(self):
        return self.as_expr().as_base_exp()

    @staticmethod
    def dict_to_expr(exp_coeff_map: dict, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a dict
        """
        expr = S.Zero
        for k, v in exp_coeff_map.items():
            expr += v * gen**k
        return expr

    @staticmethod
    def list_to_expr(exp_coeff_pairs: list, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a list
        """
        expr = 0
        for k, v in exp_coeff_pairs:
            expr += v * gen**k
        return expr

    @staticmethod
    def dict_from_expr(
        expr: sp.Expr, *gens: Union[sp.Symbol, sp.Expr]
    ) -> dict[sp.Expr, sp.Expr]:
        """
        Variant of polyutils '_parallel_dict_from_expr_if_gens' that will
        work with rational powers of generators in an expression and thus
        create powers and coefficients of a generalized polynomial.
        """
        assert len(gens) == 1
        gen = gens[0]
        if expr.is_Poly:
            if expr.gen == gen:
                return expr.as_dict()
            else:
                expr = expr.as_expr()
        else:
            expr = sp.sympify(expr)
        w = sp.Wild("w")
        if not isinstance(gen, sp.Symbol):
            # fix issues with compound generators
            if isinstance(gen, sp.exp):
                r = gen.func(gen.exp * w)
                gen = sp.Dummy(positive=True)
                expr = expr.replace(r, gen**w)
            else:
                print("Warning compound generators may cause problems")

        e_c_map = exponent_coeff_dict(expr, gen)
        return e_c_map

    def as_expr(self) -> sp.Expr:
        return self.dict_to_expr(self.rep, self.gen)

    @property
    def gen(self) -> sp.Symbol:
        return self.gens[0]

    @property
    def is_zero(self) -> bool:
        return not bool(self.rep)

    def degree(self, s_dict: dict = None) -> sp.Expr:
        return self.degmax(s_dict) - self.degmin(s_dict)

    def degmin(self, s_dict: dict = None) -> sp.Expr:
        if s_dict is None:
            return self._smin(self.exponents())
        else:
            exp_values = [exp.subs(s_dict) for exp in self.exponents()]
            return min(exp_values, default=0)

    def degmax(self, s_dict: dict = None) -> sp.Expr:
        if s_dict is None:
            return self._smax(self.exponents())
        else:
            exp_values = [exp.subs(s_dict) for exp in self.exponents()]
            return max(exp_values, default=0)

    def _smax(self, args, default=0):
        m = None
        for a in args:
            if m is None or sp.assumptions.ask(a > m):
                m = a
        if m is None:
            return default
        else:
            return m

    def _smin(self, args, default=0):
        m = None
        for a in args:
            if m is None or sp.assumptions.ask(a < m):
                m = a
        if m is None:
            return default
        else:
            return m

    def LM(self, s_dict: dict = None):
        """Return the leading monom"""
        return QPoly.from_list([(self.degmax(), self.rep[self.degmax()])], self.gen)

    def TM(self, s_dict: dict = None):
        """Return the trailing monom"""
        return QPoly.from_list([(self.degmin(), self.rep[self.degmin()])], self.gen)

    def LC(self, s_dict: dict = None) -> sp.Expr:
        """Return the Leading coefficient"""
        return self.rep[self.degmax(s_dict)]

    def TC(self, s_dict: dict = None) -> sp.Expr:
        """Return the Trailing coefficient"""
        return self.rep[self.degmin(s_dict)]

    def exponents(self) -> list[sp.Expr]:
        return [k for k in self.rep.keys()]

    def coefficients(self) -> list[sp.Expr]:
        return [v for v in self.rep.values()]

    def get_coeff_denom_lcm(self) -> sp.Expr:
        """
        Compute the least common multiple of all coefficient denominators
        """
        coeffs = self.coefficients()
        lcm = 1
        for c in coeffs:
            nc, dc = c.as_numer_denom()
            lcm = sp.lcm(lcm, dc)
        return lcm


def to_base_exponent(term, gen):
    """
    Attempt to write `term` as coeff * gen**exp where coeff and exp do not
    contain `gen`.

    Note:
        The splitting is conservative:
      - Only factors that are exactly `sym` or `sym**e` (with e independent of `sym`)
        are pulled into the exponent.
      - Any other occurrence of `sym` in a factor causes failure.

    Returns: Tuple(coeff, exp, ok), where ok is True on success.
    """
    # quick exit
    if term == gen:
        return sp.Integer(1), sp.Integer(1), True

    # term is a pure power of the generator
    if term.is_Pow and term.base == gen:
        e = term.exp
        if gen in e.free_symbols:
            return None, None, False
        return sp.Integer(1), e, True

    # term is an atom (symbol/number) and not gen
    if term.is_Atom:
        # it either equals gen (already handled) or is independent of gen
        if gen in term.free_symbols:
            # atom involves sym in a non-power way (unlikely except sym itself)
            return None, None, False
        return term, sp.Integer(0), True

    # a product, scan factors
    if term.is_Mul:
        coeff = sp.Integer(1)
        exp = sp.Integer(0)
        for f in sp.Mul.make_args(term):
            if f == gen:
                exp += 1
                continue
            if f.is_Pow and f.base == gen:
                e = f.exp
                if gen in e.free_symbols:
                    return (None, None, False)
                exp += e
                continue
            # a one-over term, parse the denominator and return its inverse
            if f.is_Pow and f.exp == sp.S.NegativeOne and gen in f.base.free_symbols :
                _c, _e, _ok = to_base_exponent(sp.factor_terms(f.base), gen)
                coeff /= _c
                exp -= _e
                continue
            # any other factor that contains gen makes it invalid
            if gen in f.free_symbols:
                return None, None, False
            coeff *= f
        return sp.simplify(coeff), sp.simplify(exp), True

    # a power with non-sym base or other composite:
    # - If it contains gen anywhere, we reject it
    if gen in term.free_symbols:
        return None, None, False

    # Otherwise independent of sym -> constant term
    return term, sp.Integer(0), True


def exponent_coeff_dict(expr, gen, *, strict=True, expand=True):
    """
    Parse `expr` as a quasi polynomial in `sym`, collecting coefficients
    by exponent: sum of c*sym**e with c,e independent of `sym`.

    Parameters
    ----------
    expr : sp.Expr
    gen : sp.Symbol
    strict : bool (default True)
        - If True, any term that cannot be written as c*sym**e with c,e
          independent of `sym` raises ValueError.
        - If False, such terms are accumulated in a remainder and returned.
    expand : bool (default True)
        - If True, expands `expr` before collecting. Set to False if you already
          control the addends.

    Returns
    -------
    dict[Expr, Expr]  (or tuple if strict=False)
        Mapping exponent -> coefficient. Exponent 0 is the constant term.
        If strict=False, also returns the symbolic remainder.

    Notes
    -----
    - Combination is by SymPy structural equality of exponents.
    - Only collects explicit factors whose base is exactly `sym`.
    """
    expr = sp.expand(expr) if expand else expr
    terms = expr.as_ordered_terms() if expr.is_Add else [expr]

    out = defaultdict(lambda: sp.Integer(0))
    other = sp.Integer(0)

    for t in terms:
        tf = sp.factor_terms(t, fraction=True)
        coeff, exp, ok = to_base_exponent(tf, gen)
        if not ok:
            if strict:
                raise ValueError(f"Term {t} is not a separable c*{gen}**e with c,e independent of {gen}.")
            else:
                other += t
                continue
        out[exp] += coeff

    # Prune zeros and simplify coefficients
    out = {e: sp.simplify(c) for e, c in out.items() if not c.equals(0)}

    if strict:
        return out
    else:
        return out, sp.simplify(other)
