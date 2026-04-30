from collections import defaultdict
from operator import lt, le, gt, ge
from functools import reduce
from typing import Union, Any

import sympy as sp
from sympy import ExactQuotientFailed, CoercionFailed
from sympy.core.numbers import S
from sympy.core.sympify import CantSympify
from sympy.polys.fields import FracElement
from sympy.polys.domains.domain import DomainElement
from sympy.polys.domains.ring import Ring
from sympy.printing.defaults import DefaultPrinting
from sympy.polys.orderings import lex

__all__ = ["QPoly", "QuasiPolynomialElement", "QuasiPolynomialRing"]


class QuasiPolynomialElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of univariate quasipolynomial ring."""

    is_Poly = False
    is_QPoly = True

    def __init__(self, ring, init):
        super().__init__(init)
        self.ring = ring

    def new(self, init):
        """Create a new quasi polynomial element in the same ring."""
        return self.__class__(self.ring, init)

    def parent(self):
        return self.ring.to_domain()

    def is_polynomial(self, sym):
        return False

    def is_rational_function(self, sym):
        return False

    _hash = None

    def __hash__(self):
        # XXX: This computes a hash of a dictionary, but currently we don't
        # protect dictionary from being changed so any use site modifications
        # will make hashing go wrong. Use this feature with caution until we
        # figure out how to make a safe API without compromising speed of this
        # low-level class.
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def _hashable_content(self):
        return tuple(self.items()) + self.gens

    def copy(self):
        """Return a copy of quasi polynomial self."""
        return self.new(self)

    def subs(self, x, a=None):
        if isinstance(x, list) and a is None:
            f = self.copy()
            for X, a in x:
                f = f.subs(X, a)
            return f

        ring = self.ring
        a = ring.domain.convert(a)
        terms = {}
        for monom, coeff in self.iterterms():
            terms[monom.subs(x,a)] = coeff.subs(x,a)
        syms = [s.subs(x,a) for s in ring.symbols]
        raise NotImplementedError

    def as_expr(self):
        return self.dict_to_expr(self.to_dict(), *self.ring.symbols)

    def __eq__(q1, q2):
        if not q2:
            return not q1
        elif q1.ring.is_element(q2):
            return dict.__eq__(q1, q2)
        elif len(q1) > 1:
            return False
        else:
            return q1.get(q1.ring.zero_monom) == q2

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(q1, q2, op):
        if q1.ring.is_element(q2):
            return op(q1.degree(), q2.degree())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)

    def __le__(p1, p2):
        return p1._cmp(p2, le)

    def __gt__(p1, p2):
        return p1._cmp(p2, gt)

    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def to_dict(self):
        return dict(self)

    @property
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC() == 1)

    @property
    def is_term(self):
        return len(self) <= 1

    # @property
    # def is_negative(self):
    #     return self.ring.domain.is_negative(self.LC)
    #
    # @property
    # def is_positive(self):
    #     return self.ring.domain.is_positive(self.LC)
    #
    # @property
    # def is_nonnegative(self):
    #     return self.ring.domain.is_nonnegative(self.LC)
    #
    # @property
    # def is_nonpositive(self):
    #     return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    # def __new__(cls, rep, *gens):
    #     """Create a new quasipolynomial instance out of something useful."""
    #     if isinstance(rep, QPoly):
    #         return cls.from_dict(rep.as_dict(), *gens)
    #     if isinstance(rep, dict):
    #         return cls.from_dict(rep, *gens)
    #     elif isinstance(rep, list):
    #         return cls.from_list(rep, *gens)
    #     elif isinstance(rep, sp.Add):
    #         if any(isinstance(arg, QPoly) for arg in rep.args):
    #             return cls(reduce(lambda f, g: f + g, rep.args), *gens)
    #
    #     # last resort
    #     rep = sp.sympify(rep, evaluate=type(rep) is not str)
    #     return cls.from_expr(rep, *gens)

    @classmethod
    def from_qpoly(cls, expr, ring: Ring):
        return cls.from_dict(expr.as_dict(), ring)

    @classmethod
    def from_expr(cls, expr: sp.Expr, ring: Ring):
        return cls.from_dict(cls.dict_from_expr(expr, ring), ring)

    @classmethod
    def from_list(cls, exp_coeff_list: list[tuple], ring: Ring):
        return cls.from_dict({k: v for k, v in exp_coeff_list}, ring)

    @classmethod
    def from_dict(cls, exp_coeff_map: dict, ring: Ring):
        cff_dom = ring.domain
        red_map = {}
        for exp, cff in exp_coeff_map.items():
            if not isinstance(cff, DomainElement):
                cff = cff_dom.from_sympy(cff)
            else:
                cff = cff_dom.convert(cff)
            if cff == cff_dom.zero:
                continue
            red_map[exp] = cff
        return cls(ring, red_map)

    def __pos__(self):
        return self

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

    def __divmod__(f, g):
        ring = f.ring

        if not g:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(g):
            return f.div(g)
        elif isinstance(g, QuasiPolynomialElement):
            if isinstance(ring.domain, QuasiPolynomialRing) and ring.domain.ring == g.ring:
                pass
            elif isinstance(g.ring.domain, QuasiPolynomialRing) and g.ring.domain.ring == ring:
                return g.__rdivmod__(f)
            else:
                return NotImplemented

        try:
            g = ring.domain_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.div(g)

    def __rdivmod__(f, g):
        ring = f.ring
        try:
            g = ring.ground_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return g.div(f)

    def __mod__(f, g):
        ring = f.ring

        if not g:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(g):
            return f.div(g)[1]
        elif isinstance(g, QuasiPolynomialElement):
            if isinstance(ring.domain, QuasiPolynomialRing) and ring.domain.ring == g.ring:
                pass
            elif isinstance(g.ring.domain, QuasiPolynomialRing) and g.ring.domain.ring == ring:
                return g.__rmod__(f)
            else:
                return NotImplemented

        try:
            g = ring.domain_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.div_ground(g)[1]

    def __rmod__(f, g):
        ring = f.ring
        try:
            g = ring.ground_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return g.rem(f)

    def __floordiv__(f, g):
        ring = f.ring

        if not g:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(g):
            return f.div(g)[0]
        elif isinstance(g, QuasiPolynomialElement):
            if isinstance(ring.domain, QuasiPolynomialRing) and ring.domain.ring == g.ring:
                pass
            elif isinstance(g.ring.domain, QuasiPolynomialRing) and g.ring.domain.ring == ring:
                return g.__rtruediv__(f)
            else:
                return NotImplemented

        try:
            g = ring.domain_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.div_ground(g)

    def __rfloordiv__(f, g):
        ring = f.ring
        try:
            g = ring.ground_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return g.div(f)[0]

    def __truediv__(f, g):
        ring = f.ring

        if not g:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(g):
            return f.exquo(g)
        elif isinstance(g, QuasiPolynomialElement):
            if isinstance(ring.domain, QuasiPolynomialRing) and ring.domain.ring == g.ring:
                pass
            elif isinstance(g.ring.domain, QuasiPolynomialRing) and g.ring.domain.ring == ring:
                return g.__rtruediv__(f)
            else:
                return NotImplemented

        try:
            g = ring.domain_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.quo_ground(g)

    def __rtruediv__(f, g):
        ring = f.ring
        try:
            g = ring.ground_new(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return g.exquo(f)

    def __pow__(self, n):
        """ raise quasipolynomial to power `n` """
        ring = self.ring
        if not n:
            if self:
                return ring.one
            else:
                raise ValueError("0**0")
        elif len(self) == 1:
            monom, coeff = list(self.items())[0]
            p = ring.zero
            p[monom*n] = coeff**n
            return p
        else:
            raise NotImplementedError("No general exponentiation supported")

    def __str__(self):
        return str(
            f"QuasiPolynomialElement({self.exponents()}, domain={self.ring.domain})"
        )

    # def __repr__(self):
    #     return str(f"QuasiPolynomialElement({self.exponents()}, gens={self.gens})")

    def add_ground(f, g):
        if g == S.Zero:
            return f.copy()
        c0 = {0: f.coeff(0) + g}
        return QuasiPolynomialElement.from_dict(f.to_dict() | c0, f.ring)

    def add(f, g):
        if not f:
            return g
        if not isinstance(g, QuasiPolynomialElement):
            g = sp.sympify(g)
            if f.gen not in g.as_expr().free_symbols:
                return f.add_ground(g)
            else:
                g = QuasiPolynomialElement.from_expr(g, f.ring)

        # extract exponents
        fe = set(f.exponents())
        ge = set(g.exponents())
        common_exps = fe.intersection(ge)
        f_only_exps = fe.difference(common_exps)
        g_only_exps = ge.difference(common_exps)

        # join coefficients
        rd = {}
        for exp in common_exps:
            fc = f.coeff(exp)
            gc = g.coeff(exp)
            try:
                rc = fc + gc
            except TypeError:
                # help sympys domain system
                fn, gn = f._unify_coeffs(fc, gc)
                rc = fn + gn
            rd[exp] = rc
        rd.update({exp: f.coeff(exp) for exp in f_only_exps})
        rd.update({exp: g.coeff(exp) for exp in g_only_exps})

        return QuasiPolynomialElement.from_dict(rd, f.ring)

    def neg(f):
        rd = {exp: -cff for exp, cff in f.items()}
        return QuasiPolynomialElement.from_dict(rd, f.ring)

    def sub(f, g):
        return f.add(-g)

    def mul_ground(f, g):
        g = f.ring.domain(g)
        rd = {}
        for exp, cff in f.items():
            rd[exp] = cff * g
        # gens = f.gens + tuple(g.free_symbols)
        return QuasiPolynomialElement.from_dict(rd, f.ring)

    def mul(f, g):
        if g == S.Zero:
            return f.ring.zero
        if not isinstance(g, QuasiPolynomialElement):
            g = sp.sympify(g)
            if f.gen not in g.as_expr().free_symbols:
                return f.mul_ground(g)
            g = f.ring(g)

        r_dict = {}
        for fd, fc in f.items():
            for gd, gc in g.items():
                if isinstance(fd, DomainElement):
                    gd = fd.dom.convert(gd)
                # except TypeError:
                #     # help sympys domain system
                #     fn, gn = f._unify_exponents(fd, gd)
                rd = fd + gd

                try:
                    rc = fc * gc
                except TypeError:
                    # help sympys domain system
                    fn, gn = f._unify_coeffs(fc, gc)
                    rc = fn * gn
                if rd in r_dict:
                    r_dict[rd] = r_dict[rd] + rc
                else:
                    r_dict[rd] = rc
        return QuasiPolynomialElement.from_dict(r_dict, f.ring)

    def div_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.items():
            rd[exp] = cff / g
        return QuasiPolynomialElement.from_dict(rd, f.ring)

    def _qdiv(f, g):


        g_dmax = g.degmax()
        g_dmin = g.degmin()

        q = f.from_dict({0: 0}, f.ring)
        r = f
        while True:
            r_dmax = r.degmax()
            r_dmin = r.degmin()

            degmax_correct = sp.assumptions.ask(r_dmax <= g_dmax)
            degmin_correct = sp.assumptions.ask(r_dmin >= g_dmin)

            if degmax_correct is None or degmin_correct is None:
                raise ValueError(
                    "Degree order cannot be decided. Make sure that"
                    "the given expressions exponents to not contain"
                    "any symbols."
                )
            if not degmax_correct:
                # reduce predictions
                q_kd = {(r_dmax - g_dmax): r.LC() / g.LC()}
            elif not degmin_correct:
                # reduce delays
                q_kd = {(r_dmin - g_dmin): r.TC() / g.TC()}
            else:
                break
            q_k = QuasiPolynomialElement.from_dict(q_kd, q.ring)
            r = r - q_k * g
            q += q_k
        return q, r

    def div(f, g):
        if not isinstance(g, QuasiPolynomialElement):
            return f.div_ground(g)
        assert f.ring.is_element(g)
        assert not g.is_zero
        return f._qdiv(g)

    def rdiv(f, g):
        fi = f ** (-1)
        g = f.ring(g)
        return g.mul(fi)

    def quo_ground(f, x):
        domain = f.ring.domain

        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f

        if domain.is_Field:
            quo = domain.quo
            terms = [ (monom, quo(coeff, x)) for monom, coeff in f.iterterms() ]
        else:
            terms = [ (monom, coeff // x) for monom, coeff in f.iterterms() if not (coeff % x) ]

        return f.new(terms)

    def _term_div(self, a_lm_a_lc, b_lm_b_lc):
        """ Copied from PolyElement """
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo

        a_lm, a_lc = a_lm_a_lc
        b_lm, b_lc = b_lm_b_lc
        if b_lm == zm: # apparently this is a very common case
            monom = a_lm
        else:
            monom = a_lm - b_lm

        if monom is not None:
            return monom, domain_quo(a_lc, b_lc)
        else:
            return None

    def quo_term(f, term):
        monom, coeff = term

        if not coeff:
            raise ZeroDivisionError("polynomial division")
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)

        terms = [f._term_div(t, term) for t in f.iterterms() ]
        return f.new([ t for t in terms if t is not None ])

    def exquo(f, G):
        q, r = f.div(G)
        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)

    def _unify_exponents(self, fd, gd) -> tuple[Any, Any]:
        new_dom = fd.domain.unify(gd.domain)
        fn = new_dom.convert_from(fd, fd.domain)
        gn = new_dom.convert_from(gd, gd.domain)
        return fn, gn

    def _unify_coeffs(self, fc, gc) -> tuple[Any, Any]:
        new_dom = fc.field.to_domain().unify(gc.field.to_domain())
        fn = new_dom.convert_from(fc, fc.field.to_domain())
        gn = new_dom.convert_from(gc, gc.field.to_domain())
        return fn, gn

    def _eval_is_polynomial(self, gens):
        return False

    @staticmethod
    def _get_coeff_domain(*c_gens):
        return sp.EX.frac_field(*c_gens)

    def as_numer_denom(self):
        """
        Return QuasiPolynomial with coefficient numerators as well as their common denominator
        """
        c_denoms = [c.numer for c in self.coefficients()]
        if not c_denoms:
            return sp.Integer(0), sp.Integer(1)
        denom = reduce(c_denoms[0].ring.domain.lcm, c_denoms)
        nd = {exp: cff * denom for exp, cff in self.items()}
        numer = QuasiPolynomialElement.from_dict(nd, *self.gens)
        return numer, denom

    @staticmethod
    def dict_to_expr(exp_coeff_map: dict, *gens: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a dict
        """
        assert len(gens) == 1
        expr = S.Zero
        for k, v in exp_coeff_map.items():
            expr += v.as_expr() * gens[0] ** k
        return expr

    @staticmethod
    def list_to_expr(exp_coeff_pairs: list, *gens: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a list
        """
        assert len(gens) == 1
        expr = 0
        for k, v in exp_coeff_pairs:
            expr += v * gens[0] ** k
        return expr

    @staticmethod
    def dict_from_expr(expr: sp.Expr, ring: Ring) -> dict[sp.Expr, sp.Expr]:
        """
        Variant of polyutils '_parallel_dict_from_expr_if_gens' that will
        work with rational powers of generators in an expression and thus
        create powers and coefficients of a generalized polynomial.
        """
        q_gen = ring.symbols[0]

        if expr.is_Poly:
            if expr.gen == q_gen:
                return expr.as_dict()
            else:
                expr = expr.as_expr()
        else:
            expr = sp.sympify(expr)

        w = sp.Wild("w")
        if not isinstance(q_gen, sp.Symbol):
            # fix issues with compound generators
            if isinstance(q_gen, sp.exp):
                r = q_gen.func(q_gen.exp * w)
                q_gen = sp.Dummy(positive=True)
                expr = expr.replace(r, q_gen**w)
            else:
                print("Warning compound generators may cause problems")

        e_c_map = exponent_coeff_dict(expr, q_gen)
        return e_c_map

    @property
    def gen(self) -> sp.Symbol:
        return self.ring.symbols[0]

    def degree(self, s_dict: dict = None) -> sp.Expr:
        return self.degmax(s_dict) - self.degmin(s_dict)

    def degmin(self, s_dict: dict = None) -> sp.Expr:
        if s_dict is None:
            return self._smin(self.exponents(), default=sp.S.Infinity)
        else:
            exp_values = [exp.subs(s_dict) for exp in self.exponents()]
            return min(exp_values, default=0)

    def degmax(self, s_dict: dict = None) -> sp.Expr:
        if s_dict is None:
            return self._smax(self.exponents(), default=sp.S.NegativeInfinity)
        else:
            exp_values = [exp.subs(s_dict) for exp in self.exponents()]
            return max(exp_values, default=0)

    def _smax(self, args, default=0):
        m = None
        for a in args:
            if m is None or sp.assumptions.ask(a > m):
                m = a
        return default if m is None else m

    def _smin(self, args, default=0):
        m = None
        for a in args:
            if m is None or sp.assumptions.ask(a < m):
                m = a
        return default if m is None else m

    def LT(self):
        """ Return the leading term """
        dm = self.degmax()
        return QuasiPolynomialElement.from_list(
            [(dm, self.coeff(dm))], self.ring
        )

    def TT(self):
        """ Return the trailing term """
        dm = self.degmin()
        return QuasiPolynomialElement.from_list(
            [(dm, self.coeff(dm))], self.ring
        )

    def LM(self, s_dict: dict = None):
        """ Return the leading monom """
        return QuasiPolynomialElement.from_dict({self.degmax(): 1}, self.ring)

    def TM(self, s_dict: dict = None):
        """Return the trailing monom"""
        return QuasiPolynomialElement.from_dict({self.degmin(): 1}, self.ring)

    def LC(self, s_dict: dict = None) -> sp.Expr:
        """Return the Leading coefficient"""
        ld = self.degmax(s_dict)
        return self.coeff(ld)

    def TC(self, s_dict: dict = None) -> sp.Expr:
        """Return the Trailing coefficient"""
        td = self.degmin(s_dict)
        return self.coeff(td)

    def exponents(self) -> list[sp.Expr]:
        return [k for k in self.keys()]

    def coefficients(self) -> list[sp.Expr]:
        return [v for v in self.values()]

    def coeff(self, exp):
        if exp in self.keys():
            return self[exp]
        else:
            return self.ring.domain.zero

    def iterterms(self):
        """Iterator over terms of a quasipolynomial. """
        return iter(self.items())

    # def get_coeff_denom_lcm(self) -> sp.Expr:
    #     """
    #     Compute the least common multiple of all coefficient denominators
    #     """
    #     coeffs = self.coefficients()
    #     lcm = 1
    #     for c in coeffs:
    #         nc, dc = c.as_numer_denom()
    #         lcm = sp.lcm(lcm, dc)
    #     return lcm


class QuasiPolynomialRing(Ring):
    """Domain for quasipolynomials"""

    is_PolynomialRing = is_Poly = False
    is_QuasiPolynomialRing = is_QPoly = True

    is_Composite = True

    has_assoc_Ring = True
    has_assoc_Field = False

    dtype = QuasiPolynomialElement

    def __init__(self, domain, symbols, order=lex):
        self.domain = domain
        # HACK: Base class demands this
        self.dom = self.domain
        if not isinstance(symbols, tuple):
            symbols = (symbols, )
        self.symbols = symbols
        self.order = sp.polys.orderings.monomial_key(order)

        self.zero_monom = 0

    def __str__(self):
        return str(self.domain) + "{" + ",".join(map(str, self.symbols)) + "}"

    @property
    def zero(self):
        return self.dtype(self, {})

    @property
    def one(self):
        return self.dtype(self, {0: 1})

    def characteristic(self):
        return self.domain.characteristic()

    def is_element(self, element):
        """True if ``element`` is an element of this ring. False otherwise."""
        return isinstance(element, QuasiPolynomialElement) and element.ring == self

    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        return self.term_new(0, coeff)

    def term_new(self, monom, coeff):
        coeff = self.domain_new(coeff)
        qpoly = self.zero
        if coeff:
            qpoly[monom] = coeff
        return qpoly

    def ring_new(self, element):
        if isinstance(element, QuasiPolynomialElement):
            if self == element.ring:
                return element
            elif (
                isinstance(self.domain, QuasiPolynomialElement)
                and self.domain.ring == element.ring
            ):
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return QuasiPolynomialElement.from_dict(element, self)
        elif isinstance(element, list):
            return QuasiPolynomialElement.from_list(element, self)
        elif isinstance(element, sp.Expr):
            return QuasiPolynomialElement.from_expr(element, self)
        else:
            return self.ground_new(element)

    __call__ = ring_new

    def from_sympy(self, expr):
        return QuasiPolynomialElement(self, expr)

    def to_domain(self):
        return QuasiPolynomialRing(self, self.symbols)

    def to_sympy(self, element):
        return element.as_expr()

    def convert_from(self, element, base):
        if not base.is_Composite:
            return self(base.to_sympy(element))
        # elif self.of_type(element):
        return self(base.to_sympy(element))

        # return QuasiPolynomialElement(element.to_sympy(), base.symbols)
        raise CoercionFailed("Implement more, I guess?")

    # def unify(K0, K1, symbols=None):
    #     if K1.
    #     raise NotImplementedError

    # def of_type(self, element):
    #     if element.domain == self.domain:
    #         return True
    #     else:
    #         return False


class QPoly(sp.Basic):
    """
    Class to handle quasipolynomials, which are polynomials but with real exponents
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
        free_syms = []
        for c in exp_coeff_map.values():
            if isinstance(c, FracElement):
                free_syms.append(c.as_expr().free_symbols)
            else:
                free_syms.append(c.free_symbols)
        if free_syms:
            c_gens = set.union(*free_syms)
        else:
            c_gens = []
        cff_dom = cls._get_coeff_domain(*c_gens)
        red_map = {}
        for exp, cff in exp_coeff_map.items():
            if not isinstance(cff, FracElement):
                cff = cff_dom.from_sympy(cff)
            if cff == cff_dom.zero:
                continue
            red_map[exp] = cff
        gens = (gens[0], *c_gens)
        return cls.new(red_map, *gens)

    @classmethod
    def new(cls, rep, *gens):
        assert isinstance(rep, dict)
        assert isinstance(gens[0], (sp.Symbol, sp.Expr))
        obj = sp.Basic.__new__(cls, rep, *gens)
        obj.rep = rep
        obj.gens = gens
        return obj

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
        return str(f"QPoly({self.exponents()}, gens={self.gens})")

    def __repr__(self):
        return str(f"QPoly({self.exponents()}, gens={self.gens})")

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
        rd = {}
        for exp in common_exps:
            fc = fd[exp]
            gc = gd[exp]
            try:
                rc = fc + gc
            except TypeError:
                # help sympys domain system
                fn, gn = f._unify_coeffs(fc, gc)
                rc = fn + gn
            rd[exp] = rc
        rd.update({exp: fd[exp] for exp in f_only_exps})
        rd.update({exp: gd[exp] for exp in g_only_exps})

        return QPoly.from_dict(rd, *f.gens)

    def neg(f):
        rd = {exp: -cff for exp, cff in f.as_dict().items()}
        return QPoly.from_dict(rd, *f.gens)

    def sub(f, g):
        return f.add(-g)

    def mul_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff * g
        gens = f.gens + tuple(g.free_symbols)
        return QPoly.from_dict(rd, *gens)

    def mul(f, g):
        g = sp.sympify(g)
        if g == S.Zero:
            return QPoly(0, *f.gens)
        if not isinstance(g, QPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.mul_ground(g)
            g = QPoly(g, *f.gens)

        r_dict = {}
        for fd, fc in f.as_dict().items():
            for gd, gc in g.as_dict().items():
                rd = fd + gd
                try:
                    rc = fc * gc
                except TypeError:
                    # help sympys domain system
                    fn, gn = f._unify_coeffs(fc, gc)
                    rc = fn * gn
                if rd in r_dict:
                    r_dict[rd] = r_dict[rd] + rc
                else:
                    r_dict[rd] = rc
        return QPoly.from_dict(r_dict, *f.gens)

    def _unify_coeffs(self, fc, gc) -> tuple[Any, Any]:
        new_dom = fc.field.to_domain().unify(gc.field.to_domain())
        fn = new_dom.convert_from(fc, fc.field.to_domain())
        gn = new_dom.convert_from(gc, gc.field.to_domain())
        return fn, gn

    def div_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff / g
        return QPoly.from_dict(rd, *f.gens)

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

    @staticmethod
    def _get_coeff_domain(*c_gens):
        return sp.EX.frac_field(*c_gens)

    # def as_coeff_Mul(self, *deps, **kwargs):
    #     return self.as_expr().as_coeff_Mul(*deps, **kwargs)

    # def as_coeff_Add(self, *deps, **kwargs):
    #     return self.as_expr().as_coeff_Add(*deps, **kwargs)

    def as_numer_denom(self):
        """
        Return GenPoly with coefficient numerators as well as their common denominator
        """
        c_denoms = [c.numer for c in self.coefficients()]
        if not c_denoms:
            return sp.Integer(0), sp.Integer(1)
        denom = reduce(c_denoms[0].ring.domain.lcm, c_denoms)
        nd = {exp: cff * denom for exp, cff in self.as_dict().items()}
        numer = QPoly.from_dict(nd, *self.gens)
        return numer, denom

    # def as_base_exp(self):
    #     return self.as_expr().as_base_exp()

    @staticmethod
    def dict_to_expr(exp_coeff_map: dict, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a dict
        """
        expr = S.Zero
        for k, v in exp_coeff_map.items():
            expr += v.as_expr() * gen**k
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
        q_gen = gens[0]

        if expr.is_Poly:
            if expr.gen == q_gen:
                return expr.as_dict()
            else:
                expr = expr.as_expr()
        else:
            expr = sp.sympify(expr)

        w = sp.Wild("w")
        if not isinstance(q_gen, sp.Symbol):
            # fix issues with compound generators
            if isinstance(q_gen, sp.exp):
                r = q_gen.func(q_gen.exp * w)
                q_gen = sp.Dummy(positive=True)
                expr = expr.replace(r, q_gen**w)
            else:
                print("Warning compound generators may cause problems")

        e_c_map = exponent_coeff_dict(expr, q_gen)
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
        return QPoly.from_list([(self.degmax(), self.rep[self.degmax()])], *self.gens)

    def TM(self, s_dict: dict = None):
        """Return the trailing monom"""
        return QPoly.from_list([(self.degmin(), self.rep[self.degmin()])], *self.gens)

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

    def coeff(self, exp):
        return self.rep[exp]

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
            if f.is_Pow and f.exp == sp.S.NegativeOne and gen in f.base.free_symbols:
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
                raise ValueError(
                    f"Term {t} is not a separable c*{gen}**e with c,e independent of {gen}."
                )
            else:
                other += t
                continue
        out[exp] += coeff

    # Prune zeros and simplify coefficients
    out = {e: sp.ratsimp(c) for e, c in out.items()}

    if strict:
        return out
    else:
        return out, other
