from collections import defaultdict
from functools import reduce
from operator import ge, gt, le, lt
from typing import Any, Union, Dict

import sympy as sp
from sympy import CoercionFailed, ExactQuotientFailed, PolynomialRing
from sympy.core.numbers import S
from sympy.core.sympify import CantSympify
from sympy.polys.domains.domain import DomainElement
from sympy.polys.domains.ring import Ring
from sympy.polys.fields import FracElement
from sympy.polys.orderings import lex
from sympy.polys.rings import PolyElement, PolyRing
from sympy.printing.defaults import DefaultPrinting

__all__ = ["QPoly", "QuasiPolynomialElement", "QuasiPolynomialRing"]


class CachedPoly:
    """
    Thin wrapper around PolyElement that memorizes factor_list().

    It holds checks for dunders and __getattr__ to see when the underlying
    poly might change.
    """

    __slots__ = ('_poly', '_factors')

    # Regular methods that mutate the underlying dict in place.
    _MUTATING_METHODS = frozenset({
        'strip_zero',  # del self[monom]          rings.py:1327
        'imul_num',  # in-place scalar mul      used at rings.py:2324
        '_iadd_poly_monom',  # p1[ka] = ... / del p1[ka] rings.py:2480
        # dict mutators reachable as bound methods:
        'update', 'pop', 'popitem', 'clear', 'setdefault',
    })

    def __init__(self, poly: PolyElement):
        self._poly = poly
        self._factors = None

    def __hash__(self):
        return hash(self._poly)  # reuse PolyElement.__hash__

    def __eq__(self, other):
        return self._poly == (other._poly if isinstance(other, CachedPoly) else other)

    def _invalidate(self):
        self._factors = None
        self._poly._hash = None

    def factor_list(self):
        if self._factors is None:
            self._factors = self._poly.factor_list()  # -> ring.dmp_factor_list
        return self._factors

    def __getattr__(self, name):
        attr = getattr(self._poly, name)
        if name in self._MUTATING_METHODS and callable(attr):
            def guarded(*args, **kwargs):
                self._invalidate()
                return attr(*args, **kwargs)

            return guarded
        return attr

    def __setitem__(self, k, v):
        self._invalidate()
        self._poly[k] = v

    def __delitem__(self, k):
        self._invalidate()
        del self._poly[k]

    def __iadd__(self, o):
        self._invalidate();
        self._poly += o;
        return self

    def __isub__(self, o):
        self._invalidate();
        self._poly -= o;
        return self

    def __imul__(self, o):
        self._invalidate();
        self._poly *= o;
        return self

    def __ipow__(self, o):
        self._invalidate();
        self._poly **= o;
        return self


class QuasiPolynomialElement(DomainElement, DefaultPrinting, CantSympify):
    """Element of univariate quasipolynomial ring."""

    __slots__ = (
        "ring",
        "_numer",
        "_denom",
        "_hash",
        "_degmin",
        "_degmax",
    )

    is_Poly = False
    is_QPoly = True

    def __init__(self, ring, numer, denom=None, normalize=True):
        self.ring = ring
        P = ring.domain  # Polynomial domain K[s]

        numer = {
            exp: P.convert(coeff)  # this will fail if a FractionElement is provided
            for exp, coeff in numer.items()
            if coeff  # this will filter zero coefficients
        }

        if denom is None:
            denom = P.one
        else:
            denom = P.convert(denom)

        if not denom:
            raise ZeroDivisionError("zero quasipolynomial denominator")
        if ring.symbols[0] in denom.ring.symbols:
            raise ValueError("Denominator must no contain any shift operators")

        if not numer:
            denom = P.one
        elif normalize:
            numer, denom = self.normalize_fraction(P, numer, denom)
        else:
            numer, denom = self.normalize_denominator(P, numer, denom)

        self._numer = numer
        self._denom = denom
        self._hash = None
        self._degmin = None
        self._degmax = None

    def new(self, numer, denom=None, normalize=True):
        """Create a new quasi polynomial element in the same ring."""
        return self.__class__(self.ring, numer, denom, normalize)

    def parent(self):
        return self.ring.to_domain()

    @staticmethod
    def is_polynomial(sym):
        return False

    @staticmethod
    def is_rational_function(sym):
        return False

    def is_proper(self):
        n_deg = max([max(n.degree, 0) for n in self._numer.values()])
        d_deg = max(self._denom.degree, 0)
        return d_deg >= n_deg

    def __hash__(self):
        # TODO
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring,
                                       frozenset(self._numer, self._denom)))
        return _hash

    def copy(self):
        """Return a copy of quasi polynomial self."""
        return self.new(self)

    def subs(self, x, a=None):
        # TODO
        raise NotImplementedError
        if isinstance(x, list) and a is None:
            f = self.copy()
            for X, a in x:
                f = f.subs(X, a)
            return f

        ring = self.ring
        a = ring.domain.convert(a)
        terms = {}
        for monom, coeff in self.iterterms():
            terms[monom.subs(x, a)] = coeff.subs(x, a)
        syms = [s.subs(x, a) for s in ring.symbols]

    def as_expr(self):
        den_expr = self._denom.as_expr()
        rat_coeffs = {
            exp: coeff.as_expr() / den_expr
            for exp, coeff in self._numer.items()
        }
        return self.dict_to_expr(rat_coeffs, *self.ring.symbols)

    def __bool__(f):
        return len(f._numer) > 0

    def __eq__(q1, q2):
        if not q2:
            return not q1

        q2 = q1.ring.convert(q2)
        if q1._numer == q2._numer and q1._denom == q2._denom:
            return True

        return False

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

    # def to_dict(self):
    #     return dict(self)

    @property
    def numer(self):
        """ Return the numerator as quasi polynomial. """
        return self.new(self._numer, self.ring.domain.one, normalize=False)

    @property
    def denom(self):
        """ Return the denominator polynomial. """
        return self._denom

    @property
    def is_monomial(self):
        return not self or len(self._numer) == 1

    @property
    def is_monic(self):
        return self.coeff_numer(self.degmax()) == self.ring.domain.one

    @property
    def is_primitive_monomial(self):
        return self.is_monomial and not self.is_monic

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    @classmethod
    def from_qpoly(cls, ring: Ring, expr):
        """ For backward compatibility """
        return cls.from_dict(ring, expr.as_dict())

    @classmethod
    def from_expr(cls, ring: Ring, expr: sp.Expr):
        return cls.from_dict(ring, cls.dict_from_expr(expr, ring))

    @classmethod
    def from_list(cls, ring: Ring, exp_coeff_list: list[tuple]):
        return cls.from_dict(ring, {k: v for k, v in exp_coeff_list})

    @classmethod
    def from_dict(cls, ring: Ring, exp_coeff_map: dict):
        P = ring.domain
        F = P.get_field()

        # convert everything to FractionElements
        rat_coeffs = {}
        for exp, coeff in exp_coeff_map.items():
            if not coeff :
                continue

            if not isinstance(coeff , DomainElement):
                coeff  = F.from_sympy(coeff)

            coeff  = F.convert(coeff )
            rat_coeffs[exp] = coeff
        return cls.from_fraction_coefficients(ring, rat_coeffs)

    @classmethod
    def from_fraction_coefficients(cls, ring: Ring, coeffs: dict):
        P = ring.domain

        if not coeffs:
            return ring.zero

        denom = P.one
        for frac in coeffs.values():
            denom = P.lcm(denom, frac.denom)

        numer = {}
        for exp, frac in coeffs.items():
            scale = P.exquo(denom, frac.denom)
            coeff = frac.numer * scale
            if coeff:
                numer[exp] = coeff

        return cls(ring, numer, denom, normalize=True)

    def evaluate(self, val):
        coeff_dom  = self.ring.domain
        var = coeff_dom.symbols[0]

        ev_denom = self._denom(val)
        ev_numer = coeff_dom.zero
        for exp, coeff in f._numer.items():
            ev_numer += sp.exp(exp * val) * coeff(val)

        if ev_denom == coeff_dom.zero:
            if ev_numer == coeff_dom.zero:
                new_numer = self.numer.diff(var)
                new_denom = ev_denom.diff(var)
                return self.new(new_numer, new_denom)(val)
            else:
                raise ValueError("Qpolys must not contain poles")
        else:
            return ev_numer / ev_denom

    def __call__(self, val):
        return self.evaluate(val)

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
        g = ring.convert(g)
        return f.div(g)

    def __rdivmod__(f, g):
        ring = f.ring
        g = ring.convert(g)
        return g.div(f)

    def __mod__(f, g):
        return f.div(g)[1]

    def __rmod__(f, g):
        return g.div(f)[1]

    def __floordiv__(f, g):
        return f.__divmod__(g)[0]

    def __rfloordiv__(f, g):
        return g.__rdivmod__(f)[0]

    def __truediv__(f, g):
        ring = f.ring
        g = ring.convert(g)
        return f.exquo(g)

    def __rtruediv__(f, g):
        ring = f.ring
        g = ring.convert(g)
        return g.exquo(f)

    def __pow__(self, n):
        """ raise quasipolynomial to power `n` """
        ring = self.ring
        if not n:
            if self:
                return ring.one
            else:
                raise ValueError("0**0")
        elif len(self._numer) == 1:
            monom, coeff = list(self._numer.items())[0]
            p = ring.zero
            p[monom * n] = coeff ** n
            return p
        else:
            raise NotImplementedError("No general exponentiation supported")

    def __str__(self):
        return str(
            f"QuasiPolynomialElement({self.exponents()}, domain={self.ring.domain})"
        )

    def add_ground(f, g):
        P = f.ring.domain
        if not P.is_element(g):
            raise TypeError("add_ground only allowed for elements from coefficient ring")

        numer = dict(f._numer)
        numer[0] = numer.get(0, P.zero) + P.convert(g)
        if not numer[0]:
            del numer[0]

        return f.new(numer, f._denom, normalize=False)

    def add(f, g):
        ring = f.ring
        g = ring.convert(g)

        if not f:
            return g
        if not g:
            return f

        P = ring.domain

        if f._denom == g._denom:
            # easy path
            numer = dict(f._numer)
            denom = f._denom
            g_scale = P.one
        else:
            denom = P.lcm(f._denom, g._denom)
            f_scale = P.exquo(denom, f._denom)
            g_scale = P.exquo(denom, g._denom)
            # scale f's part of new numerator
            numer = {
                exp: coeff * f_scale
                for exp, coeff in f._numer.items()
            }

        # add them together
        for exp, coeff in g._numer.items():
            coeff_new = numer.get(exp, P.zero) + coeff * g_scale
            if coeff_new:
                numer[exp] = coeff_new
            else:
                numer.pop(exp, None)

        # construct new qpoly
        return f.new(numer, denom, False)


    @staticmethod
    def _chop_coeff(cff):
        if cff.field.domain.is_Exact:
            return cff

        tol = 10 * cff.field.domain.dtype(2) ** -53  # ~1.1e-16, or just use 1e-16
        c_cff = cff.new(chop_poly(cff.numer, tol),
                        chop_poly(cff.denom, tol))
        return c_cff

    def neg(f):
        numer = {exp: -cff for exp, cff in f._numer.items()}
        return f.new(numer, f.denom, normalize=False)

    def sub(f, g):
        return f.add(-g)

    # def mul_ground(f, g):
    #     g = f.ring.domain(g)
    #     rd = {}
    #     for exp, cff in f.items():
    #         rd[exp] = cff * g
    #     # gens = f.gens + tuple(g.free_symbols)
    #     return QuasiPolynomialElement.from_dict(rd, f.ring)

    def mul(f, g):
        ring = f.ring
        g = ring.convert(g)

        if not f or not g:
            return ring.zero

        P = ring.domain
        numer = {}
        for f_exp, f_coeff in f._numer.items():
            for g_exp, g_coeff in g._numer.items():
                exp = f_exp + g_exp
                coeff = f_coeff * g_coeff

                coeff_new = numer.get(exp, P.zero) + coeff
                if coeff_new:
                    if coeff_new not in numer:
                        numer[exp] = coeff_new
                    else:
                        numer[exp] += coeff_new
                else:
                    numer.pop(exp, None)

        denom = f._denom * g._denom
        return f.new(numer, denom, True)

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
        return QuasiPolynomialElement.from_dict(f.ring, r_dict)

    # def div_ground(f, g):
    #     g = sp.sympify(g)
    #     rd = {}
    #     for exp, cff in f.items():
    #         rd[exp] = cff / g
    #     return QuasiPolynomialElement.from_dict(f.ring, rd)

    def _qdiv(f, g):
        """
        Quasipolynomial long division
        """
        if g.degree() > f.degree():
            raise ValueError("Ambiguous data: dividend has higher degree than divisor.")
        g_dmax = g.degmax()
        g_dmin = g.degmin()

        q = f.ring.zero
        r = f
        while True:
            r_dmax = r.degmax()
            r_dmin = r.degmin()

            max_done = r_dmax <= g_dmax
            min_done = r_dmin >= g_dmin

            if not max_done:
                mode = "leading"
            elif not min_done:
                mode = "trailing"
            else:
                break

            q_k = r.boundary_ratio(g, mode=mode)
            # TODO: missing correction factor

            r = r - q_k * g
            q = q + q_k

        return q, r

    def div(f, g):
        R = f.ring
        P = R.domain

        if P.of_type(g):
            return f.new(f.numer, f.denom * g)
        if not R.is_element(g):
            g = R.convert(g)

        if g.is_zero:
            raise ZeroDivisionError()
        if g.is_one:
            return f, R.zero
        if g.is_monomial:
            return f.quo_monom(g), R.zero

        return f._qdiv(g)

    def rdiv(f, g):
        fi = f ** (-1)
        g = f.ring(g)
        return g.mul(fi)

    def _monom_div(f, g):
        """ Divide the Quasipolynomial monoms """
        g_exp, g_coeff = list(g._numer.items())[0]
        g_den = g.denom

        denom = f.denom * g_coeff
        numer = {
            exp - g_exp: coeff * g_den
            for exp, coeff in f._numer.items()
        }
        return f.new(numer, denom, True)

    def quo_monom(f, g):
        if not g or len(g._numer) != 1:
            raise ZeroDivisionError()
        if g.is_one:
            return f
        return f._monom_div(g)

    def exquo(f, g):
        q, r = f.div(g)
        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, g)

    def _unify_coeffs(self, fc, gc) -> tuple[Any, Any]:
        new_dom = fc.field.to_domain().unify(gc.field.to_domain())
        fn = new_dom.convert_from(fc, fc.field.to_domain())
        gn = new_dom.convert_from(gc, gc.field.to_domain())
        return fn, gn

    def _eval_is_polynomial(self, gens):
        return False

    def as_numer_denom(self):
        """
        Return QuasiPolynomial with coefficient polynomials and their normalized common denominator
        """
        return self.numer, self.denom

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

        expr = sp.sympify(expr)
        if expr.is_Poly:
            if expr.gen == q_gen:
                return expr.as_dict()
            else:
                expr = expr.as_expr()

        w = sp.Wild("w")
        if not isinstance(q_gen, sp.Symbol):
            # fix issues with compound generators
            if isinstance(q_gen, sp.exp):
                r = q_gen.func(q_gen.exp * w)
                q_gen = sp.Dummy(positive=True)
                expr = expr.replace(r, q_gen ** w)
            else:
                print("Warning compound generators may cause problems")

        # convert to dict of fractions
        e_c_map = exponent_coeff_dict(expr, q_gen)
        return e_c_map

    @property
    def gen(self) -> sp.Symbol:
        return self.ring.symbols[0]

    def degree(self) -> sp.Expr:
        return self.degmax() - self.degmin()

    def degmin(self) -> sp.Expr:
        if self._degmin is None:
            self._degmin = self._smin(self.exponents(), default=sp.S.Infinity)
        return self._degmin

    def degmax(self) -> sp.Expr:
        if self._degmax is None:
            self._degmax = self._smax(self.exponents(), default=sp.S.Infinity)
        return self._degmax

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
        return self.new({dm: self.coeff_fraction(dm)})

    def TT(self):
        """ Return the trailing term """
        dm = self.degmin()
        return self.new({dm: self.coeff_fraction(dm)})

    def LM(self):
        """ Return the leading monom """
        return self.new({self.degmax(): 1})

    def TM(self):
        """Return the trailing monom"""
        return self.new({self.degmin(): 1})

    def LC(self) -> sp.Expr:
        """Return the Leading coefficient"""
        return self.coeff_fraction(self.degmax())

    def TC(self) -> sp.Expr:
        """Return the Trailing coefficient"""
        return self.coeff_fraction(self.degmin())

    def exponents(self) -> list[sp.Expr]:
        return list(self._numer.keys())

    def coefficients(self) -> list[tuple[sp.Expr, sp.Expr]]:
        return [self.coeff_pair(e) for e in self.exponents()]

    def coeff_numer(self, exp):
        return self._numer.get(exp, self.ring.domain.zero)

    def coeff_pair(self, exp):
        return self._numer.get(exp, self.ring.domain.zero), self._denom

    def coeff_fraction(self, exp):
        P = self.ring.domain
        return P.get_field().field.new(self._numer.get(exp, self.ring.domain.zero), self._denom)

    def iterterms(self):
        """Iterator over terms of a quasipolynomial. """
        return iter(zip(self.exponents(), self.coefficients()))

    # def diff(f, x):
    #     """Compute partial derivative w.r.t. ``x``.
    #
    #     This will take into account that we have q_gen = exp(c_gen)
    #
    #     """
    #     gd = {}
    #     for expv, coeff in f.iterterms():
    #         idx = coeff.field.symbols.index(x)
    #         c_dx = coeff.diff(coeff.field.gens[idx])
    #         gd[expv] = c_dx + coeff * expv
    #     return f.ring(gd)

    def boundary_ratio(f, g, *, mode: str):
        ring = f.ring
        g = ring.convert(g)

        if mode == "leading":
            f_deg = f.degmax()
            g_deg = g.degmax()
        elif mode == "trailing":
            f_deg = f.degmin()
            g_deg = g.degmin()
        else:
            raise ValueError("Invalid mode")

        f_num = f.coeff_numer(f_deg)
        g_num = g.coeff_numer(g_deg)

        if not g_num:
            raise ZeroDivisionError("zero pivot coefficient")

        quotient_num = f_num * g._denom
        quotient_den = g_num * f._denom

        return f.new({f_deg - g_deg: quotient_num}, quotient_den, True)

    @staticmethod
    def normalize_fraction(poly_ring, numer: Dict[sp.Expr, DomainElement], denom: DomainElement):
        """
        Perform a full normalisation
        """
        P = poly_ring

        if not numer:
            return {}, P.one

        gcd = denom
        for coeff in numer.values():
            gcd = P.gcd(gcd, coeff)
            if gcd == P.one:
                break

        if gcd != P.one:
            denom = P.exquo(denom, gcd)
            numer = {
                exp: P.exquo(coeff, gcd)
                for exp, coeff in numer.items()
            }

        return QuasiPolynomialElement.normalize_denominator(P, numer, denom)

    @staticmethod
    def normalize_denominator(poly_ring: Ring, numer: Dict[sp.Expr, DomainElement], denom: DomainElement):
        K = poly_ring.domain
        d_lc = denom.LC

        if d_lc == K.one:
            return numer, denom

        if K.is_Field:
            lc_inv = K.exquo(K.one, d_lc)
            numer = {exp: coeff.mul_ground(lc_inv) for exp, coeff in numer.items()}
            denom = denom.mul_ground(lc_inv)

        return numer, denom


class QuasiPolynomialRing(Ring):
    """Domain for quasipolynomials"""

    is_PolynomialRing = is_Poly = False
    is_QuasiPolynomialRing = is_QPoly = True

    is_Composite = True
    is_Field = False

    has_assoc_Ring = True
    has_assoc_Field = False

    dtype = QuasiPolynomialElement

    def __init__(self, coefficient_ring, shift_symbol, order=lex):
        self.domain = coefficient_ring
        self.dom = coefficient_ring.domain

        self.symbols = (shift_symbol,)
        self.order = sp.polys.orderings.monomial_key(order)

    def __str__(self):
        return str(self.domain) + "{" + ",".join(map(str, self.symbols)) + "}"

    @property
    def zero(self):
        return self.dtype(self, {})

    @property
    def one(self):
        return self.dtype(self, {0: self.domain.one})

    def characteristic(self):
        return self.domain.characteristic()

    def is_element(self, element):
        """True if ``element`` is an element of this ring. False otherwise."""
        return isinstance(element, self.dtype) and element.ring == self

    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        coeff = self.domain_new(coeff)
        return self.dtype(self, {0: coeff}, self.domain.one, False)

    def ring_new(self, element):
        P = self.domain
        K = P.domain
        if self.is_element(element):
            return element
        elif (
                isinstance(P, PolynomialRing) and P.of_type(element)
            or
                isinstance(P, PolyRing) and P.is_element(element)
            ):
            return self.ground_new(element)
        elif isinstance(element, K.dtype):
            return P.ground_new(element)
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return self.dtype.from_dict(self, element)
        elif isinstance(element, list):
            return self.dtype.from_list(self, element)
        elif isinstance(element, sp.Expr):
            return self.dtype.from_expr(self, element)
        else:
            return self.ground_new(element)

    __call__ = ring_new

    def from_sympy(self, a):
        return self.dtype.from_expr(self, a)

    def from_ZZ(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def to_domain(self):
        return QuasiPolynomialRing(self, self.symbols)

    def to_sympy(self, a):
        return a.as_expr()

    # def convert_from(self, element, base):
    #     if not base.is_Composite:
    #         return self(base.to_sympy(element))
    #     return self(base.to_sympy(element))


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

    def __bool__(f):
        return not f.is_zero

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
        c_denoms = [c.denom for c in self.coefficients()]
        if not c_denoms:
            return sp.Integer(0), sp.Integer(1)
        denom = reduce(c_denoms[0].ring.domain.lcm, c_denoms).as_expr()
        nd = {exp: cff.as_expr() * denom for exp, cff in self.as_dict().items()}
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
            expr += v.as_expr() * gen ** k
        return expr

    @staticmethod
    def list_to_expr(exp_coeff_pairs: list, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a list
        """
        expr = 0
        for k, v in exp_coeff_pairs:
            expr += v * gen ** k
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
                expr = expr.replace(r, q_gen ** w)
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


def chop_poly(p, tol):
    ring = p.ring
    d = {m: c for m, c in p.terms() if abs(c) >= tol}
    return ring.from_dict(d)


def hermite_interpolator(q: QuasiPolynomialElement, roots):
    """
    Construct the unique polynomial h with degree < D such that

        h^(j)(lambda_i) = f^(j)(lambda_i),
        j = 0, ..., multiplicity_i - 1.

    Parameters
    ----------
    q:
        Quasipolynomial with polynomial coefficients.
    roots:
        Dictionary {root: multiplicity}.

    Returns
    -------
    h:
        Hermite interpolation polynomial.
    """
    ring = q.ring
    coeff_ring = ring.domain
    var = coeff_ring.symbols[0]

    # create coefficients
    D = sum(roots.values())
    p_dict = sp.symbols(f"c0:{D}")

    h_trial = sum(p_dict[k] * var**k for k in range(D))

    equations = []
    q_derivs = [q.as_expr().subs(q.gen, sp.exp(var))]
    h_derivs = [h_trial]
    for root, multiplicity in roots.items():
        for j in range(multiplicity):
            if len(q_derivs) <= j:
                q_derivs.append(q_derivs[j-1].diff(var))
                h_derivs.append(h_derivs[j-1].diff(var))

            lhs = h_derivs[j].subs(var, root)
            rhs = q_derivs[j].subs(var, root)
            equations.append(lhs - rhs)

    solution = sp.solve(equations, p_dict, dict=True)
    if not solution:
        raise ValueError("The Hermite interpolation system has no solution.")

    h = coeff_ring(h_trial.subs(solution[0]))
    return h
