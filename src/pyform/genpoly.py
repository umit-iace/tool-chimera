from typing import Union
from functools import reduce

import sympy as sp
from sympy.core.numbers import S
from sympy.core.exprtools import decompose_power_rat
from sympy.utilities.iterables import iterable


__all__ = ["GenPoly"]


class GenPoly(sp.Basic):
    """
    Class to handle signomials, which are polynomials but with rational exponents
    """

    is_commutative = True
    is_GenPoly = True

    def __new__(cls, rep, *gens):
        """Create a new generalized polynomial instance out of something useful."""
        if isinstance(rep, GenPoly):
            return cls.from_dict(rep.as_dict(), *gens)
        if isinstance(rep, dict):
            return cls.from_dict(rep, *gens)
        elif isinstance(rep, list):
            return cls.from_list(rep, *gens)
        elif isinstance(rep, sp.Add) and all(
            isinstance(arg, GenPoly) for arg in rep.args
        ):
            return reduce(lambda f, g: f.add(g), rep.args)
        else:
            rep = sp.sympify(rep, evaluate=type(rep) is not str)  # type: ignore
            return cls.from_expr(rep, *gens)

    @classmethod
    def from_gpoly(cls, expr, *gens: sp.Symbol):
        return cls.from_dict(expr.as_dict(), *gens)

    @classmethod
    def from_expr(cls, expr: sp.Expr, *gens: sp.Symbol):
        return cls.from_dict(cls.dict_from_expr(expr, *gens), *gens)

    @classmethod
    def from_list(cls, exp_coeff_list: list[tuple], *gens: sp.Symbol):
        return cls.from_dict({k: v for k, v in exp_coeff_list}, *gens)

    @classmethod
    def from_dict(cls, exp_coeff_map: dict, *gens: sp.Symbol):
        if not exp_coeff_map:
            return sp.sympify(0)
        return cls.new(exp_coeff_map, *gens)

    @classmethod
    def new(cls, rep, *gens):
        obj = sp.Basic.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        return obj

    @property
    def expr(self):
        return self.as_expr()

    @property
    def args(self):
        return (self.expr,) + self.gens

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
        if not isinstance(g, GenPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.div_ground(g)
        return f.as_expr() / g.as_expr()

    def __rtruediv__(f, g):
        if not isinstance(f, GenPoly):
            if g.gen not in f.as_expr().free_symbols:
                return f.div_ground(g)
        return g.as_expr() / f.as_expr()

    def __str__(self):
        return str(f"GenPoly({self.exponents()}, {self.gen})")

    def __repr__(self):
        return str(f"GenPoly({self.rep}, {self.gens})")

    def add_ground(f, g):
        rd = {}
        for exp, cff in f.as_dict().items():
            rc = cff + g
            if rc != 0:
                rd[exp] = rc
        return GenPoly.from_dict(rd, *f.gens)

    def add(f, g):
        g = sp.sympify(g)
        if not isinstance(g, GenPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.add_ground(g)
            else:
                g = GenPoly(g, *f.gens)

        rd = {}
        fd = f.as_dict()
        gd = g.as_dict()
        for exp, cff in fd.items():
            rc = cff + gd.pop(exp, 0)
            if rc != 0:
                rd[exp] = rc
        rd.update(gd)
        return GenPoly.from_dict(rd, f.gen)

    def neg(f):
        rd = {exp: -cff for exp, cff in f.as_dict().items()}
        return GenPoly.from_dict(rd, f.gen)

    def sub(f, g):
        return f.add(-g)

    # def sub_ground(f, g):
    #     return f.add_ground(-g)

    def mul_ground(f, g):
        g = sp.sympify(g)
        if not isinstance(g, GenPoly):
            g = GenPoly(g, f.gen)
        if g.exponents():
            return f.mul(g)

        g = g.as_expr()
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff * g

        return GenPoly.from_dict(rd, f.gen)

    def mul(f, g):
        g = sp.sympify(g)
        if g == S.Zero:
            return S.Zero
        if not isinstance(g, GenPoly):
            g = GenPoly(g, f.gen)

        r_dict = {}
        for fd, fc in f.as_dict().items():
            for gd, gc in g.as_dict().items():
                rd = (fd + gd).simplify()
                rc = fc * gc
                if rd in r_dict:
                    r_dict[rd] += rc
                else:
                    r_dict[rd] = rc
        return GenPoly.from_dict(r_dict, f.gen)

    def div_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff / g
        return GenPoly.from_dict(rd, f.gen)

    def as_dict(self) -> dict:
        return self.rep

    def as_numers_denom(self):
        """
        Return coefficient numerators as well as their common denominator
        """
        dens = [c.as_numer_denom()[1] for c in self.coefficients()]
        denom = sp.lcm(dens)
        numers = [(c * denom).simplify() for c in self.coefficients()]
        return numers, denom

    @staticmethod
    def dict_to_expr(exp_coeff_map: dict, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from a dict
        """
        expr = 0
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
    ) -> dict[sp.Expr : sp.Expr]:
        """
        Variant of polyutils '_parallel_dict_from_expr_if_gens' that will
        work with rational powers of generators in an expression and thus
        create powers and coefficients of a signomial.
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

        sig = {}
        expr = expr.expand(power_exp=False)
        for term in sp.Add.make_args(expr):
            coeff = []
            exp = []
            # term = term.simplify().expand(power_exp=False).collect(gen**w).collect(gen)
            term = term.expand(power_exp=False).collect(gen**w).collect(gen)
            for factor in sp.Mul.make_args(term):
                factor = factor.collect(gen**w).collect(gen)
                if not factor.has_free(gen):
                    coeff.append(factor)
                else:
                    p_base, p_exp = decompose_power_rat(factor)
                    if p_base != gen:
                        if isinstance(p_base, sp.Pow) and p_base.args[0] == gen:
                            # push symbol from the base into the exponent
                            p_exp *= p_base.args[1]
                        elif isinstance(p_base, sp.Mul) and p_base.args[0] == gen:
                            # power of one found, use remaining factors for coeff
                            p_exp *= 1
                            coeff.append(p_base.args[1])
                        else:
                            error = ValueError(
                                f"The computed base {p_base} does not match the generator {gen}"
                            )
                            raise error
                    if p_exp not in exp:
                        exp.append(p_exp)
            e = sp.Add(*exp)
            c = sp.Mul(*coeff)
            if e in sig:
                sig[e] += c
            else:
                sig[e] = c

        return {e: r for e, c in sig.items() if (r := c.simplify()) != 0}

    def as_expr(self) -> sp.Expr:
        return self.dict_to_expr(self.rep, self.gen)

    @property
    def gen(self) -> sp.Symbol:
        return self.gens[0]

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
