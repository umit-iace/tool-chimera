from typing import Union

import sympy as sp
from sympy.core.exprtools import decompose_power_rat


__all__ = ["GenPoly"]


class GenPoly:
    """
    Class to handle signomials, which are polynomials but with rational exponents
    """

    def __init__(self, expr: sp.Expr, gen: sp.Symbol, exp_coeff_map: dict = None):
        self._gen = gen
        if expr is not None:
            self._sig = self.dict_from_expr(expr, gen)
        else:
            if exp_coeff_map is not None:
                self._sig = exp_coeff_map
            else:
                raise ValueError("Either expr or map has to be provided")

    def __str__(self):
        return str(f"Signomial in {self._gen} with exponents {self.exponents()})")

    def __repr__(self):
        return str(
            f"Signomial with generator {self._gen} and exponent-coeffieicent map {self._sig}"
        )

    def as_dict(self) -> dict:
        return self._sig

    def as_numers_denom(self):
        """
        Return coeffiecent numerators as well as their common denominator
        """
        dens = [c.as_numer_denom()[1] for c in self.coefficients()]
        denom = sp.lcm(dens)
        numers = [(c * denom).simplify() for c in self.coefficients()]
        return numers, denom

    @classmethod
    def from_dict(cls, sd: dict, gen: sp.Symbol):
        return cls(None, exp_coeff_map=sd, gen=gen)

    @staticmethod
    def dict_to_expr(s_dict: dict, gen: sp.Symbol) -> sp.Expr:
        """
        Helper function to build an expr from an s_dict
        """
        expr = 0
        for k, v in s_dict.items():
            expr += v * gen**k
        return expr

    @staticmethod
    def dict_from_expr(
        expr: sp.Expr, gen: Union[sp.Symbol, sp.Expr]
    ) -> dict[sp.Expr : sp.Expr]:
        """
        Variant of polyutils '_parallel_dict_from_expr_if_gens' that will
        work with rational powers of generators in an expression and thus
        create powers and coefficients of a signomial.
        """
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
        expr = expr.expand()
        for term in sp.Add.make_args(expr):
            coeff = []
            exp = []
            term = term.simplify().expand(power_exp=False).collect(gen**w).collect(gen)
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
                # sig[exp] = (sig[exp] + c).simplify()
                sig[e] += c
            else:
                sig[e] = c

        return {e: r for e, c in sig.items() if (r := c.simplify()) != 0}

    @classmethod
    def from_expr(cls, expr, gen):
        return cls(expr=expr, gen=gen)

    def to_expr(self) -> sp.Expr:
        return self.dict_to_expr(self._sig, self._gen)

    @property
    def gen(self) -> sp.Symbol:
        return self._gen

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
        return self._sig[self.degmax(s_dict)]

    def TC(self, s_dict: dict = None) -> sp.Expr:
        """Return the Trailing coefficient"""
        return self._sig[self.degmin(s_dict)]

    def exponents(self) -> list[sp.Expr]:
        return list(self._sig.keys())

    def coefficients(self) -> list[sp.Expr]:
        return list(self._sig.values())

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
