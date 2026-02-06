import warnings
from functools import reduce
from typing import Union, List

import sympy as sp
from sympy import ring
from sympy.core.exprtools import decompose_power_rat
from sympy.core.numbers import S
from sympy.polys import ring, field
from sympy.polys.domains import EX, QQ
from sympy.polys.fields import FracElement
from sympy.polys.rings import PolyElement

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
        if isinstance(rep, list):
            return cls.from_list(rep, *gens)
        # elif isinstance(rep, sp.Add):
        #     # gp = next((o for o in rep.args if isinstance(rep, GenPoly)), None)
        #     # if gp is not None:
        #     if any(isinstance(arg, GenPoly) for arg in rep.args):
        #         return cls(reduce(lambda f, g: f + g, rep.args), *gens)

        # last resort
        rep = sp.sympify(rep, evaluate=type(rep) is not str)
        return cls.from_expr(rep, *gens)

    @classmethod
    def from_gpoly(cls, expr, *gens: List[sp.Symbol]):
        return cls.from_dict(expr.as_dict(), *gens)

    @classmethod
    def from_expr(cls, expr: sp.Expr, *gens: List[sp.Symbol]):
        return cls.from_dict(cls.dict_from_expr(expr, *gens), *gens)

    @classmethod
    def from_list(cls, exp_coeff_list: list[tuple], *gens: List[sp.Symbol]):
        return cls.from_dict({k: v for k, v in exp_coeff_list}, *gens)

    @classmethod
    def from_dict(cls, exp_coeff_map: dict, *gens: List[sp.Symbol]):
        if not exp_coeff_map:
            return S.Zero
        red_map = {deg: cff for deg, cff in exp_coeff_map.items() if cff != 0}
        return cls.new(red_map, *gens)

    @classmethod
    def new(cls, rep, *gens):
        assert isinstance(rep, dict)
        assert all([isinstance(g, (sp.Symbol, sp.Expr)) or g is None for g in gens])
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

    # def _hashable_content(self):
    #     return tuple(self.rep.items()) + self.gens

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
        return str(f"GenPoly({self.exponents()}, {self.gen})")

    def __repr__(self):
        return str(f"GenPoly({self.as_dict()}, {self.gen})")

    def add_ground(f, g):
        if g == S.Zero:
            return GenPoly(f, *f.gens)
        rd = f.as_dict()
        c0 = rd.get(0, S.Zero)
        rd[0] = c0 + c0.ring.from_expr(g)
        return GenPoly.from_dict(rd, *f.gens)

    def add(f, g):
        g = sp.sympify(g)
        if not f.rep:
            return g
        if not isinstance(g, GenPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.add_ground(g)
            else:
                g = GenPoly(g, *f.gens)

        # extract exponents
        fe = set(f.exponents())
        ge = set(g.exponents())
        common_exps = fe.intersection(ge)
        f_only_exps = fe.difference(common_exps)
        g_only_exps = ge.difference(common_exps)

        # join coefficients
        fd = f.as_dict()
        gd = g.as_dict()
        rd = {exp: fd[exp] + gd[exp] for exp in common_exps}
        rd.update({exp: fd[exp] for exp in f_only_exps})
        rd.update({exp: gd[exp] for exp in g_only_exps})

        return GenPoly.from_dict(rd, f.gen)

    def neg(f):
        rd = {exp: -cff for exp, cff in f.as_dict().items()}
        return GenPoly.from_dict(rd, f.gen)

    def sub(f, g):
        return f.add(-g)

    def mul_ground(f, g):
        # g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff * g

        return GenPoly.from_dict(rd, f.gen)

    def mul(f, g):
        g = sp.sympify(g)
        if g == S.Zero:
            return GenPoly(0, *f.gens)
        if not isinstance(g, GenPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.mul_ground(g)
            g = GenPoly(g, *f.gens)

        r_dict = {}
        for fd, fc in f.as_dict().items():
            for gd, gc in g.as_dict().items():
                rd = fd + gd
                rc = fc * gc
                if rd in r_dict:
                    r_dict[rd] += rc
                else:
                    r_dict[rd] = rc
        return GenPoly.from_dict(r_dict, *f.gens)

    def div_ground(f, g):
        g = sp.sympify(g)
        rd = {}
        for exp, cff in f.as_dict().items():
            rd[exp] = cff / g
        return GenPoly.from_dict(rd, *f.gens)

    def div(f, g):
        g = sp.sympify(g)
        if not isinstance(g, GenPoly):
            if f.gen not in g.as_expr().free_symbols:
                return f.div_ground(g)

        # fallback
        return f.as_expr() / g.as_expr()

    def as_dict(self) -> dict:
        return self.rep

    # def _eval_is_polynomial(self, gens):
    #     return False

    # def as_coeff_Mul(self, *deps, **kwargs):
    #     return self.as_expr().as_coeff_Mul(*deps, **kwargs)

    # def as_base_exp(self):
    #     return self.as_expr().as_base_exp()

    def as_numer_denom(self):
        """
        Return GenPoly with coefficient numerators as well as their common denominator
        """
        c_denoms = []
        for c in self.coefficients():
            if isinstance(c, PolyElement):
                c_denoms.append(c.ring.one)
            elif isinstance(c, FracElement):
                c_denoms.append(c.denom)

        denom = reduce(lambda x, y: x.lcm(y), c_denoms)
        numer = self.mul_ground(denom)
        return numer, denom

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
            expr += v.as_expr() * gen**k
        return expr

    @staticmethod
    def dict_from_expr(expr: sp.Expr, *gens: List[sp.Symbol]) -> dict[sp.Expr, sp.Expr]:
        """
        Variant of polyutils '_parallel_dict_from_expr_if_gens' that will
        work with rational powers of generators in an expression and thus
        create powers and coefficients of a generalized polynomial.
        """
        gp_gen, cff_gen = gens
        w = sp.Wild("w")

        # try some shortcuts
        if expr.is_Poly:
            if expr.gen == gp_gen:
                return expr.as_dict()
            else:
                expr = expr.as_expr()
        else:
            expr = sp.sympify(expr)

        # fix issues with compound generators
        if not isinstance(gp_gen, sp.Symbol):
            warnings.warn("Compound generators can cause problems")
            if isinstance(gp_gen, sp.exp):
                r = gp_gen.func(gp_gen.exp * w)
                gp_gen = sp.Dummy(positive=True)
                expr = expr.replace(r, gp_gen**w)

        e_c_map = {}
        expr = expr.expand(power_exp=False)
        for term in sp.Add.make_args(expr):
            coeff = []
            exp = []
            term = term.expand(power_exp=False).collect(gp_gen**w).collect(gp_gen)
            for factor in sp.Mul.make_args(term):
                factor = factor.collect(gp_gen**w).collect(gp_gen)
                for sub_factor in sp.Mul.make_args(factor):
                    if not sub_factor.has_free(gp_gen):
                        coeff.append(sub_factor)
                    else:
                        p_base, p_exp = sub_factor.as_base_exp()
                        if p_base != gp_gen:
                            if isinstance(p_base, sp.Pow) and p_base.args[0] == gp_gen:
                                # push symbol from the base into the exponent
                                p_exp *= p_base.args[1]
                            elif (
                                isinstance(p_base, sp.Mul) and p_base.args[0] == gp_gen
                            ):
                                # power of one found, use remaining factors for coeff
                                p_exp *= 1
                                coeff.append(p_base.args[1])
                            else:
                                error = ValueError(
                                    f"The computed base {p_base} does not match the generator {gp_gen}"
                                )
                                raise error
                        if p_exp not in exp:
                            exp.append(p_exp)
            e = sp.Add(*exp)
            c = sp.Mul(*coeff)
            assert gp_gen not in c.free_symbols
            if e in e_c_map:
                e_c_map[e] += c
            else:
                e_c_map[e] = c

        if cff_gen is not None:
            # treat coefficients as polynomials or rationals in s
            Fc, fc_map = construct_fraction_field(e_c_map.values(), cff_gen)
            red_map = {}
            for e, c in e_c_map.items():
                if c.is_polynomial(cff_gen):
                    cp = Fc.ring.from_expr(c)
                else:
                    cp = Fc.from_expr(c)
                red_map[e] = cp
            return red_map
        else:
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
        """Return the Leading monom"""
        return GenPoly.from_list([(self.degmax(), self.rep[self.degmax()])], *self.gens)

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


def construct_fraction_field(
    exprs,
    gen=None,
    extra_params=None,
    treat_functions_as_params=True,
    normalize_numeric=True,
):
    """
    Lift PolyElements over EX[x] to a polynomial ring over a fraction field K = QQ(params),
    returning (Rk, xk, polys_k, placeholder_map).

    gen: the generator symbol used to build K[x]
    extra_params: optional iterable of Symbols to force into the parameter field
    treat_functions_as_params: if True, replace function atoms (e.g., exp(e)) by fresh Symbols
    normalize_numeric: if True, nsimplify numeric coefficients to rationals

    Returns:
      Rk: polynomial ring K[x], where K = QQ(params) (as a fraction field)
      placeholder_map: dict mapping placeholder Symbols back to their original expressions
                       (useful to restore exp(e), etc., after computations)
    """
    # 1) Collect symbols from coefficients (exclude gen)
    #    We assume inputs are polynomials in gen; coefficients should not depend on x_symbol.
    params = set()
    for e in exprs:
        params |= {s for s in e.free_symbols if s != gen}

    # 2) Optionally add extra params
    if extra_params:
        params |= set(extra_params)

    # 3) Replace function atoms in coefficients by stand-in Symbols
    #    e.g., exp(e) -> Symbol('exp_e'). Keep a map to restore later.
    placeholder_map = {}

    def make_placeholder(fun_expr):
        # Build a readable name like 'exp_e' or 'sin_a_b'
        name = (
            fun_expr.func.__name__ + "_" + "_".join(str(a) for a in fun_expr.args)
            if fun_expr.args
            else fun_expr.func.__name__
        )
        sym = sp.Symbol(name)
        return sym

    replacements = {}
    if treat_functions_as_params:
        # Gather function atoms that appear outside x
        fun_atoms = set()
        for e in exprs:
            fun_atoms |= e.atoms(sp.Function)
        # Only keep those that do not depend on x (they must be coefficients)
        fun_atoms = {F for F in fun_atoms if gen not in F.free_symbols}
        for F in fun_atoms:
            ph = make_placeholder(F)
            replacements[F] = ph
            placeholder_map[ph] = F

    # 4) Apply replacements and optionally rationalize numerics
    exprs_param = []
    for e in exprs:
        ep = e.xreplace(replacements)
        if normalize_numeric:
            # nsimplify tries to turn Floats into rationals while preserving symbols
            try:
                ep = sp.nsimplify(ep, rational=True)
            except Exception:
                pass
        exprs_param.append(ep)

    # 5) Build parameter ring Rp = QQ[params, placeholders], then K = FracField(Rp)
    #    Note: include both plain Symbols (params) and placeholder symbols.
    all_param_syms = sorted(params | set(placeholder_map.keys()), key=lambda s: s.name)
    Rp, *gens = ring(", ".join(sym.name for sym in all_param_syms), sp.QQ)
    K = Rp.to_field()  # QQ(params, placeholders)

    # 6) Build the outer polynomial ring Rk = K[x]
    if gen is None:
        Rk, = ring('', K)
    else:
        Rk, xk = ring(gen.name, K)
    Fk = Rk.to_field()

    return Fk, placeholder_map
