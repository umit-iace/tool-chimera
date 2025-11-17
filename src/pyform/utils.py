#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import sympy as sp
from sympy import PolynomialError

from pynf.signomial import Signomial

# seed using the 'cat ran over the keyboard' method
rng = np.random.default_rng(56802470204575498132489694523498)


class Shift(sp.Function):

    @classmethod
    def eval(cls, x):
        return None

    # def inverse(self, argindex=1):
    #     return Shift

def get_coeffs(expr: sp.Expr, gen: sp.Symbol) -> dict:
    s = Signomial(expr=expr, gen=gen)
    c = s.as_dict()
    return c


def get_matrix_coeffs(mat: sp.Matrix, gen: sp.Symbol, exponents_only=False) ->list[list[dict]]:
    coeffs = []
    for j in range(mat.rows):
        r = []
        for i in range(mat.cols):
            c = get_coeffs(mat[j,i], gen)
            if exponents_only:
                r.append(c.keys())
            else:
                r.append(c)
        coeffs.append(r)
    return coeffs


def show_coeffs(expr: sp.Expr, gen: sp.Symbol):
    c = get_coeffs(expr, gen)
    for k,v in c.items():
        print(k, v)


def show_mat_coeffs(mat: sp.Matrix, gen: sp.Symbol):
    mc = get_matrix_coeffs(mat, gen)
    for i,row in enumerate(mc):
        for j,c in enumerate(row):
            print(f"Element: {j,i}")
            for k,v in c.items():
                print(k, v)


def generate_transform(dim, max_entry=4):
    found=False
    i=0
    while not found:
        T = rng.integers(low=-max_entry, high=max_entry, size=(dim, dim), endpoint=True)
        det=np.linalg.det(T)
        if np.abs(det) == 1:
            Tinv=np.linalg.inv(T)
            if np.max(np.abs(Tinv*det)) < max_entry + 1:
                found=True
        i+=1
    
    print(f"Integer Transformation found after {i} iterations")
    return T, Tinv


def get_leading_term_matrix(mat, sym, subs=None):
    if subs is not None:
        mat = mat.subs(subs).expand()
    
    degs = sp.MutableDenseMatrix.zeros(*mat.shape)
    coeffs = sp.MutableDenseMatrix.zeros(*mat.shape)
    for idx, e in mat.iter_items():
        try:
            # maybe it is a polynomial expression?
            p = sp.Poly(e, sym)
            d = p.degree()
            c = p.LC()
        except PolynomialError:
            try:
                # maybe it is a rational expression?
                _np, _dp = [sp.Poly(expr, sym) for expr in e.as_numer_denom()]
                d = _np.degree() - _dp.degree()
                c = _np.LC() / _dp.LC()
            except PolynomialError:
                try:
                    p = Signomial(expr=e, gen=sym)
                    d = p.degmax()
                    c = p.LC()
                except ValueError:
                    print(f"Given Expression:\n{e}\n is of unsupported structure in {sym} ")
        degs[idx] = d
        coeffs[idx] = c
        
    return degs, coeffs


def get_col_lc_mat(mat, sym, subs=None):
    degs, coeffs = get_leading_term_matrix(mat, sym, subs)
    col_degs = [max(degs[:,c]) for c in range(mat.shape[1])]

    r = sp.Matrix.zeros(*mat.shape)
    for c_idx in range(mat.shape[1]):
        for r_idx in range(mat.shape[0]):
            if degs[r_idx, c_idx] == col_degs[c_idx]:
                r[r_idx, c_idx] = coeffs[r_idx, c_idx]
    return r

def get_col_lt_mat(mat, sym, subs=None):
    degs, coeffs = get_leading_term_matrix(mat, sym, subs)
    col_degs = [max(degs[:,c]) for c in range(mat.shape[1])]

    r = sp.Matrix.zeros(*mat.shape)
    for c_idx in range(mat.shape[1]):
        for r_idx in range(mat.shape[0]):
            if degs[r_idx, c_idx] == col_degs[c_idx]:
                r[r_idx, c_idx] = coeffs[r_idx, c_idx] * sym**degs[r_idx,c_idx]
    return r
    # lcm = get_col_lc_mat(mat, sym, subs)
    # cdm = sp.Matrix.diag([sym**d for d in get_col_degrees(mat, sym, subs)])
    # return lcm @ cdm

def is_col_reduced(mat, sym, subs=None):
    lcm = get_col_lc_mat(mat, sym, subs)
    return lcm.rank() == min(mat.shape)

    



def to_col_reduced(mat: sp.Matrix, sym: sp.Symbol, debug: bool=False) -> tuple[sp.Matrix, sp.Matrix]:
    """
    Compute right transformation :math:`M U = S` such that S is colum reduced

    Returns: Tuple(S, U)
    """
    u = sp.eye(mat.shape[0])
    m = mat.copy()
    n = 0
    while not is_col_reduced(m, sym):
        # display(m)
        if debug:
            print(f"Iteration {n}")
            # display(m)
        # sort into decreasing col degress
        cds = get_col_degrees(m, sym)
        #display(cds)
        if sorted(cds, reverse=True) != cds:
            #print("Permutating")
            s_idxs = np.argsort(cds).tolist()[::-1]
            perm = sp.combinatorics.Permutation(s_idxs)
            p = sp.PermutationMatrix(perm).as_explicit().T 
            #display(p)
            m = m @ p
            u = u @ p
            #display(m)

        # select pivot column index k
        mh = get_col_lc_mat(m, sym)
        cds = get_col_degrees(m, sym)
        
        #display(mh)
        #display(cds)
        for k in range(m.shape[1]):
            #print(f"Testing k={k}")
            lc = False

            for q in range(k+1, m.shape[1]):
                # check if mh[k] can be expressed as linear combination 
                # of columns mh[k:]
                #print(f"Testing colums {k+1} to {q}")
                
                # compute basis coefficients
                cs = sp.symbols(f"c_{k+2}:{q-k+2}")
                #display(cs)
                # pad them for the first k columns
                csp = tuple(0 for n in range(k+1)) + cs
                expr = mh[:,k] - sp.Add(*[csp[i] * mh[:,i] 
                                       for i in range(k+1, q+1)])
                #display(expr)
                cfs = sp.solve(expr, cs)
                #display(cfs)
                if cfs:
                    lc = True
                    #print("Found a linear combination")
                    break

            if lc:
                break

        # eliminate kth column from mh
        r = sp.Matrix.zeros(*m.shape)
        for i in range(r.rows):
            for j in range(r.cols):
                if i == j:
                    r[i,j] = 1
                elif i > k and i <= q and j == k:
                    r[i,j] = -cfs[csp[i-k]] * sym**(cds[k] - cds[i])
                else:
                    r[i,j] = 0
        #display(r)
        m = m @ r
        m.expand()
        m.simplify()
        u = u @ r
        n = n+1

    return m, u


def check_s_cond(mat):
    return is_col_reduced(mat, s)


def check_epsilon_cond(mat):
    cds = get_col_degrees(mat, e)
    #print(cds)
    if sorted(cds) != cds:
        #print("Not sorted")
        return False
    if not set(cds) == set(tau_m):
        #print("Not unique")
        return False

    # Test structural identity
    lcms = get_col_lc_mat(mat, s)
    lcmse = get_col_lc_mat(lcms, e)
    lcme = get_col_lc_mat(mat, e)
    lcmes = get_col_lc_mat(lcme, s)
    if lcmse != lcmes:
        #print("Not reduced")
        return False
    
    return True


def check_delta_cond(mat):
    cds = get_col_degrees(mat, d)
    return set(cds) == set(tau_p)


def check_all_conds(mat):
    return check_s_cond(mat) and check_epsilon_cond(mat) and check_delta_cond(mat)


def check_equal(Expr1, Expr2, n=10, positive=False, strictly_positive=False):
    """ 
    Determine over what range to generate random numbers
    
    Stolen from Stackoverflow 
    """
    sample_min = -1
    sample_max = 1
    if positive:
        sample_min = 0
        sample_max = 1
    if strictly_positive:
        sample_min = 1
        sample_max = 2
    
    # Regroup all free symbols from both expressions
    free_symbols = set(Expr1.free_symbols) | set(Expr2.free_symbols)
    
    # Numeric (brute force) equality testing n-times
    for i in range(n):
        test_values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        s_dict = {s: v for s,v in zip(free_symbols, test_values)}
        res = (Expr1 - Expr2).evalf(subs=s_dict)
        if not np.isclose(res, 0):
            return False

    return True



def test_conditions():
    # not sorted
    m = sp.Matrix([[e**3, e**2], [1, 1]])
    assert check_epsilon_cond(m) == False
    # not unique
    m = sp.Matrix([[e, e**3], [1, 1]])
    assert check_epsilon_cond(m) == False
    m = sp.Matrix([[e**2, e**3], [e, 1]])
    assert check_epsilon_cond(m) == False
    # not reduced in s and e
    m = sp.Matrix([[e*s**2, s*e**2], [e, s**2]])
    assert check_epsilon_cond(m) == False
    # valid
    m = sp.Matrix([[e*s**2, s*e], [e*s, e**2*s**2]])
    assert check_epsilon_cond(m) == True
    
    # not unique
    m = sp.Matrix([[d**2, 5], [0, 1]])
    assert check_delta_cond(m) == False
    # valid
    m = sp.Matrix([[d**2, d], [d, 1]])
    assert check_delta_cond(m) == True
    

def test_col_funcs():
    # Tests
    if 1:
        t = sp.Matrix([[s**3+1, s**4-3*s**3+5*s**2+7, s**5+17], 
                       [0, s**2 + 1, 1], 
                       [s**2, s**2, s**2]])
        tr, u = to_col_reduced(t, s)
        #display(is_col_reduced(tr))
        #display(tr)
        #display(u.det())
    
    if 1:
        th = get_col_lc_mat(tr)
        td = sp.Matrix.diag([s**d for d in get_col_degrees(tr)])
        ts = tr - th @ td
        display(th)
    
    if 1:
        _m = sp.Matrix([[s**5, 2*s**3], [3*s**3, 4*s**2]])
        display(_m)
        display(get_col_degrees(_m, s))
        display(get_col_lc_mat(_m, s))
        display(is_col_reduced(_m, s))
        
    if 1:
        _m = sp.Matrix([[e*d**2*s**5/(s**2+1), 2*e*s**3], [3*e*s**3, 4*d**3*s**2]])
        display(_m)
        for subs in [True, False]:
            for sym in [s, e, d]:
                if sym == s and subs:
                    continue
                display(get_col_degrees(_m, sym, subs))
                display(get_col_lc_mat(_m, sym, subs))
                display(get_col_lt_mat(_m, sym, subs))
                display(is_col_reduced(_m, sym, subs))


def test_generate_transform():
    # tests
    Arnf=np.array([
        [0,1,0,0,0],
        [0,0,1,0,0],
        [-1,-2,-1,3,1],
        [0,0,0,0,1],
        [1,3,-3,1,-1]])
    Brnf=np.array([
        [0,0],
        [0,0],
        [1,1],
        [0,0],
        [0,1]])

    print("Starting with system in MIMO-CCF:")
    print(Arnf)
    print(Brnf)
    
    T, Tinv = generate_transform(Arnf.shape[0])
    print(T)

    A=Tinv@Arnf@T
    B=Tinv@Brnf
    print("Resulting 'original' system:")
    print(A)
    print(B)

    b1=B[:,0:1]
    b2=B[:,1:]
    S=np.round(np.hstack((B,A@B,A@A@B,A@A@A@B,A@A@A@A@B)).astype(np.int64)*1e5)/1e5
    Sn=np.round(np.hstack((b1,A@b1,A@A@b1,b2,A@b2))*1e5)/1e5

    Sinv=np.round(np.linalg.inv(Sn)*1e5)/1e5
    A=np.round(A,8)
    B=np.round(B,8)
    
    
    q11=Sinv[2,:]
    q12=Sinv[2,:]@A
    q13=Sinv[2,:]@A@A
    q21=Sinv[4,:]
    q22=Sinv[4,:]@A
    Q=np.vstack((q11,q12,q13,q21,q22))



if __name__ == "__main__":
    test_col_funcs()
    test_generate_transform()

