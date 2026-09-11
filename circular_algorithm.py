#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import sympy as sp

# ============================================================
# Theory notation
# ============================================================
#
# Fix n.
#
#   A  = QQ[t] / (t^n - 1)
#   mu = multiplication tensor of A
#
# Cyclotomic CRT:
#
#   t^n - 1 = product_{d | n} Phi_d(t)
#   A_d     = QQ[t] / Phi_d(t)
#   A       ~= product_{d | n} A_d
#
# For each d | n:
#
#   dim_QQ A_d        = phi(d)
#   rank_QQ(mu_d)     = 2 phi(d) - 1
#
# Choose rational points q_{d,s}, s = 0,...,2 phi(d)-2.
# Define
#
#   epsilon_{d,s}(X) = X(q_{d,s}) on A_d
#   l_{d,s}          = Lagrange output element in A_d
#
# Lift by CRT:
#
#   epsilon_tilde_{d,s} = epsilon_{d,s} o pi_d
#   l_tilde_{d,s}       = e_d l_{d,s} in A
#
# For cyclic matrix-vector multiplication we use
#
#   a(t)     = sum_i a_i t^i
#   x^vee(t) = sum_j x_j t^(-j)
#
# The printed scalar is
#
#   M_{d,s} = epsilon_tilde_{d,s}(a(t))
#             epsilon_tilde_{d,s}(x^vee(t))
#
# and the printed vector
#
#   l_{d,s}
#
# means the coefficient vector of the lifted element l_tilde_{d,s}.
#
# Final formula:
#
#   y = sum_{d | n} sum_s M_{d,s} l_{d,s}
#
# ============================================================

u = sp.Symbol("u")
t = sp.Symbol("t")
QQ = sp.QQ


# ============================================================
# Basic polynomial helpers over QQ
# ============================================================

def poly(expr: Any, var: sp.Symbol = t) -> sp.Poly:
    return sp.Poly(sp.expand(expr), var, domain=QQ)


def mod_poly(expr: Any, modulus: Any, var: sp.Symbol = t) -> sp.Expr:
    return poly(expr, var).rem(poly(modulus, var)).as_expr()


def exact_quotient(expr: Any, divisor: Any, var: sp.Symbol = t) -> sp.Expr:
    return poly(expr, var).exquo(poly(divisor, var)).as_expr()


def coeff_vector(expr: Any, length: int, var: sp.Symbol = t) -> list[sp.Rational]:
    p = poly(expr, var)
    return [sp.Rational(p.nth(i)) for i in range(length)]


def dot_expr(coeffs: Iterable[Any], symbols: Iterable[sp.Symbol]) -> sp.Expr:
    return sp.expand(sum(sp.Rational(c) * x for c, x in zip(coeffs, symbols)))


# ============================================================
# Construction helpers
# ============================================================

def q_points(phi_d: int, mode: Literal["standard", "symmetric"] = "standard") -> list[sp.Rational]:
    count = 2 * phi_d - 1
    if mode == "standard":
        return [sp.Rational(i) for i in range(count)]
    if mode == "symmetric":
        half = count // 2
        return [sp.Rational(i) for i in range(-half, half + 1)]
    raise ValueError(f"unknown point mode: {mode}")


def lagrange_output_polynomial(points: list[sp.Rational], s: int) -> sp.Expr:
    """Lagrange polynomial Lambda_{d,s}(u)."""
    qs = points[s]
    num = sp.Integer(1)
    den = sp.Integer(1)
    for r, qr in enumerate(points):
        if r == s:
            continue
        num *= u - qr
        den *= qs - qr
    return sp.expand(num / den)


def crt_block_data(F: sp.Expr, Phi_d: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """
    Return M_d, U_d, V_d, e_d where

        M_d = F / Phi_d,
        U_d M_d + V_d Phi_d = 1,
        e_d = U_d M_d mod F.
    """
    M_d = exact_quotient(F, Phi_d)
    U_d, V_d, h = sp.gcdex(M_d, Phi_d, t)
    h = sp.expand(h)
    if h != 1:
        U_d = sp.expand(U_d / h)
        V_d = sp.expand(V_d / h)
    e_d = mod_poly(U_d * M_d, F)
    return sp.expand(M_d), sp.expand(U_d), sp.expand(V_d), sp.expand(e_d)


def epsilon_tilde_coeffs(Phi_d: sp.Expr, q: sp.Rational, n: int) -> list[sp.Rational]:
    """Coefficient vector of epsilon_tilde_{d,s} on the basis 1,t,...,t^{n-1}."""
    return [sp.Rational(mod_poly(t**i, Phi_d).subs(t, q)) for i in range(n)]


def right_coeffs_for_x_vee(eps: list[sp.Rational]) -> list[sp.Rational]:
    """
    Convert epsilon_tilde(q(t)) into epsilon_tilde(x^vee(t)).

    If q(t)=sum_i q_i t^i and x^vee(t)=sum_j x_j t^{-j}, then q_i=x_{-i mod n}.
    """
    n = len(eps)
    return [eps[(-j) % n] for j in range(n)]


# ============================================================
# Data
# ============================================================

@dataclass(frozen=True)
class Term:
    d: int
    s: int
    q: sp.Rational

    A_coeffs: list[sp.Rational]
    X_coeffs: list[sp.Rational]
    l_coeffs: list[sp.Rational]

    # Detail-only data
    epsilon_ds: list[sp.Rational]
    Lambda_ds: sp.Expr
    l_ds: sp.Expr
    l_tilde: sp.Expr


@dataclass(frozen=True)
class Block:
    d: int
    Phi_d: sp.Expr
    phi_d: int
    rank_mu_d: int
    points: list[sp.Rational]
    terms: list[Term]

    # Detail-only data
    M_d: sp.Expr
    U_d: sp.Expr
    V_d: sp.Expr
    e_d: sp.Expr


@dataclass(frozen=True)
class Decomposition:
    n: int
    F: sp.Expr
    blocks: list[Block]

    @property
    def divisors(self) -> list[int]:
        return [b.d for b in self.blocks]

    @property
    def tau(self) -> int:
        return len(self.blocks)

    @property
    def rank_mu(self) -> int:
        return sum(b.rank_mu_d for b in self.blocks)

    @property
    def terms(self) -> list[Term]:
        return [term for block in self.blocks for term in block.terms]


# ============================================================
# Build decomposition
# ============================================================

def build_decomposition(
    n: int,
    *,
    point_mode: Literal["standard", "symmetric"] = "standard",
) -> Decomposition:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    F = t**n - 1
    blocks: list[Block] = []

    for d in sp.divisors(n):
        Phi_d = sp.cyclotomic_poly(d, t, polys=False)
        phi_d = int(sp.totient(d))
        rank_mu_d = 2 * phi_d - 1
        points = q_points(phi_d, point_mode)
        M_d, U_d, V_d, e_d = crt_block_data(F, Phi_d)

        terms: list[Term] = []
        for s, q in enumerate(points):
            epsilon_ds = [sp.Rational(q**i) for i in range(phi_d)]
            Lambda_ds = lagrange_output_polynomial(points, s)
            l_ds = mod_poly(Lambda_ds.subs(u, t), Phi_d)

            A_coeffs = epsilon_tilde_coeffs(Phi_d, q, n)
            X_coeffs = right_coeffs_for_x_vee(A_coeffs)

            l_tilde = mod_poly(e_d * l_ds, F)
            l_coeffs = coeff_vector(l_tilde, n)

            terms.append(
                Term(
                    d=int(d),
                    s=s,
                    q=q,
                    A_coeffs=A_coeffs,
                    X_coeffs=X_coeffs,
                    l_coeffs=l_coeffs,
                    epsilon_ds=epsilon_ds,
                    Lambda_ds=sp.expand(Lambda_ds),
                    l_ds=sp.expand(l_ds),
                    l_tilde=sp.expand(l_tilde),
                )
            )

        blocks.append(
            Block(
                d=int(d),
                Phi_d=sp.expand(Phi_d),
                phi_d=phi_d,
                rank_mu_d=rank_mu_d,
                points=points,
                terms=terms,
                M_d=sp.expand(M_d),
                U_d=sp.expand(U_d),
                V_d=sp.expand(V_d),
                e_d=sp.expand(e_d),
            )
        )

    return Decomposition(n=n, F=sp.expand(F), blocks=blocks)


# ============================================================
# Formatting
# ============================================================

def qstr(q: Any) -> str:
    return str(sp.Rational(q))


def vstr(v: list[Any]) -> str:
    return "[" + ", ".join(qstr(x) for x in v) + "]"


def estr(expr: Any) -> str:
    return sp.sstr(sp.factor(expr))


def line(title: str = "") -> None:
    print("-" * 78)
    if title:
        print(title)
        print("-" * 78)


def print_summary(dec: Decomposition) -> None:
    print(f"n = {dec.n}")
    print(f"divisors d | n     : {dec.divisors}")
    print(f"tau(n)             : {dec.tau}")
    print(f"rank_QQ(mu) terms  : 2n - tau(n) = {dec.rank_mu} = {dec.rank_mu}")
    print()
    print("block ranks:")
    for b in dec.blocks:
        print(f"  d = {b.d:<4}  phi(d) = {b.phi_d:<4}  rank_QQ(mu_{b.d}) = {b.rank_mu_d}")


def print_M_terms(dec: Decomposition) -> None:
    n = dec.n
    a = [sp.Symbol(f"a_{i}") for i in range(n)]
    x = [sp.Symbol(f"x_{i}") for i in range(n)]

    line("M terms")
    for term in dec.terms:
        left = dot_expr(term.A_coeffs, a)
        right = dot_expr(term.X_coeffs, x)
        print(f"M_{{{term.d},{term.s}}} = ({estr(left)}) * ({estr(right)})")


def print_l_terms(dec: Decomposition) -> None:
    line("l coefficients")
    for term in dec.terms:
        print(f"l_{{{term.d},{term.s}}} = {vstr(term.l_coeffs)}")


def print_formula(dec: Decomposition) -> None:
    line("Final formula")
    names = [f"M_{{{t.d},{t.s}}} l_{{{t.d},{t.s}}}" for t in dec.terms]
    print("y = " + " + ".join(names))
    print()


def print_default(dec: Decomposition) -> None:
    print_summary(dec)
    print()
    print_M_terms(dec)
    print()
    print_l_terms(dec)
    print()
    print_formula(dec)


def print_detail(dec: Decomposition) -> None:
    line("Detail")
    print(f"F(t) = {estr(dec.F)}")
    print()
    for b in dec.blocks:
        print(f"d = {b.d}")
        print(f"  Phi_{b.d}(t) = {estr(b.Phi_d)}")
        print(f"  M_{b.d}(t)   = {estr(b.M_d)}")
        print(f"  U_{b.d}(t)   = {estr(b.U_d)}")
        print(f"  V_{b.d}(t)   = {estr(b.V_d)}")
        print(f"  e_{b.d}(t)   = {estr(b.e_d)}")
        print(f"  q_{{{b.d},s}} = {vstr(b.points)}")
        for term in b.terms:
            print(f"    s = {term.s}, q = {term.q}")
            print(f"      epsilon_{{{term.d},{term.s}}} = {vstr(term.epsilon_ds)}")
            print(f"      Lambda_{{{term.d},{term.s}}}(u) = {estr(term.Lambda_ds)}")
            print(f"      l in A_{term.d} = {estr(term.l_ds)}")
            print(f"      lifted l in A = {estr(term.l_tilde)}")
        print()


# ============================================================
# JSON export
# ============================================================

def rational_json(q: Any) -> dict[str, int]:
    r = sp.Rational(q)
    return {"num": int(r.p), "den": int(r.q)}


def vector_json(v: list[Any]) -> list[dict[str, int]]:
    return [rational_json(x) for x in v]


def to_json_data(dec: Decomposition) -> dict[str, Any]:
    return {
        "n": dec.n,
        "rank_QQ_mu": dec.rank_mu,
        "divisors": dec.divisors,
        "tau_n": dec.tau,
        "blocks": [
            {
                "d": b.d,
                "phi_d": b.phi_d,
                "rank_QQ_mu_d": b.rank_mu_d,
                "Phi_d": str(b.Phi_d),
                "detail": {
                    "M_d": str(b.M_d),
                    "U_d": str(b.U_d),
                    "V_d": str(b.V_d),
                    "e_d": str(b.e_d),
                    "q_points": [str(q) for q in b.points],
                },
            }
            for b in dec.blocks
        ],
        "terms": [
            {
                "d": term.d,
                "s": term.s,
                "q_ds": str(term.q),
                "M": {
                    "A_coeffs": vector_json(term.A_coeffs),
                    "X_coeffs": vector_json(term.X_coeffs),
                },
                "l_coeffs": vector_json(term.l_coeffs),
                "detail": {
                    "epsilon_ds": vector_json(term.epsilon_ds),
                    "Lambda_ds": str(term.Lambda_ds),
                    "l_ds": str(term.l_ds),
                    "l_tilde": str(term.l_tilde),
                },
            }
            for term in dec.terms
        ],
        "formula": "y = sum_{d|n} sum_s M_{d,s} l_{d,s}",
    }


def write_json(dec: Decomposition, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_data(dec), f, ensure_ascii=False, indent=2)


# ============================================================
# Optional verification
# ============================================================

def verify(dec: Decomposition) -> bool:
    n = dec.n
    a = sp.symbols(f"a_0:{n}")
    x = sp.symbols(f"x_0:{n}")
    a_poly = sum(a[i] * t**i for i in range(n))
    x_vee = sum(x[j] * t**((-j) % n) for j in range(n))
    target = sp.rem(sp.expand(a_poly * x_vee), dec.F, t)

    reconstructed = 0
    for term in dec.terms:
        reconstructed += dot_expr(term.A_coeffs, a) * dot_expr(term.X_coeffs, x) * term.l_tilde

    diff = sp.rem(sp.expand(reconstructed - target), dec.F, t)
    return sp.expand(diff) == 0


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the QQ tensor decomposition for cyclic convolution via CRT."
    )
    parser.add_argument("n", type=int, help="cyclic length n")
    parser.add_argument(
        "--points",
        choices=["standard", "symmetric"],
        default="standard",
        help="q_{d,s} points: standard = 0,1,...; symmetric = centered integers",
    )
    parser.add_argument("--detail", action="store_true", help="print CRT and Lagrange construction details")
    parser.add_argument("--json", metavar="PATH", help="optional JSON output path")
    parser.add_argument("--verify", action="store_true", help="symbolically verify the formula")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dec = build_decomposition(args.n, point_mode=args.points)

    print_default(dec)

    if args.detail:
        print()
        print_detail(dec)

    if args.verify:
        print()
        line("Verification")
        print(verify(dec))

    if args.json:
        write_json(dec, args.json)
        print()
        line("JSON")
        print(args.json)


if __name__ == "__main__":
    main()
