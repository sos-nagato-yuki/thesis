from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

P = 5

F5 = Tuple[int, ...]
CoordMatrix = List[List[int]]
Matrix = Tuple[Tuple["OmegaPoly", ...], ...]

SearchMode = Literal["complete", "projective"]
OmegaRelation = Literal["formal", "root", "primitive"]

OMEGA_ORDER: Optional[int] = 3
OMEGA_RELATION: OmegaRelation = "root"


def set_omega(order: Optional[int], relation: OmegaRelation = "root") -> None:
    global OMEGA_ORDER, OMEGA_RELATION

    if relation not in ("formal", "root", "primitive"):
        raise ValueError("relation must be 'formal', 'root', or 'primitive'")

    if relation in ("root", "primitive"):
        if order is None or order <= 0:
            raise ValueError("root/primitive relation needs a positive omega order")

    OMEGA_ORDER = order
    OMEGA_RELATION = relation


def inv5(a: int) -> int:
    a %= P
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in F_5")
    return pow(a, -1, P)


def f5_signed(a: int) -> int:
    a %= P
    return a if a <= 2 else a - 5


def f5_str(a: int) -> str:
    return str(f5_signed(a))


def normalize_exp(exp: int) -> int:
    if OMEGA_RELATION == "formal" or OMEGA_ORDER is None:
        return exp
    return exp % OMEGA_ORDER


@dataclass(frozen=True)
class OmegaPoly:
    coeffs: Tuple[Tuple[int, int], ...] = ()

    @staticmethod
    def zero() -> "OmegaPoly":
        return OmegaPoly(())

    @staticmethod
    def one() -> "OmegaPoly":
        return OmegaPoly.monomial(0, 1)

    @staticmethod
    def monomial(exp: int, coeff: int = 1) -> "OmegaPoly":
        return OmegaPoly.from_dict({exp: coeff})

    @staticmethod
    def from_dict(raw: Dict[int, int]) -> "OmegaPoly":
        d: Dict[int, int] = {}

        for e, c in raw.items():
            c %= P
            if c == 0:
                continue

            ee = normalize_exp(e)
            d[ee] = (d.get(ee, 0) + c) % P
            if d[ee] == 0:
                del d[ee]

        if OMEGA_RELATION == "primitive":
            assert OMEGA_ORDER is not None
            top = OMEGA_ORDER - 1
            top_coeff = d.pop(top, 0) % P

            if top_coeff:
                for e in range(top):
                    d[e] = (d.get(e, 0) - top_coeff) % P
                    if d[e] == 0:
                        del d[e]

        return OmegaPoly(
            tuple(sorted((e, c % P) for e, c in d.items() if c % P != 0))
        )

    def to_dict(self) -> Dict[int, int]:
        return dict(self.coeffs)

    def __add__(self, other: "OmegaPoly") -> "OmegaPoly":
        d = self.to_dict()
        for e, c in other.coeffs:
            d[e] = (d.get(e, 0) + c) % P
            if d[e] == 0:
                del d[e]
        return OmegaPoly.from_dict(d)

    def __neg__(self) -> "OmegaPoly":
        return OmegaPoly.from_dict({e: -c for e, c in self.coeffs})

    def __sub__(self, other: "OmegaPoly") -> "OmegaPoly":
        return self + (-other)

    def scale(self, a: int) -> "OmegaPoly":
        a %= P
        if a == 0:
            return OmegaPoly.zero()
        return OmegaPoly.from_dict({e: a * c for e, c in self.coeffs})

    def is_zero(self) -> bool:
        return not self.coeffs

    def is_one(self) -> bool:
        return self.coeffs == ((0, 1),)

    def __str__(self) -> str:
        if not self.coeffs:
            return "0"

        pieces: List[str] = []

        for e, c in self.coeffs:
            cs = f5_signed(c)

            if e == 0:
                body = str(abs(cs))
            elif abs(cs) == 1:
                body = f"ω^{e}"
            else:
                body = f"{abs(cs)}ω^{e}"

            if not pieces:
                pieces.append(body if cs > 0 else f"-{body}")
            else:
                pieces.append(("+ " if cs > 0 else "- ") + body)

        return " ".join(pieces)


def make_matrix_from_exponents(
    exponents: Sequence[Sequence[int]],
    *,
    omega_order: Optional[int] = None,
    relation: Optional[OmegaRelation] = None,
) -> Matrix:
    if omega_order is not None or relation is not None:
        set_omega(
            OMEGA_ORDER if omega_order is None else omega_order,
            OMEGA_RELATION if relation is None else relation,
        )

    return tuple(tuple(OmegaPoly.monomial(e) for e in row) for row in exponents)


def matrix_shape(X: Matrix) -> Tuple[int, int]:
    if not X:
        return 0, 0
    return len(X), len(X[0])


def serialize_matrix(X: Matrix) -> Tuple[Tuple[Tuple[Tuple[int, int], ...], ...], ...]:
    return tuple(tuple(cell.coeffs for cell in row) for row in X)


def deserialize_matrix(
    data: Tuple[Tuple[Tuple[Tuple[int, int], ...], ...]]
) -> Matrix:
    return tuple(tuple(OmegaPoly.from_dict(dict(cell)) for cell in row) for row in data)


def exponent_index_for_matrix(X: Matrix) -> Dict[int, int]:
    if OMEGA_RELATION == "root":
        assert OMEGA_ORDER is not None
        return {e: e for e in range(OMEGA_ORDER)}

    if OMEGA_RELATION == "primitive":
        assert OMEGA_ORDER is not None
        return {e: e for e in range(OMEGA_ORDER - 1)}

    exps = sorted({e for row in X for cell in row for e, _ in cell.coeffs})
    return {e: i for i, e in enumerate(exps)}


def flatten_row(row: Sequence[OmegaPoly], exponent_index: Dict[int, int]) -> F5:
    dim = len(exponent_index)
    out = [0] * (len(row) * dim)

    for j, poly in enumerate(row):
        base = j * dim
        for e, c in poly.coeffs:
            out[base + exponent_index[e]] = (
                out[base + exponent_index[e]] + c
            ) % P

    return tuple(out)


def rref_basis_with_coordinates(
    vectors: Sequence[F5],
) -> Tuple[List[int], CoordMatrix, int]:
    if not vectors:
        return [], [], 0

    width = len(vectors[0])

    echelon_rows: List[List[int]] = []
    echelon_as_basis: List[List[int]] = []
    pivots: List[int] = []

    basis_indices: List[int] = []
    all_coords: CoordMatrix = []

    for row_idx, vec in enumerate(vectors):
        v = list(vec)
        combo = [0] * len(echelon_rows)

        for b, pivot in enumerate(pivots):
            factor = v[pivot] % P
            if factor:
                for k in range(width):
                    v[k] = (v[k] - factor * echelon_rows[b][k]) % P

                for a, t in enumerate(echelon_as_basis[b]):
                    combo[a] = (combo[a] + factor * t) % P

        if all(x % P == 0 for x in v):
            all_coords.append(combo)
            continue

        pivot = next(i for i, x in enumerate(v) if x % P != 0)
        inv = inv5(v[pivot])

        new_echelon = [(x * inv) % P for x in v]

        new_echelon_coord = [(-inv * c) % P for c in combo] + [inv]

        for old in all_coords:
            old.append(0)

        for old in echelon_as_basis:
            old.append(0)

        echelon_rows.append(new_echelon)
        echelon_as_basis.append(new_echelon_coord)
        pivots.append(pivot)

        basis_indices.append(row_idx)

        all_coords.append([0] * (len(basis_indices) - 1) + [1])

    return basis_indices, all_coords, len(basis_indices)


def bs5(X: Matrix) -> Tuple[List[int], CoordMatrix]:
    m, _ = matrix_shape(X)

    if m == 0:
        return [], []

    exp_index = exponent_index_for_matrix(X)
    vectors = [flatten_row(row, exp_index) for row in X]

    J, C, _ = rref_basis_with_coordinates(vectors)
    return J, C


def saturation(X: Matrix) -> Tuple[List[int], CoordMatrix, Matrix]:
    J, C = bs5(X)
    X_sat = tuple(X[j] for j in J)
    return J, C, X_sat


def nonzero_columns(X: Matrix) -> List[int]:
    m, n = matrix_shape(X)
    return [
        j
        for j in range(n)
        if any(not X[i][j].is_zero() for i in range(m))
    ]


def allowed_specialization_vectors(
    n: int,
    active_cols: Sequence[int],
    *,
    mode: SearchMode = "complete",
) -> Iterable[Tuple[int, ...]]:
    active = sorted(set(active_cols))
    pivot_values = (1, P - 1) if mode == "complete" else (1,)

    for k in active:
        suffix = [j for j in active if j > k]

        for pivot in pivot_values:
            for values in product(range(P), repeat=len(suffix)):
                u = [0] * n
                u[k] = pivot

                for j, value in zip(suffix, values):
                    u[j] = value

                yield tuple(u)


def linear_form_coefficients(u: Sequence[int]) -> Tuple[int, ...]:
    u = tuple(x % P for x in u)

    k = next(i for i, x in enumerate(u) if x % P != 0)
    inv = inv5(u[k])

    out = [0] * len(u)
    out[k] = 1

    for j in range(k + 1, len(u)):
        out[j] = (u[j] * inv) % P

    return tuple(out)


def specialize(
    X: Matrix, u: Sequence[int]
) -> Tuple[int, int, Matrix, List[int], CoordMatrix, Matrix, Tuple[int, ...]]:
    m, n = matrix_shape(X)
    u = tuple(x % P for x in u)

    if len(u) != n:
        raise ValueError(f"u has length {len(u)}, expected {n}")

    if all(x == 0 for x in u):
        raise ValueError("u must be nonzero")

    k = next(i for i, x in enumerate(u) if x != 0)
    inv = inv5(u[k])

    L = linear_form_coefficients(u)

    eliminated_col: Matrix = tuple((X[i][k],) for i in range(m))

    J_col, C_col, col_sat = saturation(eliminated_col)
    rank_weight = len(J_col)

    rows: List[Tuple[OmegaPoly, ...]] = []

    for i in range(m):
        new_row: List[OmegaPoly] = []

        for j in range(n):
            if j == k:
                new_row.append(OmegaPoly.zero())

            elif j > k:
                factor = (-u[j] * inv) % P
                new_row.append(X[i][j] + X[i][k].scale(factor))

            else:
                new_row.append(X[i][j])

        rows.append(tuple(new_row))

    return k, rank_weight, tuple(rows), J_col, C_col, col_sat, L


def primitive_terminal_matrix(X: Matrix) -> Matrix:
    if OMEGA_ORDER is None:
        return X

    old_relation = OMEGA_RELATION

    try:
        set_omega(OMEGA_ORDER, "primitive")
        return tuple(
            tuple(OmegaPoly.from_dict(cell.to_dict()) for cell in row)
            for row in X
        )
    finally:
        set_omega(OMEGA_ORDER, old_relation)


def terminal_saturation(
    X: Matrix,
    terminal_relation: OmegaRelation,
) -> Tuple[List[int], CoordMatrix, Matrix, Matrix]:
    if terminal_relation == "primitive":
        X_term = primitive_terminal_matrix(X)

        old_relation = OMEGA_RELATION

        try:
            set_omega(OMEGA_ORDER, "primitive")
            J, C, X_sat = saturation(X_term)
        finally:
            set_omega(OMEGA_ORDER, old_relation)

        return J, C, X_sat, X_term

    J, C, X_sat = saturation(X)
    return J, C, X_sat, X


def scalar_mul_cost(coeff: OmegaPoly, *, count_unit_multiplications: bool) -> int:
    if coeff.is_zero():
        return 0

    if not count_unit_multiplications and coeff.is_one():
        return 0

    return 1


def column_charged_cost(
    col_sat: Matrix,
    *,
    count_unit_multiplications: bool,
) -> int:
    return sum(
        scalar_mul_cost(row[0], count_unit_multiplications=count_unit_multiplications)
        for row in col_sat
    )


def final_charged_cost(
    final_matrix: Matrix,
    final_active_column: Optional[int],
    *,
    count_unit_multiplications: bool,
) -> int:
    if final_active_column is None:
        return 0

    return sum(
        scalar_mul_cost(
            row[final_active_column],
            count_unit_multiplications=count_unit_multiplications,
        )
        for row in final_matrix
    )


@dataclass
class SearchStep:
    u: Tuple[int, ...]
    eliminated_col: int
    linear_form_coeffs: Tuple[int, ...]

    rank_weight: int
    charged_weight: int

    sp_basis_rows: List[int]
    sp_coordinates: CoordMatrix
    sp_saturated_column: Matrix

    x_u_matrix: Matrix

    sat_basis_rows: Optional[List[int]] = None
    sat_coordinates: Optional[CoordMatrix] = None
    sat_matrix: Optional[Matrix] = None


@dataclass
class SearchResult:
    total_weight: int
    total_rank_weight: int

    steps: List[SearchStep]

    final_weight: int
    final_rank_weight: int

    final_matrix: Matrix
    final_raw_matrix: Matrix
    final_basis_rows: List[int]
    final_coordinates: CoordMatrix
    final_active_column: Optional[int]

    visited_states: int = 0
    tried_vectors: int = 0

    search_mode: SearchMode = "complete"
    count_unit_multiplications: bool = False


def exhaustive_search(
    X: Matrix,
    *,
    do_saturation_each_step: bool = True,
    max_nodes: Optional[int] = None,
    search_mode: SearchMode = "complete",
    terminal_relation: OmegaRelation = "primitive",
    count_unit_multiplications: bool = False,
) -> SearchResult:
    visited_states = 0
    tried_vectors = 0

    @lru_cache(maxsize=None)
    def solve(
        serialized: Tuple[Tuple[Tuple[Tuple[int, int], ...], ...], ...]
    ) -> SearchResult:
        nonlocal visited_states, tried_vectors

        visited_states += 1

        if max_nodes is not None and visited_states > max_nodes:
            raise RuntimeError(f"search exceeded max_nodes={max_nodes}")

        cur = deserialize_matrix(serialized)
        active = nonzero_columns(cur)

        if len(active) <= 1:
            J, C, X_sat, X_term = terminal_saturation(cur, terminal_relation)

            final_active = nonzero_columns(X_term)
            final_active_column = final_active[0] if final_active else None

            final_rank_weight = len(J)
            final_weight = final_charged_cost(
                X_sat,
                final_active_column,
                count_unit_multiplications=count_unit_multiplications,
            )

            return SearchResult(
                total_weight=final_weight,
                total_rank_weight=final_rank_weight,
                steps=[],
                final_weight=final_weight,
                final_rank_weight=final_rank_weight,
                final_matrix=X_sat,
                final_raw_matrix=cur,
                final_basis_rows=J,
                final_coordinates=C,
                final_active_column=final_active_column,
                search_mode=search_mode,
                count_unit_multiplications=count_unit_multiplications,
            )

        _, n = matrix_shape(cur)
        best: Optional[SearchResult] = None

        for u in allowed_specialization_vectors(n, active, mode=search_mode):
            tried_vectors += 1

            (
                k,
                rank_weight,
                X_u,
                J_col,
                C_col,
                col_sat,
                L,
            ) = specialize(cur, u)

            charged_weight = column_charged_cost(
                col_sat,
                count_unit_multiplications=count_unit_multiplications,
            )

            step = SearchStep(
                u=u,
                eliminated_col=k,
                linear_form_coeffs=L,
                rank_weight=rank_weight,
                charged_weight=charged_weight,
                sp_basis_rows=J_col,
                sp_coordinates=C_col,
                sp_saturated_column=col_sat,
                x_u_matrix=X_u,
            )

            next_X = X_u

            if do_saturation_each_step:
                J_rem, C_rem, X_sat = saturation(X_u)

                step.sat_basis_rows = J_rem
                step.sat_coordinates = C_rem
                step.sat_matrix = X_sat

                next_X = X_sat

            sub = solve(serialize_matrix(next_X))

            candidate = SearchResult(
                total_weight=charged_weight + sub.total_weight,
                total_rank_weight=rank_weight + sub.total_rank_weight,
                steps=[step] + sub.steps,
                final_weight=sub.final_weight,
                final_rank_weight=sub.final_rank_weight,
                final_matrix=sub.final_matrix,
                final_raw_matrix=sub.final_raw_matrix,
                final_basis_rows=sub.final_basis_rows,
                final_coordinates=sub.final_coordinates,
                final_active_column=sub.final_active_column,
                search_mode=search_mode,
                count_unit_multiplications=count_unit_multiplications,
            )

            if best is None:
                best = candidate
            else:
                old_key = (best.total_weight, best.total_rank_weight)
                new_key = (candidate.total_weight, candidate.total_rank_weight)

                if new_key < old_key:
                    best = candidate

        assert best is not None
        return best

    ans = solve(serialize_matrix(X))

    ans.visited_states = visited_states
    ans.tried_vectors = tried_vectors
    ans.search_mode = search_mode
    ans.count_unit_multiplications = count_unit_multiplications

    return ans


def matrix_pretty(X: Matrix, *, indent: str = "") -> str:
    if not X:
        return indent + "[]"

    return "\n".join(
        indent + "[" + ", ".join(str(c) for c in row) + "]"
        for row in X
    )


def coord_pretty(C: CoordMatrix, *, indent: str = "") -> str:
    if not C:
        return indent + "[]"

    return "\n".join(
        indent + "[" + ", ".join(f5_str(x) for x in row) + "]"
        for row in C
    )


def format_vector(v: Sequence[int]) -> str:
    return "(" + ", ".join(f5_str(x) for x in v) + ")"


def format_linear_form(coeffs: Sequence[int], *, var_prefix: str = "x") -> str:
    terms: List[str] = []

    for idx, c in enumerate(coeffs, start=1):
        c = f5_signed(c)

        if c == 0:
            continue

        var = f"{var_prefix}_{idx}"

        if abs(c) == 1:
            body = var
        else:
            body = f"{abs(c)}{var}"

        if not terms:
            terms.append(body if c > 0 else f"-{body}")
        else:
            terms.append(("+ " if c > 0 else "- ") + body)

    return " ".join(terms) if terms else "0"


Expr = Dict[str, int]


@dataclass
class TermDef:
    name: str
    formula: str
    counted: bool
    note: str


def add_expr(a: Expr, b: Expr, scale: int = 1) -> Expr:
    out = dict(a)

    for name, coeff in b.items():
        out[name] = (out.get(name, 0) + scale * coeff) % P
        if out[name] == 0:
            del out[name]

    return out


def mat_expr(C: CoordMatrix, vec: List[Expr]) -> List[Expr]:
    out: List[Expr] = []

    for row in C:
        e: Expr = {}

        for coeff, expr in zip(row, vec):
            coeff %= P
            if coeff:
                e = add_expr(e, expr, coeff)

        out.append(e)

    return out


def build_direct_expressions(result: SearchResult) -> Tuple[List[TermDef], List[Expr]]:
    terms: List[TermDef] = []
    step_local_exprs: List[List[Expr]] = []

    mult_count = 0
    free_count = 0

    for t, step in enumerate(result.steps, start=1):
        L_name = f"L_{t}"
        local: List[Expr] = []

        for row in step.sp_saturated_column:
            coeff = row[0]

            if coeff.is_zero():
                local.append({})
                continue

            if not result.count_unit_multiplications and coeff.is_one():
                free_count += 1
                name = f"f_{free_count}"
                terms.append(
                    TermDef(
                        name=name,
                        formula=L_name,
                        counted=False,
                        note=f"free copy of {L_name}",
                    )
                )
                local.append({name: 1})
            else:
                mult_count += 1
                name = f"p_{mult_count}"
                terms.append(
                    TermDef(
                        name=name,
                        formula=f"({coeff}) * {L_name}",
                        counted=True,
                        note="multiplication",
                    )
                )
                local.append({name: 1})

        step_local_exprs.append(local)

    final_local: List[Expr] = []

    if result.final_active_column is not None:
        for row in result.final_matrix:
            coeff = row[result.final_active_column]

            if coeff.is_zero():
                final_local.append({})
                continue

            if not result.count_unit_multiplications and coeff.is_one():
                free_count += 1
                name = f"f_{free_count}"
                terms.append(
                    TermDef(
                        name=name,
                        formula="L_final",
                        counted=False,
                        note="free copy of L_final",
                    )
                )
                final_local.append({name: 1})
            else:
                mult_count += 1
                name = f"p_{mult_count}"
                terms.append(
                    TermDef(
                        name=name,
                        formula=f"({coeff}) * L_final",
                        counted=True,
                        note="multiplication",
                    )
                )
                final_local.append({name: 1})

    expr_next = mat_expr(result.final_coordinates, final_local)

    for step, local_exprs in reversed(list(zip(result.steps, step_local_exprs))):
        sp_part = mat_expr(step.sp_coordinates, local_exprs)

        if step.sat_coordinates is not None:
            next_part = mat_expr(step.sat_coordinates, expr_next)
        else:
            next_part = expr_next

        expr_next = [
            add_expr(a, b)
            for a, b in zip(sp_part, next_part)
        ]

    return terms, expr_next


def expr_to_str(expr: Expr, term_order: Sequence[str]) -> str:
    if not expr:
        return "0"

    pieces: List[str] = []

    for name in term_order:
        if name not in expr:
            continue

        c = f5_signed(expr[name])

        if c == 0:
            continue

        if abs(c) == 1:
            body = name
        else:
            body = f"{abs(c)}{name}"

        if not pieces:
            pieces.append(body if c > 0 else f"-{body}")
        else:
            pieces.append(("+ " if c > 0 else "- ") + body)

    return " ".join(pieces) if pieces else "0"


def print_direct_report(
    result: SearchResult,
    *,
    var_prefix: str = "x",
    output_prefix: str = "psi",
    show_path: bool = True,
) -> None:
    terms, output_exprs = build_direct_expressions(result)
    term_order = [t.name for t in terms]

    charged_decomposition = [s.charged_weight for s in result.steps] + [result.final_weight]
    rank_decomposition = [s.rank_weight for s in result.steps] + [result.final_rank_weight]

    print("========== Exhaustive Specialization Search ==========")
    print(f"minimum charged scalar multiplications: {result.total_weight}")
    print(
        "charged decomposition: "
        + " + ".join(str(x) for x in charged_decomposition)
        + f" = {result.total_weight}"
    )
    print(f"rank weight including unit rows: {result.total_rank_weight}")
    print(
        "rank decomposition: "
        + " + ".join(str(x) for x in rank_decomposition)
        + f" = {result.total_rank_weight}"
    )
    print(
        "count 1 * L as multiplication: "
        + ("yes" if result.count_unit_multiplications else "no")
    )
    print(f"search mode: {result.search_mode}")
    print(f"visited states: {result.visited_states}")
    print(f"tried specialization vectors: {result.tried_vectors}")
    print()

    print("---------- Linear forms ----------")
    for t, step in enumerate(result.steps, start=1):
        print(
            f"L_{t} = "
            f"{format_linear_form(step.linear_form_coeffs, var_prefix=var_prefix)}"
        )

    if result.final_active_column is not None:
        coeffs = [0] * (result.final_active_column + 1)
        coeffs[result.final_active_column] = 1
        print(f"L_final = {format_linear_form(coeffs, var_prefix=var_prefix)}")

    print()

    counted_terms = [t for t in terms if t.counted]
    free_terms = [t for t in terms if not t.counted]

    print("---------- Scalar multiplications to compute ----------")
    if counted_terms:
        for t in counted_terms:
            print(f"{t.name} = {t.formula}")
    else:
        print("none")

    print()

    if free_terms:
        print("---------- Free copies, not counted as multiplication ----------")
        for t in free_terms:
            print(f"{t.name} = {t.formula}    # {t.note}")
        print()

    print("---------- Direct recovery of the outputs ----------")
    for i, expr in enumerate(output_exprs, start=1):
        print(f"{output_prefix}_{i} = {expr_to_str(expr, term_order)}")

    print()

    if show_path:
        print("---------- Search path ----------")
        for t, step in enumerate(result.steps, start=1):
            print(
                f"Step {t}: u_{t} = {format_vector(step.u)}, "
                f"eliminate x_{step.eliminated_col + 1}, "
                f"L_{t} = {format_linear_form(step.linear_form_coeffs, var_prefix=var_prefix)}, "
                f"charged w_{t} = {step.charged_weight}, "
                f"rank w_{t} = {step.rank_weight}"
            )

        print(
            f"Final: charged w_final = {result.final_weight}, "
            f"rank w_final = {result.final_rank_weight}"
        )

        if result.final_rank_weight == 0:
            print("Final reason: terminal primitive relation makes the last matrix zero.")

        print()


if __name__ == "__main__":
    X = make_matrix_from_exponents(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        omega_order=3,
        relation="root",
    )

    result = exhaustive_search(
        X,
        do_saturation_each_step=True,
        max_nodes=None,
        search_mode="complete",
        terminal_relation="primitive",
        count_unit_multiplications=False,  
    )

    print_direct_report(result, show_path=True)