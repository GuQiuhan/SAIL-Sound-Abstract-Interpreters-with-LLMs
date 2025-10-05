"""
CEGIS-style synthesis of best L-transformers, with:
- Example sets E+ (positive) / E- (negative)
- CheckSoundness (SMT)
- CheckPrecision  (SMT)
- Synthesize / MaxSatSynthesize (enumeration + partial MaxSAT drop)
"""

from __future__ import annotations

import itertools
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ===== Optional SMT (Z3) =====
try:
    from z3 import And, Implies, Int, Not, Or, Real, Solver, sat

    HAS_Z3 = True
except Exception:
    HAS_Z3 = False

# ===== Optional ANTLR integration =====
USE_MY_DSL = False  # 改为 True 可接入你自己的 .g4 DSL
# 如果改为 True，请确保同目录下已有 miniDSLLexer.py / miniDSLParser.py / miniDSLVisitor.py
if USE_MY_DSL:
    try:
        from antlr4 import CommonTokenStream, InputStream

        # 假设语法名是 miniDSL（按你的 g4 名称对应改）
        from miniDSLLexer import miniDSLLexer
        from miniDSLParser import miniDSLParser
        from miniDSLVisitor import miniDSLVisitor

        HAS_ANTLR = True
    except Exception:
        HAS_ANTLR = False
        print(
            "[WARN] ANTLR modules not found. Set USE_MY_DSL=False or generate parser from your .g4."
        )
else:
    HAS_ANTLR = False

# ==============================
# Core Types
# ==============================
Abs = Any  # abstract value
Conc = Any  # concrete value
Prog = Any  # program (DSL AST or handle)
ExPos = Tuple[Abs, Conc]  # ⟨a, c'⟩
ExNeg = Tuple[Abs, Conc]  # ⟨a, c'⟩

# ==============================
# Example set
# ==============================
@dataclass
class ExampleSet:
    Epos: List[ExPos] = field(default_factory=list)
    Eneg: List[ExNeg] = field(default_factory=list)

    def add_pos(self, e: ExPos):
        if e not in self.Epos:
            self.Epos.append(e)

    def add_neg(self, e: ExNeg):
        if e not in self.Eneg:
            self.Eneg.append(e)

    def remove_many_neg(self, dropped: Set[ExNeg]):
        self.Eneg = [e for e in self.Eneg if e not in dropped]


# ==============================
# DSL Adapter interface
# ==============================
class DSLAdapter:
    # Abstract lattice
    def is_bot(self, a: Abs) -> bool:
        raise NotImplementedError

    def bot(self) -> Abs:
        raise NotImplementedError

    def abs_leq(self, a1: Abs, a2: Abs) -> bool:
        raise NotImplementedError

    def abs_equal(self, a1: Abs, a2: Abs) -> bool:
        raise NotImplementedError

    # Concretization & concrete semantics
    def concrete_in_gamma(self, a: Abs, c: Conc) -> bool:
        raise NotImplementedError

    def gamma_concrete_samples(
        self, a: Abs, *, max_samples: int = 32
    ) -> Iterable[Conc]:
        raise NotImplementedError

    def concrete_forward(self, c: Conc) -> Conc:
        raise NotImplementedError

    # DSL program execution & enumeration
    def eval_prog_on_abs(self, prog: Prog, a: Abs) -> Abs:
        raise NotImplementedError

    def enumerate_programs(self, depth: int) -> Iterable[Prog]:
        raise NotImplementedError

    def program_cost(self, prog: Prog) -> int:
        return 0

    def pretty(self, prog: Prog) -> str:
        return str(prog)

    # SMT hooks (optional)
    def has_z3(self) -> bool:
        return HAS_Z3

    def z3_soundness_cex(self, prog: Prog) -> Optional[ExPos]:
        return None

    def z3_precision_cex(
        self, prog: Prog, Epos: List[ExPos], Eneg: List[ExNeg]
    ) -> Optional[ExNeg]:
        return None


# ==============================
# Synthesizer helpers
# ==============================
@dataclass
class SynthOptions:
    max_depth: int = 3
    timeout_s: float = 300.0
    prefer_small: bool = True


def satisfies_examples(adapter: DSLAdapter, prog: Prog, E: ExampleSet) -> bool:
    for a, cprime in E.Epos:
        out = adapter.eval_prog_on_abs(prog, a)
        if not adapter.concrete_in_gamma(out, cprime):
            return False
    for a, cprime in E.Eneg:
        out = adapter.eval_prog_on_abs(prog, a)
        if adapter.concrete_in_gamma(out, cprime):
            return False
    return True


def synthesize(adapter: DSLAdapter, E: ExampleSet, opt: SynthOptions) -> Optional[Prog]:
    t0 = time.time()
    best = None
    for depth in range(1, opt.max_depth + 1):
        for prog in adapter.enumerate_programs(depth):
            if time.time() - t0 > opt.timeout_s:
                return best[1] if best else None
            if satisfies_examples(adapter, prog, E):
                cost = adapter.program_cost(prog)
                if best is None or (opt.prefer_small and cost < best[0]):
                    best = (cost, prog)
                    return best[1]  # early accept
    return best[1] if best else None


def maxsat_synthesize(
    adapter: DSLAdapter, E: ExampleSet, opt: SynthOptions
) -> Tuple[Optional[Prog], Set[ExNeg]]:
    for k in range(0, len(E.Eneg) + 1):
        for dropped in itertools.combinations(E.Eneg, k):
            drop_set = set(dropped)
            trial = ExampleSet(list(E.Epos), [e for e in E.Eneg if e not in drop_set])
            prog = synthesize(adapter, trial, opt)
            if prog is not None:
                return prog, drop_set
    return None, set()


# ==============================
# Checks
# ==============================
def check_soundness(
    adapter: DSLAdapter, prog: Prog, sample_As: Iterable[Abs], *, perA_samples: int = 16
) -> Tuple[bool, Optional[ExPos]]:
    if adapter.has_z3():
        cex = adapter.z3_soundness_cex(prog)
        return (cex is None), cex
    # sampling fallback
    for a in sample_As:
        out = adapter.eval_prog_on_abs(prog, a)
        for c in adapter.gamma_concrete_samples(a, max_samples=perA_samples):
            cprime = adapter.concrete_forward(c)
            if not adapter.concrete_in_gamma(out, cprime):
                return False, (a, cprime)
    return True, None


def check_precision(
    adapter: DSLAdapter,
    prog: Prog,
    E: ExampleSet,
    sample_As: Iterable[Abs],
    *,
    perA_samples: int = 16,
) -> Tuple[bool, Optional[ExNeg]]:
    if adapter.has_z3():
        cex = adapter.z3_precision_cex(prog, E.Epos, E.Eneg)
        return (cex is None), cex
    opt = SynthOptions(max_depth=CONFIG["MAX_DEPTH"], timeout_s=CONFIG["TIMEOUT_S"])
    for a in sample_As:
        out = adapter.eval_prog_on_abs(prog, a)
        for c in adapter.gamma_concrete_samples(out, max_samples=perA_samples):
            cand = (a, c)
            if cand in E.Eneg:
                continue
            trial = ExampleSet(list(E.Epos), list(E.Eneg) + [cand])
            h = synthesize(adapter, trial, opt)
            if h is not None:
                return False, cand
    return True, None


def check_consistency(adapter: DSLAdapter, E: ExampleSet, opt: SynthOptions) -> bool:
    return synthesize(adapter, E, opt) is not None


# ==============================
# Main loop (Alg.1 with fair scheduling)
# ==============================
def synthesize_best_L_transformer(
    adapter: DSLAdapter,
    A_universe: Iterable[Abs],
    *,
    max_depth: int = 3,
    timeout_s: float = 600.0,
    schedule_k: int = 3,
) -> Optional[Prog]:
    E = ExampleSet()
    opt = SynthOptions(max_depth=max_depth, timeout_s=timeout_s)
    current = None
    if check_consistency(adapter, E, opt):
        current = synthesize(adapter, E, opt)
    else:
        current, dropped = maxsat_synthesize(adapter, E, opt)
        if current is None:
            return None

    isSound = False
    isPrecise = False
    consec_prec = 0
    rounds = 0
    poolA = list(A_universe)

    while not (isSound and isPrecise):
        rounds += 1
        if rounds > CONFIG["MAX_ROUNDS"]:
            break
        if current is None:
            return None
        if not satisfies_examples(adapter, current, E):
            if check_consistency(adapter, E, opt):
                current = synthesize(adapter, E, opt)
            else:
                current, dropped = maxsat_synthesize(adapter, E, opt)
                if current is None:
                    return None
                E.remove_many_neg(dropped)

        run_precision = True
        if consec_prec >= schedule_k:
            run_precision = False

        if run_precision:
            isPrecise, neg_cex = check_precision(
                adapter, current, E, poolA, perA_samples=CONFIG["PER_A_SAMPLES"]
            )
            if not isPrecise:
                isSound = False
                E.add_neg(neg_cex)
                if check_consistency(adapter, E, opt):
                    current = synthesize(adapter, E, opt)
                else:
                    current, dropped = maxsat_synthesize(adapter, E, opt)
                    if current is None:
                        return None
                    E.remove_many_neg(dropped)
                consec_prec += 1
            else:
                consec_prec += 1
        else:
            isSound, pos_cex = check_soundness(
                adapter, current, poolA, perA_samples=CONFIG["PER_A_SAMPLES"]
            )
            if not isSound:
                isPrecise = False
                E.add_pos(pos_cex)
                if check_consistency(adapter, E, opt):
                    current = synthesize(adapter, E, opt)
                else:
                    current, dropped = maxsat_synthesize(adapter, E, opt)
                    if current is None:
                        return None
                    E.remove_many_neg(dropped)
                consec_prec = 0
            else:
                consec_prec = 0
    return current


# ==============================
# Demo Adapter: Interval + abs
# ==============================
class IntervalAbsAdapter(DSLAdapter):
    """
    Demo transformer: output is an interval [L(a), R(a)].
    We enumerate simple expression trees over {a.l, a.r, 0, +, -, min, max, neg}.
    """

    def _mk(self, l, r):
        return (l, r)

    def is_bot(self, a):
        l, r = a
        return l > r

    def bot(self):
        return (1, 0)

    def abs_equal(self, a1, a2):
        return a1 == a2

    def abs_leq(self, a1, a2):
        l1, r1 = a1
        l2, r2 = a2
        return l2 <= l1 and r1 <= r2

    # concretization & concrete semantics
    def concrete_in_gamma(self, a, c: int):
        l, r = a
        return l <= c <= r

    def gamma_concrete_samples(self, a, *, max_samples=32):
        l, r = a
        if self.is_bot(a):
            return []
        l0 = max(l, -50)
        r0 = min(r, 50)
        if l0 > r0:
            return []
        if r0 - l0 + 1 <= max_samples:
            return list(range(l0, r0 + 1))
        return sorted(random.sample(range(l0, r0 + 1), max_samples))

    def concrete_forward(self, c: int) -> int:
        return abs(c)

    # DSL enumerate & eval
    def enumerate_programs(self, depth: int) -> Iterable[Prog]:
        # Transformer program is a pair of expressions E_L, E_R
        def exprs(d):
            if d == 1:
                yield ("l",)
                yield ("r",)
                yield ("const", 0)
            else:
                for e in exprs(d - 1):
                    yield e
                    yield ("neg", e)
                for e1 in exprs(d - 1):
                    for e2 in exprs(d - 1):
                        yield ("add", e1, e2)
                        yield ("sub", e1, e2)
                        yield ("min", e1, e2)
                        yield ("max", e1, e2)

        es = list(exprs(depth))
        seen = set()
        for eL in es:
            for eR in es:
                key = (eL, eR)
                if key in seen:
                    continue
                seen.add(key)
                yield (eL, eR)

    def _eval_E(self, e, a):
        l, r = a
        tag = e[0]
        if tag == "l":
            return l
        if tag == "r":
            return r
        if tag == "const":
            return e[1]
        if tag == "neg":
            return -self._eval_E(e[1], a)
        if tag == "add":
            return self._eval_E(e[1], a) + self._eval_E(e[2], a)
        if tag == "sub":
            return self._eval_E(e[1], a) - self._eval_E(e[2], a)
        if tag == "min":
            return min(self._eval_E(e[1], a), self._eval_E(e[2], a))
        if tag == "max":
            return max(self._eval_E(e[1], a), self._eval_E(e[2], a))
        raise ValueError(f"Unknown expr tag {tag}")

    def eval_prog_on_abs(self, prog: Prog, a: Abs) -> Abs:
        if self.is_bot(a):
            return self.bot()
        eL, eR = prog
        L = self._eval_E(eL, a)
        R = self._eval_E(eR, a)
        if L > R:
            L, R = R, L
        return (L, R)

    def program_cost(self, prog: Prog) -> int:
        def size(e):
            t = e[0]
            if t in ("l", "r", "const"):
                return 1
            if t == "neg":
                return 1 + size(e[1])
            return 1 + size(e[1]) + size(e[2])

        eL, eR = prog
        return size(eL) + size(eR)

    # ===== Optional: strict SMT soundness for abs on intervals =====
    def has_z3(self) -> bool:
        return HAS_Z3

    def z3_soundness_cex(self, prog: Prog) -> Optional[ExPos]:
        if not HAS_Z3:
            return None
        l, r, c, cprime = Real("l"), Real("r"), Real("c"), Real("cprime")
        # eval prog(a) to get [L(a), R(a)] as affine min/max over l,r (over-approx via sampling fallback)
        # 为简单起见，这里把 eval_prog_on_abs 直接跑一遍，用常数符号近似（更严格的做法可把表达式符号化）
        # 采用小范围枚举 l,r 采样（以避免复杂的符号构造）；找到反例就返回
        for l0 in range(-10, 11, 2):
            for r0 in range(l0, 11, 2):
                out = self.eval_prog_on_abs(prog, (l0, r0))
                L0, R0 = out
                s = Solver()
                # c ∈ [l0, r0]
                s.add(c >= l0, c <= r0)
                # abs concrete semantics
                s.add(Or(And(c >= 0, cprime == c), And(c < 0, cprime == -c)))
                # c' ∉ [L0, R0]
                s.add(Or(cprime < L0, cprime > R0))
                if s.check() == sat:
                    m = s.model()
                    return ((l0, r0), float(m[cprime].as_decimal(10).replace("?", "")))
        return None


# ==============================
# (Optional) Adapter skeleton for your ANTLR DSL
# ==============================
class MyDSLAdapter(DSLAdapter):
    """
    如何接入你的 .g4：
    1) 用 antlr4 生成 Python 解析器（-visitor 建议开启）
    2) 实现 parse(text) -> AST，和 visit(AST, a_abs) -> abstract value
    3) 在 enumerate_programs 中生成候选 DSL 程序字符串（或 AST），交给 parse+eval
    """

    def __init__(self):
        if not (USE_MY_DSL and HAS_ANTLR):
            raise RuntimeError(
                "Enable USE_MY_DSL=True and ensure ANTLR parser files are present."
            )

    # ===== 抽象域（请替换成你的） =====
    def is_bot(self, a):
        return False

    def bot(self):
        return None

    def abs_leq(self, a1, a2):
        return True

    def abs_equal(self, a1, a2):
        return a1 == a2

    # ===== 具体语义与γ（请替换成你的） =====
    def concrete_in_gamma(self, a, c):
        return True

    def gamma_concrete_samples(self, a, *, max_samples=32):
        return []

    def concrete_forward(self, c):
        return c

    # ===== DSL 解析与执行（你需要实现） =====
    class _EvalVisitor(miniDSLVisitor):
        def __init__(self, adapter: "MyDSLAdapter", a_abs: Abs):
            self.adapter = adapter
            self.a = a_abs

        # TODO: 举例：
        # def visitVarExp(self, ctx):
        #     # 返回抽象值或表达式的数值
        #     ...
        # def visitBinopExp(self, ctx):
        #     lhs = self.visit(ctx.left)
        #     rhs = self.visit(ctx.right)
        #     op = ctx.op.text
        #     ...
        #     return result

    def parse(self, text: str):
        # 解析 text -> parse tree -> AST （或直接用 visitor 求值）
        input_stream = InputStream(text)
        lexer = miniDSLLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = miniDSLParser(stream)
        tree = parser.expr()  # 依据你的起始规则修正
        return tree

    def eval_prog_on_abs(self, prog: Prog, a: Abs) -> Abs:
        # prog 可以是字符串，也可以是 parse tree
        if isinstance(prog, str):
            tree = self.parse(prog)
        else:
            tree = prog
        v = MyDSLAdapter._EvalVisitor(self, a)
        return v.visit(tree)

    def enumerate_programs(self, depth: int) -> Iterable[Prog]:
        # TODO: 按你的 DSL 产出候选程序（字符串或 AST）
        # 举例：yield "(max 0 (min (add PREV L) R))"
        yield from ()


# ==============================
# Config
# ==============================
CONFIG: Dict[str, Any] = {
    "MAX_DEPTH": 3,
    "TIMEOUT_S": 180.0,
    "MAX_ROUNDS": 200,
    "PER_A_SAMPLES": 32,
}

# ==============================
# Demo universe for IntervalAbs
# ==============================
def demo_universe() -> Iterable[Abs]:
    ints = list(range(-10, 11, 2))
    As = []
    for l0 in ints:
        for r0 in ints:
            if l0 <= r0:
                As.append((l0, r0))
    As += [(-5, 7), (-3, 0), (0, 9)]
    seen = set()
    out = []
    for a in As:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


# ==============================
# Main
# ==============================
def main():
    if USE_MY_DSL:
        adapter = MyDSLAdapter()
        A_pool = demo_universe()  # 也可以改成你的抽象输入生成器
    else:
        adapter = IntervalAbsAdapter()
        A_pool = demo_universe()

    prog = synthesize_best_L_transformer(
        adapter,
        A_pool,
        max_depth=CONFIG["MAX_DEPTH"],
        timeout_s=CONFIG["TIMEOUT_S"],
        schedule_k=3,
    )
    if prog is None:
        print("Result: ⊥  (language or depth bound likely insufficient)")
        return

    print("=== Best L-transformer (candidate) ===")
    print(adapter.pretty(prog))
    print("Sanity checks:")
    for a in [(-5, -2), (-5, 7), (0, 9), (3, 6)]:
        out = adapter.eval_prog_on_abs(prog, a)
        print(f"  f#({a}) = {out}")


if __name__ == "__main__":
    main()
