import math as m
import random
from math import cos, exp, log, sin, sqrt, tan, tanh
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import dreal
from dreal import *


class DRealSolver:
    def __init__(self, delta: float = 1e-6):
        self.delta = 0  # float(delta)

    def check(self, phi, delta: float = 1e-6) -> Tuple[bool, Optional[Dict[Any, Any]]]:
        box = CheckSatisfiability(phi, delta)
        if box is None:
            # print(f"[dReal] delta-unsat (delta={delta})")
            return False, None
        else:
            # print(f"[dReal] delta-sat (delta={delta}), interval box:")
            # for v, I in box.items():
            # print(f"  {str(v)} in [{I.lb()}, {I.ub()}]")
            return True, box

    # get the midpoints of each interval box w.r.t. variables
    def midpoint(self, box: Dict[Any, Any], eps: float = 1e-6) -> Dict[str, float]:
        midpoints = {}
        for v, I in box.items():
            lb, ub = I.lb(), I.ub()
            name = f"{v}"  # name
            if lb == float("-inf") and ub == float("inf"):
                continue
            elif lb == float("-inf"):
                midpoints[name] = ub - eps
            elif ub == float("inf"):
                midpoints[name] = lb + eps
            else:
                midpoints[name] = 0.5 * (lb + ub)
        return midpoints

    def is_counterexample(
        self,
        phi,
        lhs: List[Callable[[Dict[str, float]], bool]],
        rhs: Callable[[Dict[str, float]], bool],
        cex: Dict[str, float],
        tol: float,
        ball_eta=1e-6,
        delta_refine=1e-8,
    ) -> bool:
        # All lhs predicates must hold
        for f in lhs:
            if not f(cex):
                return False
        # rhs must fail
        if rhs(cex) is not False:
            return False

        local_constraints = []
        for name, val in cex.items():
            v = Variable(name)
            local_constraints.append(abs(v - float(val)) <= float(ball_eta))

        local_phi = And(phi, And(*local_constraints)) if local_constraints else phi
        refined_box = CheckSatisfiability(local_phi, float(delta_refine))

        return refined_box is not None

    def corners(self, box: Dict[Any, Any], limit: int = 64) -> List[Dict[str, float]]:
        # Enumerate corner points of the box; truncate if too many variables
        items = [
            (str(v), (I.lb(), I.ub()))
            for v, I in box.items()
            if (I.lb() != float("-inf") and I.ub() != float("inf"))
        ]
        n = len(items)
        if n == 0:
            return []
        if n > 6:  # avoid high dimension, 2^6 = 64 corners
            items = items[:6]
            n = 6
        envs: List[Dict[str, float]] = []
        for mask in range(1 << n):
            e: Dict[str, float] = {}
            for i, (name, (lb, ub)) in enumerate(items):
                e[name] = ub if (mask & (1 << i)) else lb
            envs.append(e)
            if len(envs) >= limit:
                break
        return envs

    def random_in_box(self, box: Dict[Any, Any]) -> Dict[str, float]:
        # Sample one random point from the interval box
        e: Dict[str, float] = {}
        for v, I in box.items():
            lb, ub = I.lb(), I.ub()
            if lb == float("-inf") or ub == float("inf"):
                continue
            e[str(v)] = random.uniform(lb, ub)
        return e

    def numeric_cex(
        self,
        box: Dict[Any, Any],  # dReal interval box: Variable -> Interval
        lhs_check: Iterable[Callable[[Dict[str, float]], bool]],
        rhs_check: Callable[[Dict[str, float]], bool],
        tol: float = 1e-9,
        corner_try: bool = True,
        random_try: int = 0,
    ) -> Optional[Dict[str, float]]:

        # 1. Midpoint of the interval box
        mid = self.midpoint(box)
        if self.is_counterexample(mid, lhs_check, rhs_check, tol):
            return mid

        # 2. Corner points (useful when dimension is small)
        if corner_try:
            corners = self.corners(box, limit=64)  # cap to avoid explosion
            for e in corners:
                if self.is_counterexample(e, lhs_check, rhs_check, tol):
                    return e

        # 3. Random samples
        if random_try > 0:
            for _ in range(random_try):
                e = self.random_in_box(box)
                if self.is_counterexample(e, lhs_check, rhs_check, tol):
                    return e

        # No valid counterexample found
        return None

    def print_box(self, box: Dict[Any, Any]):
        for v, I in box.items():
            print(f"  {v} in [{I.lb()}, {I.ub()}]")

    def _dedup_cex(
        self, cex: List[Dict[str, float]], tol: float = 1e-9
    ) -> List[Dict[str, float]]:
        """
        Deduplicate cex dictionaries under a numeric tolerance.
        Rounds each value to 'decimals' derived from tol and uses that as a key.
        """
        if not cex:
            return cex
        decimals = max(0, int(-m.log10(tol)))  # e.g., tol=1e-9 -> 9 decimals
        seen = set()
        uniq: List[Dict[str, float]] = []
        for e in cex:
            key = tuple(sorted((k, round(float(v), decimals)) for k, v in e.items()))
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        return uniq

    def solve(
        self,
        phi,
        lhs: List[Callable[[Dict[str, float]], bool]],
        rhs: Callable[[Dict[str, float]], bool],
        delta: float = 1e-6,
        random_try: int = 16,
        tol: float = 1e-9,
    ):
        """
        Check δ-sat; if SAT, try midpoint -> first corner -> random,
        collect ALL counterexamples found across the three strategies, and deduplicate.

        Returns:
            (is_sat, box, counterexamples) where counterexamples is a (possibly empty) list of cex dicts.
        """
        is_sat, box = self.check(phi, delta)
        if not is_sat:
            # print("Soundness proved.")
            return False, None, []

        if is_sat and box is None:
            # print("Delta-sat reported but no IntervalBox returned. No concrete counterexample found.")
            return True, None, []

        candidates: List[Dict[str, float]] = []

        mid = self.midpoint(box)
        if mid:
            candidates.append(mid)

        cs = self.corners(box, limit=64)
        if cs:
            candidates.append(cs[0])

        for _ in range(max(0, random_try)):
            e = self.random_in_box(box)
            if e:
                candidates.append(e)

        counterexamples = []
        for c in candidates:
            if self.is_counterexample(phi, lhs, rhs, c, tol):
                counterexamples.append(c)

        # deduplicate under tolerance
        counterexamples = self._dedup_cex(counterexamples, tol=tol)

        if counterexamples:
            print(f"Collected {len(counterexamples)} counterexample(s):")
            for i, ce in enumerate(counterexamples, 1):
                print(f"  CE#{i}: {ce}")
        else:
            print("Delta-sat, but no numeric counterexample found.")

        return True, box, counterexamples


if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")

    # -------------- Example 1: SAT --------------
    phi1 = And(-2 <= x, x <= 2, -2 <= y, y <= 2, x**2 + y**2 == 1)
    lhs1: List[Callable[[Dict[str, float]], bool]] = [
        lambda e: -2.0 <= e["x"] <= 2.0,
        lambda e: -2.0 <= e["y"] <= 2.0,
    ]
    EPS1 = 1e-7
    rhs1: Callable[[Dict[str, float]], bool] = (
        lambda e: abs(e["x"] ** 2 + e["y"] ** 2 - 1.0) <= EPS1
    )
    solver1 = DRealSolver()
    solver1.solve(phi=phi1, lhs=lhs1, rhs=rhs1, delta=1e-6)

    print("\n\n")

    # -------------- Example 2: UNSAT --------------
    phi2 = dreal.sin(x) >= 2
    lhs2: List[Callable[[Dict[str, float]], bool]] = []
    rhs2: Callable[[Dict[str, float]], bool] = lambda e: (m.sin(e["x"]) >= 2.0)

    solver2 = DRealSolver()
    solver2.solve(phi=phi2, lhs=lhs2, rhs=rhs2, delta=1e-6)
