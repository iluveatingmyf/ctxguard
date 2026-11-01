"""Rough-set analysis of the specification K (thesis Sec. 2.5.1; used by m2-audit,
vocab-metrics and k-compare).

For one (effect e, asset v) the decision table is
    T_{e,v} = (U, Q ∪ {δ}),  U = Σ (feasible abstract situations),
    Q = the context factors,  δ(s) = L(s, e, v) ∈ {0, +, -} (closed world: 0 if no binding).
Everything is computed exactly by enumeration of Σ (no sampling):

  conflicts(B)      number of IND(B)-classes that contain two different decisions.
                    T is consistent w.r.t. B  <=>  conflicts(B) == 0.
  core              {q : conflicts(Q - {q}) > 0}            (necessity of a factor)
  reduct            a minimal B ⊆ Q with conflicts(B) == 0, found by backward
                    elimination; consistency is monotone in B, so the result is
                    irreducible (no single factor can be dropped).
  global table      δ(s) = (L(s,e,v))_{(e,v)}; its core is the union of the per-pair
                    cores (a factor is dispensable for the vector decision iff it is
                    dispensable for every component).
  value merges      values u, w of factor q are indiscernible for the decisions iff for
                    every pair and every s with s_q = u (s' = s with s_q = w, both
                    feasible) δ(s) = δ(s'). The coarsest decision-preserving partition
                    of D_q groups indiscernible values (attribute-value granulation).
  prime check       a binding (multi-valued cube) is a prime implicant of its label's
                    ON-set iff adding any single value to any literal makes the cube
                    contain a feasible cell with a different label (Sec. 2.5.2).

Consistency of T on REAL situations (several real situations collapsing onto one
abstract cell with different expert labels) cannot be enumerated; it is tested by
the CEGAR-style refutation (binding.run_refute, vocab.run_refine).
"""
from __future__ import annotations

import numpy as np

from .model import Binding, Vocab

LAB = {"0": 0, "+": 1, "-": 2}
INV = {0: "0", 1: "+", 2: "-"}


class Tables:
    """codes: (N, n) int array of value indices for Σ;  L: (P, N) int8 labels per pair."""

    def __init__(self, vocab: Vocab, bindings: list[Binding], pairs: list[tuple] | None = None):
        self.vocab = vocab
        self.sigma = vocab.sigma()
        self.codes = np.array([[vocab.domain(n).index(v) for n, v in zip(vocab.names, c)] for c in self.sigma],
                              dtype=np.int16).reshape(len(self.sigma), len(vocab.names))
        by = {}
        for b in bindings:
            by.setdefault((b.effect, b.asset), []).append(b)
        self.pairs = sorted(set(by) | set(pairs or []))
        self.bindings = by
        self.L = np.zeros((len(self.pairs), len(self.sigma)), dtype=np.int8)
        for pi, pair in enumerate(self.pairs):
            unset = np.ones(len(self.sigma), dtype=bool)
            for b in by.get(pair, []):
                m = self.mask(b.phi) & unset
                self.L[pi, m] = LAB[b.label]
                unset &= ~m

    def mask(self, phi: dict) -> np.ndarray:
        m = np.ones(len(self.sigma), dtype=bool)
        for n, vs in phi.items():
            i = self.vocab.index[n]
            idx = [self.vocab.domain(n).index(v) for v in vs]
            m &= np.isin(self.codes[:, i], idx)
        return m

    # ------------------------------------------------------------ IND(B)
    def _classes(self, keep: list[int]) -> np.ndarray:
        if not keep:
            return np.zeros(len(self.sigma), dtype=np.int64)
        _, inv = np.unique(self.codes[:, keep], axis=0, return_inverse=True)
        return inv.reshape(-1)

    def conflicts(self, keep: list[int], rows: list[int] | None = None) -> np.ndarray:
        """Per pair: number of IND(keep) classes with more than one decision value."""
        inv = self._classes(keep)
        G = int(inv.max()) + 1 if len(inv) else 0
        rows = range(len(self.pairs)) if rows is None else rows
        out = []
        for pi in rows:
            lo = np.full(G, 9, dtype=np.int8)
            hi = np.full(G, -1, dtype=np.int8)
            np.minimum.at(lo, inv, self.L[pi])
            np.maximum.at(hi, inv, self.L[pi])
            out.append(int((lo != hi).sum()))
        return np.array(out, dtype=np.int64)

    def all_cols(self):
        return list(range(len(self.vocab.names)))

    def core(self) -> dict:
        """Per factor: conflicts per pair after removing it (0 everywhere => dispensable)."""
        res = {}
        for i, n in enumerate(self.vocab.names):
            res[n] = self.conflicts([j for j in self.all_cols() if j != i])
        return res

    def reduct(self, rows: list[int], order: list[int] | None = None) -> list[str]:
        """Backward elimination: drop factors (in `order`, default: reverse vocabulary order)
        while the table restricted to `rows` stays consistent."""
        keep = self.all_cols()
        for i in (order if order is not None else keep[::-1]):
            cand = [j for j in keep if j != i]
            if self.conflicts(cand, rows).sum() == 0:
                keep = cand
        return [self.vocab.names[j] for j in keep]

    def support(self, pair) -> list[str]:
        used = {n for b in self.bindings.get(pair, []) for n in b.phi}
        return [n for n in self.vocab.names if n in used]

    # ------------------------------------------------------------ value granulation
    def value_merge_conflicts(self, name: str) -> np.ndarray:
        """M[u, w] = number of (pair, context) where values u and w of `name` get different
        decisions (contexts = assignments of all other factors where both are feasible)."""
        i = self.vocab.index[name]
        D = len(self.vocab.domain(name))
        inv = self._classes([j for j in self.all_cols() if j != i])
        R = int(inv.max()) + 1
        M = np.zeros((D, D), dtype=np.int64)
        val = self.codes[:, i]
        for pi in range(len(self.pairs)):
            grid = np.full((R, D), -1, dtype=np.int8)
            grid[inv, val] = self.L[pi]
            for u in range(D):
                for w in range(u + 1, D):
                    both = (grid[:, u] >= 0) & (grid[:, w] >= 0)
                    k = int((grid[both, u] != grid[both, w]).sum())
                    M[u, w] += k
                    M[w, u] += k
        return M

    def value_partition(self, name: str) -> tuple[list[list[str]], np.ndarray]:
        """Coarsest decision-preserving grouping of the values (greedy in domain order;
        every group is verified pairwise conflict-free)."""
        dom = list(self.vocab.domain(name))
        M = self.value_merge_conflicts(name)
        groups: list[list[int]] = []
        for u in range(len(dom)):
            for g in groups:
                if all(M[u, w] == 0 for w in g):
                    g.append(u)
                    break
            else:
                groups.append([u])
        return [[dom[u] for u in g] for g in groups], M

    # ------------------------------------------------------------ prime implicants
    def prime_check(self) -> dict:
        """For every binding: is it a prime implicant of its label's ON-set (Sec. 2.5.2),
        and is the cover irredundant (each binding covers some cell no other one does)?"""
        n_b = n_prime = n_irr = 0
        bad = []
        for pi, pair in enumerate(self.pairs):
            bs = self.bindings.get(pair, [])
            for lab in ("-", "+"):
                cur = [b for b in bs if b.label == lab]
                if not cur:
                    continue
                off = self.L[pi] != LAB[lab]
                masks = [self.mask(b.phi) for b in cur]
                for k, b in enumerate(cur):
                    n_b += 1
                    prime = True
                    for n in self.vocab.names:
                        allowed = b.phi.get(n, frozenset(self.vocab.domain(n)))
                        for v in self.vocab.domain(n):
                            if v in allowed:
                                continue
                            phi2 = dict(b.phi)
                            phi2[n] = frozenset(allowed | {v})
                            if not (self.mask(phi2) & off).any():
                                prime = False
                                break
                        if not prime:
                            break
                    others = np.zeros(len(self.sigma), dtype=bool)
                    for j, m in enumerate(masks):
                        if j != k:
                            others |= m
                    irr = bool((masks[k] & ~others).any())
                    n_prime += prime
                    n_irr += irr
                    if not (prime and irr):
                        bad.append({"pair": f"{pair[0]} x {pair[1]}", "label": lab, "prime": prime, "irredundant": irr})
        return {"bindings": n_b, "prime": n_prime, "irredundant": n_irr, "violations": bad}

    # ------------------------------------------------------------ export
    def label_strings(self) -> dict:
        return {f"{e}|{v}": "".join(INV[int(x)] for x in self.L[pi]) for pi, (e, v) in enumerate(self.pairs)}


def compare(ta: Tables, tb: Tables) -> dict:
    """Cell-level agreement of two specifications on the same Σ (k-compare)."""
    assert ta.sigma == tb.sigma, "both specs must be loaded with the same vocabulary"
    pairs = sorted(set(ta.pairs) | set(tb.pairs))
    N = len(ta.sigma)
    zero = np.zeros(N, dtype=np.int8)
    per, agree, total = {}, 0, 0
    conf = np.zeros((3, 3), dtype=np.int64)
    for p in pairs:
        a = ta.L[ta.pairs.index(p)] if p in ta.pairs else zero
        b = tb.L[tb.pairs.index(p)] if p in tb.pairs else zero
        k = int((a == b).sum())
        per[f"{p[0]} x {p[1]}"] = round(k / N, 4)
        agree += k
        total += N
        for x in range(3):
            for y in range(3):
                conf[x, y] += int(((a == x) & (b == y)).sum())
    po = float(agree / max(1, total))
    pe = float(sum(int(conf[i, :].sum()) * int(conf[:, i].sum()) for i in range(3)) / max(1, total) ** 2)
    return {"cells_per_pair": N, "pairs": len(pairs), "agreement": round(po, 4),
            "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0,
            "confusion(rows=A,cols=B; 0,+,-)": conf.tolist(),
            "pairs_below_0.95": {k: v for k, v in per.items() if v < 0.95}}
