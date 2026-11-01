"""O4: exact-region -> minimal pattern cover (multi-valued logic minimization).

Given the ON-set (cells with label l), the OFF-set (feasible cells with a
different label) and don't-cares (infeasible cells), produce a set of cubes
(patterns) whose union equals the ON-set on Sigma.

Algorithm = ESPRESSO-style EXPAND + IRREDUNDANT (Brayton et al. 1984):
  EXPAND: grow a seed cell factor by factor (fixed vocabulary order, first try
          the full domain, then single values) while the cube contains no
          OFF-set cell. The result is a *prime* cube: no single value can be
          added to any factor (cubes are products, so a strictly larger valid
          cube implies a valid one-value extension).
  IRREDUNDANT: greedy set cover of the ON-set by prime cubes, ties broken by
          fewer constrained factors, then lexicographic order (deterministic).
Lossless: union(cubes) ∩ Sigma == ON-set exactly (checked by `verify`).
Implementation uses per-(factor,value) bitmasks over cell indices.
"""
from __future__ import annotations


class _Masks:
    def __init__(self, cells, domains):
        self.full = (1 << len(cells)) - 1
        self.m = [{v: 0 for v in d} for d in domains]
        for idx, c in enumerate(cells):
            bit = 1 << idx
            for i, v in enumerate(c):
                self.m[i][v] |= bit

    def inside(self, cube) -> int:
        acc = self.full
        for i, s in enumerate(cube):
            u = 0
            for v in s:
                u |= self.m[i][v]
            acc &= u
            if not acc:
                return 0
        return acc


def expand(cell, domains, off: _Masks):
    cube = [frozenset([v]) for v in cell]
    changed = True
    while changed:
        changed = False
        for i, dom in enumerate(domains):
            if len(cube[i]) == len(dom):
                continue
            full = cube[:i] + [frozenset(dom)] + cube[i + 1:]
            if not off.inside(full):
                cube, changed = full, True
                continue
            for v in dom:
                if v in cube[i]:
                    continue
                cand = cube[:i] + [cube[i] | {v}] + cube[i + 1:]
                if not off.inside(cand):
                    cube, changed = cand, True
    return tuple(cube)


def minimize(on_cells, off_cells, domains):
    on_cells = sorted(set(on_cells))
    if not on_cells:
        return []
    off = _Masks(list(off_cells), domains)
    on = _Masks(on_cells, domains)
    primes, covered = [], 0
    for idx, c in enumerate(on_cells):
        if covered >> idx & 1:
            continue
        p = expand(c, domains, off)
        if p not in primes:
            primes.append(p)
            covered |= on.inside(p)
    # a second pass with reversed factor order adds alternative primes
    rev = list(range(len(domains)))[::-1]
    rdom = [domains[i] for i in rev]
    roff = _Masks([tuple(o[i] for i in rev) for o in off_cells], rdom)
    for c in on_cells[:: max(1, len(on_cells) // 50)]:
        rc = tuple(c[i] for i in rev)
        rp = expand(rc, rdom, roff)
        p = tuple(rp[rev.index(i)] for i in range(len(domains)))
        if p not in primes:
            primes.append(p)
    cover_bits = {p: on.inside(p) for p in primes}
    uncovered = on.full
    chosen = []
    while uncovered:
        def score(p):
            lits = sum(1 for i, s in enumerate(p) if len(s) < len(domains[i]))
            return (-bin(cover_bits[p] & uncovered).count("1"), lits, [sorted(s) for s in p])
        best = min(primes, key=score)
        got = cover_bits[best] & uncovered
        if not got:
            raise RuntimeError("cover failed")
        chosen.append(best)
        uncovered &= ~got
    return chosen


def cube_to_pattern(cube, names, domains):
    return {n: frozenset(s) for n, s, d in zip(names, cube, domains) if len(s) < len(d)}


def verify(cubes, on_cells, sigma) -> bool:
    on = set(on_cells)
    got = {c for c in sigma if any(all(c[i] in p[i] for i in range(len(p))) for p in cubes)}
    return got == on
