"""Core formal objects (Sec. III of the paper).

Factor f = (name, semantics, D)            context factor
Vocab F                                    frozen, versioned factor vocabulary
Sigma  ⊆ D1 x ... x Dn                     feasible abstract situations
Pattern phi : name -> allowed value set    condition pattern, [[phi]] ⊆ Sigma
Binding b = <phi, e, v, label, harm>       conditional binding; label '-' = safety constraint
Spec K = list[Binding]
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import yaml

UNKNOWN = None  # runtime unknown value '?'


@dataclass(frozen=True)
class Factor:
    name: str
    scope: str                 # global | zone | action
    semantics: str
    values: tuple
    value_sem: dict = field(hash=False, compare=False, default_factory=dict)


class Vocab:
    def __init__(self, factors: list[Factor], feasibility: list[dict], version: str):
        self.factors = factors
        self.version = version
        self.names = [f.name for f in factors]
        self.by_name = {f.name: f for f in factors}
        self.index = {n: i for i, n in enumerate(self.names)}
        self.feasibility = [(normalize(c["if"], self), normalize(c["then"], self))
                            for c in feasibility]
        self._sigma = None

    @classmethod
    def load(cls, path: str) -> "Vocab":
        d = yaml.safe_load(open(path))
        fs = []
        for f in d["factors"]:
            vals = f["values"]
            if isinstance(vals, dict):
                order, sem = tuple(str(k) for k in vals), {str(k): v for k, v in vals.items()}
            else:
                order, sem = tuple(str(v) for v in vals), {}
            fs.append(Factor(f["name"], f.get("scope", "global"), f.get("semantics", ""), order, sem))
        return cls(fs, d.get("feasibility", []), d.get("version", "F-?"))

    def domain(self, name: str) -> tuple:
        return self.by_name[name].values

    # -------------------------------------------------------------- Sigma
    def feasible(self, cell: tuple) -> bool:
        for ifp, thenp in self.feasibility:
            if contains(ifp, cell, self) and not contains(thenp, cell, self):
                return False
        return True

    def sigma(self) -> list[tuple]:
        if self._sigma is None:
            self._sigma = [c for c in itertools.product(*[f.values for f in self.factors])
                           if self.feasible(c)]
        return self._sigma

    def product_size(self) -> int:
        n = 1
        for f in self.factors:
            n *= len(f.values)
        return n

    def cells_in(self, phi: dict) -> list[tuple]:
        return [c for c in self.sigma() if contains(phi, c, self)]

    def sample_cells(self, phi: dict, k: int, rng: random.Random) -> list[tuple]:
        cells = self.cells_in(phi)
        return cells if len(cells) <= k else rng.sample(cells, k)

    def cell_dict(self, cell: tuple) -> dict:
        return dict(zip(self.names, cell))

    # ------------------------------------------------------- descriptions
    def describe_vocab(self) -> str:
        lines = []
        for f in self.factors:
            vals = "; ".join(f"{v} = {f.value_sem.get(v, v)}" for v in f.values)
            lines.append(f"- {f.name} [{f.scope}]: {f.semantics} Values: {vals}")
        return "\n".join(lines)

    def describe_pattern(self, phi: dict) -> str:
        parts = []
        for n in self.names:
            if n in phi:
                parts.append(f"{n} ∈ {{{', '.join(sorted(phi[n], key=self.domain(n).index))}}}")
            else:
                parts.append(f"{n} = any")
        return "; ".join(parts)

    def describe_cell(self, cell: tuple) -> str:
        return "; ".join(f"{n} = {v}" for n, v in zip(self.names, cell))


# ------------------------------------------------------------------ patterns
def normalize(phi: dict, vocab: Vocab) -> dict:
    """Drop unconstrained factors; values as frozensets; validate."""
    out = {}
    for n, vals in (phi or {}).items():
        if n not in vocab.by_name:
            raise ValueError(f"unknown factor {n}")
        vs = frozenset(str(v) for v in (vals if isinstance(vals, (list, tuple, set, frozenset)) else [vals]))
        bad = vs - set(vocab.domain(n))
        if bad:
            raise ValueError(f"values {bad} not in domain of {n}")
        if not vs:
            raise ValueError(f"empty value set for {n}")
        if vs != frozenset(vocab.domain(n)):
            out[n] = vs
    return out


def contains(phi: dict, cell: tuple, vocab: Vocab) -> bool:
    return all(cell[vocab.index[n]] in vs for n, vs in phi.items())


def match3(phi: dict, partial: dict):
    """Kleene three-valued match of a pattern against a partial assignment.
    True if every constrained factor is known and allowed; False if any known
    factor violates; None (undetermined) otherwise."""
    unknown = False
    for n, vs in phi.items():
        v = partial.get(n, UNKNOWN)
        if v is UNKNOWN:
            unknown = True
        elif v not in vs:
            return False
    return None if unknown else True


def pattern_to_json(phi: dict, vocab: Vocab) -> dict:
    return {n: sorted(vs, key=vocab.domain(n).index) for n, vs in phi.items()}


def pattern_len(phi: dict) -> int:
    return len(phi)


@dataclass
class Binding:
    phi: dict
    effect: str
    asset: str
    label: str          # '+' or '-'
    harm: str           # mechanism text (required for '-')
    meta: dict = field(default_factory=dict)

    def to_json(self, vocab: Vocab) -> dict:
        return {"pattern": pattern_to_json(self.phi, vocab), "effect": self.effect,
                "asset": self.asset, "label": self.label, "harm": self.harm, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict, vocab: Vocab) -> "Binding":
        return cls(normalize(d["pattern"], vocab), d["effect"], d["asset"], d["label"],
                   d.get("harm", ""), d.get("meta", {}))


def load_spec(path: str, vocab: Vocab) -> list[Binding]:
    d = yaml.safe_load(open(path))
    return [Binding.from_json(b, vocab) for b in d["bindings"]]


def label_of(spec_by_pair: dict, effect: str, asset: str, cell: tuple, vocab: Vocab) -> str:
    """Closed-world label function L(s,e,v) induced by K."""
    for b in spec_by_pair.get((effect, asset), []):
        if contains(b.phi, cell, vocab):
            return b.label
    return "0"


def index_spec(spec: list[Binding]) -> dict:
    out = {}
    for b in spec:
        out.setdefault((b.effect, b.asset), []).append(b)
    return out
