from __future__ import annotations

import json
import os
import random

import yaml

from .llm import LLM
from .model import Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    return os.path.join(ROOT, *parts)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=120)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Ctx:
    """Everything a step needs: config, paths, vocabularies, LLM."""

    def __init__(self, config_path=None, backend=None):
        self.cfg = load_yaml(config_path or p("config.yaml"))
        if backend:
            self.cfg["llm"]["backend"] = backend
        self.out = p(self.cfg["paths"]["out"])
        os.makedirs(self.out, exist_ok=True)
        self.seed = self.cfg["method"]["seed"]
        mock = None
        if self.cfg["llm"]["backend"] == "mock":
            from .mock import mock_answer
            mock = mock_answer
        self.llm = LLM(self.cfg, self.out, mock=mock)
        self.channels = load_yaml(p("data/vocab/channels.yaml"))
        self.assets = load_yaml(p("data/vocab/assets.yaml"))
        self.actions = load_yaml(p("data/actions/actions.yaml"))["actions"]
        self.vocab = Vocab.load(p(self.cfg["paths"]["factors"]))

    def rng(self, salt=""):
        return random.Random(f"{self.seed}:{salt}")

    def outp(self, *parts):
        path = os.path.join(self.out, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @property
    def asset_list(self):
        return self.assets["assets"]

    @property
    def priority(self):
        return self.assets["priority"]

    def channels_desc(self):
        return "\n".join(f"- {c['id']}: {c['desc']}" for c in self.channels["channels"])

    # effect library produced by M1
    def effects(self):
        return load_yaml(self.outp("m1", "effects.yaml"))

    def spec_path(self):
        return self.outp("m3", "spec.yaml")
