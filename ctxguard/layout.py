"""Deployment description: areas, entities and automation rules of one home."""
from __future__ import annotations

import glob
import os

import yaml

from .common import load_yaml, p


class Layout:
    def __init__(self, layout_id: str, actions: list[dict]):
        lp = glob.glob(p("data/layouts", f"{layout_id}_*.yaml")) + glob.glob(p("data/layouts", f"{layout_id}.yaml"))
        rp = glob.glob(p("data/rules", f"{layout_id}_rules.yaml"))
        if not lp or not rp:
            raise FileNotFoundError(f"layout or rules for {layout_id} not found in data/layouts, data/rules")
        d = load_yaml(lp[0])
        self.id = d["layout_id"]
        self.name = d.get("name", "")
        self.areas = {a["id"]: a for a in d["areas"]}
        self.entities = {e["id"]: e for e in d["entities"]}
        self.residents = d.get("residents", [])
        self.rules = load_yaml(rp[0])
        self.actions = actions

    # ------------------------------------------------------------ helpers
    def indoor_areas(self):
        return [a for a in self.areas.values() if a["zone_type"] != "outdoor"]

    def zone_type(self, area_id):
        return self.areas[area_id]["zone_type"] if area_id in self.areas else None

    def describe_entities(self, only_domains=None) -> str:
        lines = []
        for e in self.entities.values():
            if only_domains and e["domain"] not in only_domains:
                continue
            extra = ", ".join(f"{k}={e[k]}" for k in ("device_class", "kind", "unit", "note") if e.get(k))
            lines.append(f"- {e['id']} (domain={e['domain']}{', ' + extra if extra else ''}, area={e.get('area')})")
        return "\n".join(lines)

    def rule_yaml(self, rule) -> str:
        return yaml.safe_dump(rule, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------ action typing
    def type_call(self, call: dict):
        """Map an HA service call to (action_type, entity_id, area_id)."""
        domain, service = call["service"].split(".", 1)
        eid = (call.get("target") or {}).get("entity_id") or call.get("entity_id")
        data = call.get("data") or {}
        ent = self.entities.get(eid)
        area_ent = ent
        if ent is None and data.get("media_player_entity_id"):
            area_ent = self.entities.get(data["media_player_entity_id"])
        for a in self.actions:
            m = a["match"]
            if m["domain"] != domain or m["service"] != service:
                continue
            if "device_class" in m and (ent is None or ent.get("device_class") not in m["device_class"]):
                continue
            if "kind" in m and (ent is None or ent.get("kind") not in m["kind"]):
                continue
            if any(data.get(k) != v for k, v in (m.get("data") or {}).items()):
                continue
            return a["id"], eid, (area_ent or {}).get("area")
        return None, eid, (area_ent or {}).get("area")

    def rule_actions(self, rule):
        return [self.type_call(c) for c in rule["action"]]

    def trigger_entities(self, rule):
        return [t.get("entity_id") for t in rule["trigger"] if t.get("entity_id")]

    def candidate_actions(self):
        """Every (action_type, entity, area) executable in this home."""
        out = []
        for e in self.entities.values():
            for a in self.actions:
                m = a["match"]
                if m["domain"] != e["domain"]:
                    continue
                if "device_class" in m and e.get("device_class") not in m["device_class"]:
                    continue
                if "kind" in m and e.get("kind") not in m["kind"]:
                    continue
                call = {"service": f"{m['domain']}.{m['service']}", "target": {"entity_id": e["id"]}}
                if m.get("data"):
                    call["data"] = dict(m["data"])
                out.append({"type": a["id"], "entity": e["id"], "area": e.get("area"), "call": call})
        return out


def list_layouts():
    return sorted(os.path.basename(f).split("_")[0].replace(".yaml", "")
                  for f in glob.glob(p("data/layouts", "*.yaml")))
