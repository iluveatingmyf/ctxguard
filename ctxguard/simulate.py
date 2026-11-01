"""Scenario generator for the runtime evaluation (Sec. VI-B).

A scenario = (true world, HA observation snapshot, triggered rule, fault).
The WORLD is sampled in reference facts (the v0 factor semantics); the
OBSERVATION is produced by a hand-written sensor model per entity type that is
independent of Γ (Γ is LLM-generated), so grounding is tested, not assumed.
Gold dispositions are labelled by a human from the narrative of the true world.

Types (per rule):
  random_rho    world uniformly sampled with the target zone inside [[rho]]
  targeted_A    inside [[rho]] ∩ [[phi]] of a '-' binding of Eff(a)   (gap A)
  b1_forged     world outside [[rho]]; the trigger entity is forged     (gap B1)
  b2_ambiguous  rho's factor set to an alternative reading; trigger fires
                naturally from the world                                (gap B2)
  missing       random_rho world with 30% of entities unavailable
"""
from __future__ import annotations

import copy
import csv
import random

from .common import Ctx, dump_yaml, load_yaml
from .layout import Layout
from .model import contains, normalize

GLOBAL_F = ["time_period", "home_occupancy", "active_hazard", "outdoor_condition"]
ZONE_F = ["zone_occupant", "vulnerable_present", "indoor_air", "indoor_thermal"]
HAZ_AREA = {"fire": "kitchen", "cooking_fumes": "kitchen", "gas_leak": "kitchen",
            "water_leak": "bathroom", "intrusion": "entry"}
ZH = {"day": "白天", "night": "夜间", "occupied": "有人在家", "away": "全家外出", "none": "无",
      "cooking_fumes": "正常做饭产生的油烟/CO", "fire": "真实火灾", "gas_leak": "燃气泄漏", "water_leak": "漏水",
      "intrusion": "有人闯入", "benign": "温和", "cold": "寒冷", "hot": "炎热", "rain_storm": "下雨/暴风",
      "polluted": "空气污染", "empty": "无人", "awake": "有人醒着", "asleep": "有人在睡觉", "yes": "是", "no": "否",
      "good": "良好", "poor": "差", "comfortable": "舒适"}


def zh(v):
    return ZH.get(v, v)


class World:
    def __init__(self, g: dict, zones: dict, time: str):
        self.g, self.zones, self.time = g, zones, time

    def situation(self, lay: Layout, area: str) -> dict:
        s = dict(self.g)
        s.update(self.zones[area])
        s["effect_zone"] = lay.zone_type(area)
        return s

    def to_json(self):
        return {"global": self.g, "zones": self.zones, "time": self.time}


def sample_world(ctx: Ctx, lay: Layout, area: str, pattern: dict, rng: random.Random, negate=False):
    """Sample a world whose situation at `area` is in [[pattern]] (or outside it if negate)."""
    vocab = ctx.vocab
    pat = dict(pattern)
    pat["effect_zone"] = [lay.zone_type(area)]
    try:
        zone_pat = normalize({"effect_zone": [lay.zone_type(area)]}, vocab)
        p_norm = normalize(pat, vocab)
    except ValueError:
        return None
    pool = [c for c in vocab.cells_in(zone_pat) if contains(p_norm, c, vocab) != negate]
    if not pool:
        return None
    cell = vocab.cell_dict(rng.choice(pool))
    g = {k: cell[k] for k in GLOBAL_F if k in cell}
    zones = {}
    for a in lay.indoor_areas():
        if a["id"] == area:
            zones[a["id"]] = {k: cell[k] for k in ZONE_F if k in cell}
        else:
            occ = "empty" if g.get("home_occupancy") == "away" else rng.choice(["empty", "empty", "awake", "asleep"])
            zones[a["id"]] = {"zone_occupant": occ, "vulnerable_present": rng.choice(["no", "no", "yes"]),
                              "indoor_air": rng.choice(["good", "good", "poor"]),
                              "indoor_thermal": rng.choice(["comfortable", "comfortable", "cold", "hot"])}
    if g.get("time_period") == "night":
        h = rng.choice([23, 0, 1, 2, 3, 4, 5])
    else:
        h = rng.randint(9, 17)
    return World(g, zones, f"{h:02d}:{rng.randint(0, 59):02d}")


# ------------------------------------------------------------------ sensor model
def simulate_obs(lay: Layout, w: World, rng: random.Random) -> dict:
    g, ents = w.g, {}
    haz = g.get("active_hazard", "none")
    hz_type = HAZ_AREA.get(haz)
    for e in lay.entities.values():
        eid, dom, dc, kind, area = e["id"], e["domain"], e.get("device_class"), e.get("kind"), e.get("area")
        z = w.zones.get(area, {})
        occ = z.get("zone_occupant", "empty")
        in_haz = hz_type is not None and lay.zone_type(area) == hz_type
        st, mins = "off", rng.randint(20, 300)
        if dom == "person":
            vul = any(r.get("entity") == eid and r.get("vulnerable") for r in lay.residents)
            st = "not_home" if g.get("home_occupancy") == "away" else ("home" if (vul or rng.random() < 0.8) else "not_home")
            persons = [x["id"] for x in lay.entities.values() if x["domain"] == "person"]
            if g.get("home_occupancy") == "occupied" and eid == persons[-1] and \
                    all(ents.get(x, {}).get("state") == "not_home" for x in persons[:-1]):
                st = "home"                   # occupied => at least one resident is home
        elif kind == "room_presence":
            where = [a for a, zz in w.zones.items() if zz.get("vulnerable_present") == "yes" and zz.get("zone_occupant") != "empty"]
            st = where[0] if where else ("not_home" if g.get("home_occupancy") == "away" else rng.choice([a for a in w.zones]))
        elif dc == "motion":
            if occ == "awake" or (haz == "intrusion" and lay.zone_type(area) == "entry"):
                st, mins = "on", rng.randint(0, 3)
            elif occ == "asleep":
                st, mins = ("on", 1) if rng.random() < 0.05 else ("off", rng.randint(40, 300))
        elif kind == "bed":
            st = "on" if occ == "asleep" else "off"
        elif dc == "carbon_dioxide":
            st = str(rng.randint(1100, 2000) if z.get("indoor_air") == "poor" else rng.randint(450, 850))
        elif dc == "humidity":
            st = str(rng.randint(76, 95) if z.get("indoor_air") == "poor" else rng.randint(35, 60))
        elif dc == "temperature" and area == "outdoor":
            oc = g.get("outdoor_condition")
            st = str({"cold": rng.randint(-8, 4), "hot": rng.randint(32, 38)}.get(oc, rng.randint(12, 24)))
        elif dc == "temperature":
            t = z.get("indoor_thermal", "comfortable")
            st = str({"cold": rng.randint(10, 16), "hot": rng.randint(29, 33)}.get(t, rng.randint(20, 25)))
            if haz == "fire" and in_haz:
                st = str(rng.randint(45, 70))
        elif dc == "aqi":
            st = str(rng.randint(160, 300) if g.get("outdoor_condition") == "polluted" else rng.randint(15, 60))
        elif dom == "weather":
            st = rng.choice(["pouring", "lightning-rainy"]) if g.get("outdoor_condition") == "rain_storm" else rng.choice(["sunny", "cloudy", "partlycloudy"])
        elif dom == "sun":
            st = "below_horizon" if g.get("time_period") == "night" else "above_horizon"
        elif dc == "smoke":
            st = "on" if haz == "fire" or (haz == "cooking_fumes" and lay.zone_type(area) == "kitchen") else "off"
        elif dc == "carbon_monoxide":
            st = str(rng.randint(60, 200) if haz == "fire" else rng.randint(31, 45) if haz == "cooking_fumes" else rng.randint(0, 5))
        elif dc == "gas":
            st = "on" if haz == "gas_leak" else "off"
        elif dc == "moisture":
            st = "on" if haz == "water_leak" and in_haz else "off"
        elif dc == "door":
            st = "on" if haz == "intrusion" else "off"
        elif dc == "power":
            st = str(rng.randint(800, 2000) if haz in ("cooking_fumes", "fire") else 0)
        elif dom == "alarm_control_panel":
            st = "triggered" if haz == "intrusion" else ("armed_away" if g.get("home_occupancy") == "away" else "disarmed")
        elif kind in ("doorbell",):
            st = "off"
        elif kind == "appliance_status":
            st = "idle"
        elif dom == "light":
            st = "on" if occ == "awake" and g.get("time_period") == "night" else "off"
        elif kind == "stove":
            st = "on" if haz in ("cooking_fumes", "fire") else "off"
        elif dom in ("cover",):
            st = "closed"
        elif dom == "lock":
            st = "locked"
        elif dom == "valve":
            st = "open"
        elif dom == "climate":
            st = "off"
        else:
            st = "off"
        ents[eid] = {"state": st, "minutes": mins}
    return {"time": w.time, "entities": ents}


# ------------------------------------------------------------------ triggers
def trigger_ok(rule, obs) -> bool:
    for t in rule["trigger"]:
        plat = t.get("platform")
        if plat == "time":
            return obs["time"] == t["at"][:5]
        if plat == "sun":
            return True
        e = obs["entities"].get(t.get("entity_id"), {})
        st = e.get("state")
        if plat == "state":
            ok = str(st) == str(t.get("to"))
            if ok and t.get("for"):
                ok = e.get("minutes", 0) >= t["for"].get("minutes", 0)
            return ok
        if plat == "numeric_state":
            try:
                x = float(st)
            except (TypeError, ValueError):
                return False
            return (t.get("above") is None or x > t["above"]) and (t.get("below") is None or x < t["below"])
    return False


def force_trigger(rule, obs):
    t = rule["trigger"][0]
    plat = t.get("platform")
    if plat == "time":
        obs["time"] = t["at"][:5]
    elif plat == "state":
        obs["entities"][t["entity_id"]] = {"state": str(t["to"]), "minutes": (t.get("for") or {}).get("minutes", 0) + 1}
    elif plat == "numeric_state":
        v = (t["above"] + 50) if t.get("above") is not None else (t["below"] - 3)
        obs["entities"][t["entity_id"]] = {"state": str(v), "minutes": 1}
    return obs


# ------------------------------------------------------------------ generation
def narrative(lay, w: World, area, rule, kind, forged, missing):
    g = w.g
    z = w.zones[area]
    s = (f"时间：{zh(g.get('time_period'))}（{w.time}）。全屋：{zh(g.get('home_occupancy'))}。"
         f"当前危险：{zh(g.get('active_hazard'))}。室外：{zh(g.get('outdoor_condition'))}。"
         f"动作作用的房间 {area}（{lay.zone_type(area)}）：人员 {zh(z.get('zone_occupant'))}；"
         f"老人/儿童/宠物在场 {zh(z.get('vulnerable_present'))}；空气 {zh(z.get('indoor_air'))}；温度 {zh(z.get('indoor_thermal'))}。")
    others = [f"{a}:{zh(zz.get('zone_occupant'))}" for a, zz in w.zones.items() if a != area and zz.get("zone_occupant") != "empty"]
    if others:
        s += " 其他房间有人：" + "，".join(others) + "。"
    if forged:
        s += " 【注意：触发信号是攻击者伪造的，上面才是真实情况】"
    if missing:
        s += f" 【以下传感器当前不可用：{', '.join(missing[:8])}{' 等' if len(missing) > 8 else ''}】"
    return s


def run_sim(ctx: Ctx, layout_id: str, intents: dict | None = None) -> dict:
    lay = Layout(layout_id, ctx.actions)
    vocab = ctx.vocab
    intents = intents or load_yaml(ctx.outp("m6", f"intents_{layout_id}.yaml"))["intents"]
    if not set(GLOBAL_F + ZONE_F) <= set(vocab.names):
        print("[sim] WARNING: vocabulary lacks the reference factor names "
              f"{GLOBAL_F + ZONE_F}; keep these names for factors with the same meaning, "
              "otherwise no scenarios can be generated.")
    spec = load_yaml(ctx.spec_path())["bindings"]
    lib = ctx.effects()
    per = ctx.cfg["eval"]["scenarios_per_rule"]
    rng = ctx.rng(f"sim:{layout_id}")
    scen = []
    v0 = set(GLOBAL_F + ZONE_F) <= set(vocab.names)
    for r in lay.rules:
        it = intents.get(r["id"])
        if not it or not it.get("valid", True) or not it.get("area"):
            continue
        area, rho = it["area"], it.get("rho") or {}
        effs = set(lib["eff"].get(it["action_type"], []))
        minus = [b for b in spec if b["effect"] in effs and b["label"] == "-"]
        plan = []
        plan += [("random_rho", rho, False)] * per.get("random_rho", per.get("benign", 2))
        for _ in range(per.get("gap_a", 2)):
            if minus and v0:
                b = rng.choice(minus)
                merged = dict(rho)
                for n, vs in b["pattern"].items():
                    merged[n] = sorted(set(merged.get(n, vs)) & set(vs)) if n in merged else vs
                plan.append(("targeted_A", merged, False))
        if rho and r["trigger"][0].get("platform") not in ("time", "sun"):   # clock events cannot be forged
            plan += [("b1_forged", rho, True)] * per.get("b1_forged", 1)
        alt = it.get("alternative_readings") or {}
        if alt and v0:
            n, vs = next(iter(alt.items()))
            p2 = dict(rho)
            p2[n] = vs
            plan += [("b2_ambiguous", p2, False)] * per.get("b2_ambiguous", 1)
        plan += [("missing", rho, False)] * per.get("missing", 1)
        for kind, pat, neg in plan:
            if not v0:
                break
            w = None
            for _ in range(20):
                w = sample_world(ctx, lay, area, pat, rng, negate=neg)
                if w is None:
                    break
                obs = simulate_obs(lay, w, rng)
                if kind == "b1_forged" or trigger_ok(r, obs) or r["trigger"][0].get("platform") in ("time", "sun") \
                        or (lay.entities.get(r["trigger"][0].get("entity_id"), {}).get("kind") in ("doorbell", "appliance_status")) \
                        or r["trigger"][0].get("entity_id", "").startswith("person."):
                    break
                w = None
            if w is None:
                continue
            natural = trigger_ok(r, obs)
            obs = force_trigger(r, obs) if not natural else obs
            forged = kind == "b1_forged"
            missing = []
            if kind == "missing":
                trig = set(lay.trigger_entities(r))
                cand = [e for e in obs["entities"] if e not in trig]
                missing = rng.sample(cand, int(0.3 * len(cand)))
                for e in missing:
                    obs["entities"][e] = {"state": "unavailable", "minutes": 0}
            sid = f"{r['id']}_{kind}_{len([s for s in scen if s['rule'] == r['id']])}"
            scen.append({"id": sid, "layout": lay.id, "rule": r["id"], "type": kind, "area": area,
                         "world": w.to_json(), "obs": obs, "forged": forged,
                         "trigger_forced": (not natural) and not forged, "missing": missing,
                         "narrative": narrative(lay, w, area, r, kind, forged, missing)})
    dump_yaml(scen, ctx.outp("sim", f"scenarios_{lay.id}.yaml"))
    # annotation sheet
    path = ctx.outp("annotate", f"scenario_gold_{lay.id}.csv")
    cands = lay.candidate_actions()
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["scenario", "rule", "rule_alias", "action", "true_situation", "possible_substitutes_same_room",
                     "OK_execute(1/0)", "OK_substitute(write action or 0)", "OK_defer(1/0)", "OK_block(1/0)", "OK_ask(1/0)", "note"])
        for s in scen:
            r = next(x for x in lay.rules if x["id"] == s["rule"])
            subs = sorted({f"{c['type']}:{c['entity']}" for c in cands if c["area"] == s["area"]})
            wr.writerow([s["id"], s["rule"], r["alias"], r["action"][0]["service"] + " " + str((r["action"][0].get("target") or {}).get("entity_id")),
                         s["narrative"], "; ".join(subs), "", "", "", "", "", ""])
    return {"layout": lay.id, "scenarios": len(scen), "annotate": path}


def degrade(obs: dict, ratio: float, protect: set, rng: random.Random) -> dict:
    o = copy.deepcopy(obs)
    cand = [e for e in o["entities"] if e not in protect and o["entities"][e]["state"] != "unavailable"]
    for e in rng.sample(cand, int(ratio * len(cand))):
        o["entities"][e] = {"state": "unavailable", "minutes": 0}
    return o
