"""OFFLINE SMOKE-TEST BACKEND. Deterministic hand-written oracle that imitates
the LLM so the whole pipeline can be exercised without a model.
Results produced with backend=mock are NOT experimental results."""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from .common import load_yaml, p
from .model import Vocab, contains, normalize

EFFECTS = {
    "open_window": ["air_exchange:increase", "passage_opening:increase", "indoor_temperature:decrease", "indoor_temperature:increase", "sound:increase", "visibility:increase"],
    "close_window": ["air_exchange:decrease", "passage_opening:decrease"],
    "open_blind": ["illumination:increase", "visibility:increase"],
    "close_blind": ["illumination:decrease", "visibility:decrease"],
    "open_garage": ["passage_opening:increase", "mechanical_motion:increase"],
    "close_garage": ["passage_opening:decrease", "mechanical_motion:increase"],
    "unlock_door": ["passage_opening:increase"],
    "lock_door": ["passage_opening:decrease"],
    "heat_on": ["indoor_temperature:increase", "heat_source:increase", "power_supply:increase"],
    "cool_on": ["indoor_temperature:decrease", "power_supply:increase"],
    "hvac_off": ["power_supply:decrease"],
    "fan_on": ["indoor_temperature:decrease", "sound:increase", "power_supply:increase"],
    "exhaust_fan_on": ["air_exchange:increase", "indoor_humidity:decrease", "sound:increase"],
    "light_on": ["illumination:increase", "power_supply:increase"],
    "light_off": ["illumination:decrease"],
    "outlet_on": ["power_supply:increase"],
    "outlet_off": ["power_supply:decrease"],
    "stove_off": ["heat_source:decrease", "power_supply:decrease"],
    "water_valve_close": ["water_flow:decrease"],
    "water_valve_open": ["water_flow:increase"],
    "gas_valve_close": ["gas_flow:decrease", "heat_source:decrease"],
    "alarm_arm_away": ["security_monitoring:increase", "sound:increase"],
    "alarm_disarm": ["security_monitoring:decrease"],
    "siren_on": ["sound:increase"],
    "camera_on": ["sensing_capture:increase"],
    "media_play": ["sound:increase"],
    "tts_announce": ["sound:increase", "info_disclosure:increase"],
    "vacuum_start": ["mechanical_motion:increase", "sound:increase"],
    "humidifier_on": ["indoor_humidity:increase"],
}


@lru_cache(maxsize=1)
def V():
    return Vocab.load(p("data/vocab/factors_v0.yaml"))


def oracle(e, v, s):
    g = s.get
    if e == "passage_opening:increase":
        if v == "security" and (g("home_occupancy") == "away" or g("zone_occupant") == "asleep" or g("active_hazard") == "intrusion") \
                and g("active_hazard") not in ("fire", "gas_leak"):
            return "-"
        if v == "life_health" and g("active_hazard") in ("fire", "gas_leak"):
            return "+"
    if e == "air_exchange:increase":
        if v == "life_health":
            if g("outdoor_condition") == "polluted" or (g("outdoor_condition") == "cold" and g("vulnerable_present") == "yes"):
                return "-"
            if g("active_hazard") == "fire":
                return "-"
            if g("indoor_air") == "poor" or g("active_hazard") in ("gas_leak", "cooking_fumes"):
                return "+"
        if v == "comfort" and g("outdoor_condition") in ("cold", "hot", "rain_storm"):
            return "-"
        if v == "property" and g("outdoor_condition") == "rain_storm":
            return "-"
    if e == "power_supply:decrease" and v == "life_health" and g("vulnerable_present") == "yes" and g("indoor_thermal") == "cold":
        return "-"
    if e == "sound:increase" and v == "comfort" and g("zone_occupant") == "asleep":
        return "-"
    if e == "sensing_capture:increase" and v == "privacy" and g("home_occupancy") == "occupied":
        return "-"
    if e == "mechanical_motion:increase" and v == "life_health" and g("vulnerable_present") == "yes":
        return "-"
    if e == "water_flow:decrease" and v == "property" and g("active_hazard") == "water_leak":
        return "+"
    if e == "gas_flow:decrease" and v == "life_health" and g("active_hazard") in ("gas_leak", "fire"):
        return "+"
    if e == "security_monitoring:increase" and v == "security" and g("home_occupancy") == "away":
        return "+"
    if e == "indoor_temperature:decrease" and v == "life_health" and g("vulnerable_present") == "yes" and g("indoor_thermal") == "cold":
        return "-"
    if e == "indoor_temperature:decrease" and v == "comfort" and g("indoor_thermal") == "hot":
        return "+"
    if e == "indoor_temperature:increase" and v == "life_health" and g("vulnerable_present") == "yes" and g("indoor_thermal") == "hot":
        return "-"
    return "0"


def region_answer(ctx):
    vocab = V()
    phi = normalize(ctx["phi"], vocab)
    cells = vocab.cells_in(phi)
    labs = {c: oracle(ctx["effect"], ctx["asset"], vocab.cell_dict(c)) for c in cells}
    if len(set(labs.values())) == 1:
        lab = next(iter(labs.values()))
        return {"consistent": True, "label": lab, "harm": "mock mechanism" if lab == "-" else "", "reason": "mock"}
    best = None
    for n in ctx["free"]:
        i = vocab.index[n]
        by = defaultdict(set)
        for c, l in labs.items():
            by[c[i]].add(l)
        pure = sum(1 for s in by.values() if len(s) == 1)
        if best is None or pure > best[0]:
            groups = defaultdict(list)
            for val, s in by.items():
                groups[tuple(sorted(s))].append(val)
            best = (pure, n, list(groups.values()))
    _, n, groups = best
    if len(groups) < 2:
        groups = [[v] for v in sorted({c[vocab.index[n]] for c in cells})]
    return {"consistent": False, "split_factor": n, "value_groups": groups, "reason": "mock"}


def grounding_answer(ctx):
    lay = None
    import glob
    for f in glob.glob(p("data/layouts", f"{ctx['layout']}_*.yaml")):
        lay = load_yaml(f)
    ents = lay["entities"]
    area = (ctx.get("area") or {}).get("id")
    by = lambda **kw: [e["id"] for e in ents if all(e.get(k) == v for k, v in kw.items())]
    f = ctx["factor"]
    rules, default = [], None
    if f == "time_period":
        rules, default = [{"value": "night", "when": {"time": ["22:30", "06:30"]}}], "day"
    elif f == "home_occupancy":
        ps = by(domain="person")
        rules = [{"value": "occupied", "when": {"any": [{"state": x, "eq": "home"} for x in ps]}}]
        default = "away"
    elif f == "active_hazard":
        sm, co, gas, lk, st = by(device_class="smoke"), by(device_class="carbon_monoxide"), by(device_class="gas"), by(device_class="moisture"), by(kind="stove")
        if gas:
            rules.append({"value": "gas_leak", "when": {"any": [{"state": x, "eq": "on"} for x in gas]}})
        if sm and st:
            rules.append({"value": "cooking_fumes", "when": {"all": [{"any": [{"state": x, "eq": "on"} for x in sm]}, {"state": st[0], "eq": "on"},
                                                                       *([{"num": co[0], "lt": 50}] if co else [])]}})
        if sm:
            rules.append({"value": "fire", "when": {"any": [{"state": x, "eq": "on"} for x in sm]}})
        if lk:
            rules.append({"value": "water_leak", "when": {"any": [{"state": x, "eq": "on"} for x in lk]}})
        default = "none"
    elif f == "outdoor_condition":
        t = [e["id"] for e in ents if e.get("device_class") == "temperature" and e.get("area") == "outdoor"]
        rules = [{"value": "rain_storm", "when": {"state": "weather.home", "in": ["pouring", "lightning-rainy"]}}]
        if t:
            rules += [{"value": "cold", "when": {"num": t[0], "lt": 5}}, {"value": "hot", "when": {"num": t[0], "gt": 31}}]
        aq = by(device_class="aqi")
        if aq:
            rules.append({"value": "polluted", "when": {"num": aq[0], "gt": 150}})
        default = "benign"
    elif f == "zone_occupant":
        mo = [e["id"] for e in ents if e.get("device_class") == "motion" and e.get("area") == area]
        bed = [e["id"] for e in ents if e.get("kind") == "bed" and e.get("area") == area]
        if bed:
            rules.append({"value": "asleep", "when": {"state": bed[0], "eq": "on"}})
        if mo:
            rules.append({"value": "awake", "when": {"state": mo[0], "eq": "on"}})
            if not bed:
                return {"groundable": True, "rules": rules, "default": None}
            default = "empty"
        if not rules:
            return {"groundable": False, "rules": [], "default": None}
    elif f == "vulnerable_present":
        rp = by(kind="room_presence")
        if rp:
            rules = [{"value": "yes", "when": {"state": rp[0], "eq": area}}]
            if area == "kids_room":
                rules.append({"value": "yes", "when": {"state": "binary_sensor.kids_motion", "eq": "on"}})
            default = None if area == "kids_room" else "no"
        else:
            return {"groundable": False, "rules": [], "default": None}
    elif f == "indoor_air":
        c = [e["id"] for e in ents if e.get("device_class") in ("carbon_dioxide", "humidity") and e.get("area") == area]
        if not c:
            return {"groundable": False, "rules": [], "default": None}
        thr = 1000 if "co2" in c[0] else 75
        rules, default = [{"value": "poor", "when": {"num": c[0], "gt": thr}}], "good"
    elif f == "indoor_thermal":
        t = [e["id"] for e in ents if e.get("device_class") == "temperature" and e.get("area") == area]
        if not t:
            return {"groundable": False, "rules": [], "default": None}
        rules, default = [{"value": "cold", "when": {"num": t[0], "lt": 17}}, {"value": "hot", "when": {"num": t[0], "gt": 28}}], "comfortable"
    return {"groundable": True, "rules": rules, "default": default}


def intent_answer(ctx):
    r = ctx["rule"]
    t = r["trigger"][0]
    e = t.get("entity_id", "")
    rho, alt = {}, {}
    if "co2" in e or "humidity" in e:
        rho = {"indoor_air": ["poor"]}
    elif "smoke" in e:
        rho, alt = {"active_hazard": ["fire"]}, {"active_hazard": ["cooking_fumes"]}
    elif "_co" in e:
        rho, alt = {"active_hazard": ["fire", "gas_leak"]}, {"active_hazard": ["cooking_fumes"]}
    elif "leak" in e:
        rho = {"active_hazard": ["water_leak"]}
    elif "gas" in e:
        rho = {"active_hazard": ["gas_leak"]}
    elif e.startswith("person.") and t.get("to") == "not_home":
        rho = {"home_occupancy": ["away"]}
    elif e.startswith("person.") and t.get("to") == "home":
        rho = {"home_occupancy": ["occupied"]}
    elif "outdoor_temperature" in e:
        rho = {"outdoor_condition": ["cold"]}
    elif "temperature" in e:
        rho = {"indoor_thermal": ["hot"]}
    elif t.get("platform") == "time":
        rho = {"time_period": ["day"]}
    effs = ctx["effects"]
    pref = ["air_exchange:increase", "passage_opening:increase", "passage_opening:decrease", "water_flow:decrease",
            "gas_flow:decrease", "security_monitoring:increase", "indoor_temperature:decrease", "indoor_temperature:increase",
            "illumination:increase", "illumination:decrease", "power_supply:decrease", "mechanical_motion:increase",
            "sensing_capture:increase", "sound:increase", "heat_source:decrease"]
    ie = next((x for x in pref if x in effs), effs[0] if effs else None)
    asset = {"air_exchange:increase": "life_health", "passage_opening:increase": "life_health" if "smoke" in e else "comfort",
             "water_flow:decrease": "property", "gas_flow:decrease": "life_health", "security_monitoring:increase": "security",
             "passage_opening:decrease": "security", "power_supply:decrease": "resource", "heat_source:decrease": "life_health"}.get(ie, "comfort")
    return {"rho": rho, "intended_effect": ie, "served_asset": asset, "alternative_readings": alt, "confidence": 0.5}


def mock_answer(task, ctx, seed):
    if task == "effects":
        return {"effects": [{"channel": x.split(":")[0], "direction": x.split(":")[1], "precondition": "always",
                             "requires_extra_device": False}
                            for x in EFFECTS.get(ctx["action"]["id"], [])]}
    if task in ("point_label", "point_label_audit"):
        lab = oracle(ctx["effect"], ctx["asset"], ctx["cell"])
        return {"label": lab, "harm": "mock mechanism" if lab == "-" else ""}
    if task == "relevance":
        vocab = V()
        hit = any(oracle(ctx["effect"], ctx["asset"], vocab.cell_dict(c)) != "0" for c in vocab.sigma()[::7])
        return {"can_affect": hit, "reason": "mock"}
    if task == "region":
        return region_answer(ctx)
    if task == "refute":
        return {"counterexamples": []}
    if task == "direct_constraints":
        if ctx["effect"] == "passage_opening:increase" and ctx["asset"] == "security":
            return {"items": [{"pattern": {"home_occupancy": ["away"]}, "label": "-", "harm": "entry"}]}
        return {"items": []}
    if task == "grounding":
        return grounding_answer(ctx)
    if task == "intent":
        return intent_answer(ctx)
    if task == "llm_adjudicate":
        return {"disposition": "execute", "reason": "mock"}
    if task == "extract_predicates":
        t = ctx["property"]["text"]
        parts = re.split(r"\b(?:when|if|while|and)\b", t, flags=re.I)[1:]
        return {"conditions": [x.strip(" .,") for x in parts if x.strip(" .,")] or [t]}
    if task == "induce_dims":
        if ctx["existing"]:
            return {"dimensions": []}
        idx = lambda kws: [i for i, c in enumerate(ctx["conditions"]) if any(k in c.lower() for k in kws)]
        return {"dimensions": [
            {"id": "occupancy", "desc": "whether people are home", "scope": "global", "n_variables": 1, "covers": idx(["home", "nobody"])},
            {"id": "occupant_activity", "desc": "what people in the room do", "scope": "zone", "n_variables": 1, "covers": idx(["sleep", "motion"])},
            {"id": "hazard", "desc": "dangerous events", "scope": "global", "n_variables": 1, "covers": idx(["smoke", "leak", "co "])}]}
    if task == "expand_values":
        names = list(ctx["factor"]["values"])
        return {"candidates": [{"value": "cooking", "semantics": "someone is cooking", "closest_existing": names[0],
                                "effect": "air_exchange:increase", "asset": "life_health", "why": "fumes"}]}
    if task == "contrast_label":
        return {"label": "+" if ctx["value"] == "cooking" else "0", "harm": ""}
    if task == "classify_condition":
        t = ctx["condition"].lower()
        act = "occupant_activity" if "occupant_activity" in ctx.get("dims", []) else "occupant_state"
        slot = ("hazard" if any(w in t for w in ("smoke", "co ", "leak", "gas", "intruder")) else
                "occupancy" if any(w in t for w in ("home", "present", "nobody")) else
                act if any(w in t for w in ("sleep", "motion")) else
                "indoor_environment" if any(w in t for w in ("temperature", "humid", "air")) else
                "time" if any(w in t for w in ("night", "day")) else "device_state")
        return {"slot": slot, "situation_statement": ctx["condition"]}
    if task == "slot_values":
        st = ctx["statements"]
        return {"factors": [{"name": ctx["slot"]["id"] + "_f", "semantics": ctx["slot"]["desc"],
                             "values": [{"value": "none", "semantics": "baseline"}, {"value": "present", "semantics": "holds"}]}],
                "assignment": [{"statement": i, "factor": ctx["slot"]["id"] + "_f", "value": "present"} for i in range(len(st))]}
    if task == "name_factor":
        w = re.sub(r"[^a-z ]", "", ctx["conditions"][0].lower()).split()
        return {"name": "_".join(w[:2]) or "factor", "scope": "global", "semantics": ctx["conditions"][0],
                "values": [{"value": "yes", "semantics": "holds"}, {"value": "no", "semantics": "does not hold"}]}
    if task == "map_property":
        t = ctx["property"]["text"].lower()
        if "door" in t and ("away" in t or "nobody" in t):
            return {"expressible": True, "effects": ["passage_opening:increase"], "asset": "security", "label": "-",
                    "pattern": {"home_occupancy": ["away"]}}
        if "window" in t and "sleep" in t:
            return {"expressible": True, "effects": ["passage_opening:increase"], "asset": "security", "label": "-",
                    "pattern": {"zone_occupant": ["asleep"]}}
        return {"expressible": False, "missing": "mock: not parsed"}
    raise KeyError(task)
