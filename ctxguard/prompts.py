"""All prompts and output schemas in one place (cite this file in the paper's
appendix). Each prompt = SYSTEM (role + definitions + hard constraints) +
USER (the instance). Outputs are forced to JSON by the schema (ollama `format`).
These replace the hand-pasted "task cards": the shared preamble is PREAMBLE.
"""
from __future__ import annotations

PREAMBLE = """You are a smart-home safety analyst. Use ONLY the following terms.
- Action: a device service call. Effect: a directional change an action causes on one physical channel.
- Protected asset (fixed set): life_health, security, property, privacy, resource, comfort.
- Context factor: a finite variable describing the situation; you may use ONLY the given factors and values. Never invent factors or values.
- Abstract situation: one value for every factor. Condition pattern: allowed value sets for some factors (others = any).
- Label of an effect on an asset in a situation:
  "+" the effect maintains or improves the asset;
  "-" the effect causes UNACCEPTABLE harm to the asset;
  "0" no relevant influence, or influence within acceptable bounds.
- Before giving "-" you must state the harm mechanism (how the effect hurts the asset in that situation). If you cannot state a concrete mechanism, the label is not "-".
Answer with JSON only."""


def S(props: dict, req: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": req}


STR = {"type": "string"}
BOOL = {"type": "boolean"}
LABEL = {"type": "string", "enum": ["+", "-", "0"]}
STRLIST = {"type": "array", "items": STR}
PATTERN = {"type": "object", "additionalProperties": STRLIST}

# ---------------------------------------------------------------- M1 effects
M1_SYSTEM = """You are a home-automation engineer describing the PHYSICAL consequences of device actions.
Vocabulary: an EFFECT is a directional change (increase|decrease) that the action itself causes on ONE physical channel.
Use ONLY the channel ids listed below. Channels are physical or informational quantities, not values or goals:
do not output words such as privacy, comfort, safety, temperature, humidity or illuminance as channels.
Rules:
1. Include side effects that physically follow from the device operating in a typical home
   (a motor makes noise, an open window lets air and sound in, a heater adds heat).
2. Mark requires_extra_device=true for any effect that needs an additional device, an unusual installation or an
   unknown load (e.g. "if a camera is installed", "if the outlet powers a heater"). Such effects are discarded.
3. If the direction depends on the situation (e.g. a window warms OR cools the room), list BOTH directions.
Answer with JSON only."""


def effects_prompt(action: dict, channels_desc: str):
    """M1 uses a dedicated system prompt (no assets, no labels, no context factors)."""
    sys = M1_SYSTEM + "\n\nPhysical channels (closed set; use these ids only):\n" + channels_desc
    user = (f"Action type: {action['id']}  (HA service match: {action['match']}; scope: {action['scope']}).\n"
            "List every effect of this action. For each effect give channel, direction (increase|decrease), "
            "requires_extra_device (true|false) and a precondition in words (or 'always').")
    schema = S({"effects": {"type": "array", "items": S(
        {"channel": STR, "direction": {"type": "string", "enum": ["increase", "decrease"]},
         "requires_extra_device": BOOL, "precondition": STR, "why": STR},
        ["channel", "direction", "requires_extra_device", "precondition"])}}, ["effects"])
    return sys, user, schema


# ---------------------------------------------------------------- M2 vocabulary
def extract_predicates_prompt(prop_text: str):
    sys = PREAMBLE + ("\n\nTask: extract the SITUATION CONDITIONS under which a smart-home safety/preference property applies. "
                      "A situation condition is one objective fact about the world (people, time, environment, hazards), "
                      "NOT a device name and NOT the action itself.")
    user = f"Property: {prop_text}\nReturn each atomic condition as a short device-agnostic sentence."
    schema = S({"conditions": STRLIST}, ["conditions"])
    return sys, user, schema


def name_factor_prompt(predicates: list[str]):
    sys = PREAMBLE + ("\n\nTask: the following situation conditions were clustered as describing ONE dimension of the situation. "
                      "Propose one context factor: a snake_case name, a one-sentence semantics, and 2-6 mutually exclusive values "
                      "that keep the distinctions needed for safety judgments (merge distinctions that never change a judgment). "
                      "Each value has a one-sentence decidable semantics. Say whether it is global (whole home) or zone (the room where the effect acts).")
    user = "Conditions:\n" + "\n".join(f"- {p}" for p in predicates)
    schema = S({"name": STR, "scope": {"type": "string", "enum": ["global", "zone"]}, "semantics": STR,
                "values": {"type": "array", "items": S({"value": STR, "semantics": STR}, ["value", "semantics"])}},
               ["name", "scope", "semantics", "values"])
    return sys, user, schema


def classify_condition_prompt(condition: str, slots_desc: str):
    """M2 step 2: place one extracted condition into one slot of the situation schema."""
    sys = PREAMBLE + ("\n\nTask: assign a condition taken from a smart-home property to ONE slot of the situation schema below. "
                      "Sensor detections describe the situation they indicate: 'smoke is detected' -> hazard, 'motion detected' -> "
                      "occupant_state, 'temperature above threshold' -> indoor_environment. Device or system settings ('heater is on', "
                      "'door is closed', 'home mode is away', 'alarm enabled') are device_state. Thresholds or times chosen by the user "
                      "are user_configuration. Also restate the condition as a device-agnostic fact about the world.\n\nSlots:\n" + slots_desc)
    user = f"Condition: {condition}"
    schema = S({"slot": STR, "situation_statement": STR, "reason": STR}, ["slot", "situation_statement"])
    return sys, user, schema


def slot_values_prompt(slot: dict, statements: list[str], max_factors: int):
    """M2 step 3: induce the factor(s) and their value set for one slot from corpus evidence."""
    sys = PREAMBLE + ("\n\nTask: the statements below all describe the SAME dimension of a home situation. Define at most "
                      f"{max_factors} context factor(s) for this dimension. A factor is ONE variable; its values are mutually exclusive "
                      "and together cover the dimension (include a baseline value such as none/normal/absent). Use 2-6 values; merge "
                      "distinctions that never change a safety judgment, keep distinctions that can (e.g. fire vs cooking smoke). "
                      "Each value gets a one-sentence decidable semantics. Then assign every statement to factor=value, or to null if "
                      "no value fits.")
    user = (f"Dimension: {slot['id']} - {slot['desc']} (scope: {slot['scope']})\nStatements:\n" +
            "\n".join(f"{i}. {s}" for i, s in enumerate(statements)))
    schema = S({"factors": {"type": "array", "items": S({"name": STR, "semantics": STR,
                                                         "values": {"type": "array", "items": S({"value": STR, "semantics": STR}, ["value", "semantics"])}},
                                                        ["name", "semantics", "values"])},
                "assignment": {"type": "array", "items": S({"statement": {"type": "integer"}, "factor": {"type": ["string", "null"]},
                                                            "value": {"type": ["string", "null"]}}, ["statement", "factor", "value"])}},
               ["factors", "assignment"])
    return sys, user, schema


def induce_dims_prompt(conditions: list[str], existing: list[dict]):
    """M2 step 2 (open coding): induce situation DIMENSIONS from corpus conditions."""
    sys = PREAMBLE + ("\n\nTask: qualitative open coding. Group the conditions below into DIMENSIONS of the real-world "
                      "situation. A dimension is ONE variable whose different values are different conditions: 'the user is "
                      "asleep' and 'the user is cooking' are two values of one dimension (occupant activity), not two dimensions. "
                      "Prefer few, broad dimensions. Sensor detections describe the situation they indicate ('smoke detected' -> a "
                      "hazard dimension). Device settings or states ('heater is on', 'door is closed', 'home mode is away', 'alarm "
                      "enabled') and user-configured thresholds/times are NOT dimensions: ignore them. For each dimension give a "
                      "snake_case id, a one-sentence definition, its scope (global = whole home; zone = the room where an action's "
                      "effect acts), n_variables (how many distinct physical quantities it bundles, usually 1; e.g. indoor "
                      "temperature, humidity and air quality = 3) and the indices of the conditions it covers.")
    ex = ""
    if existing:
        ex = ("Existing dimensions (do NOT repeat them; propose only NEW dimensions needed for the conditions below):\n" +
              "\n".join(f"- {d['id']}: {d['desc']}" for d in existing) + "\n\n")
    user = ex + "Conditions:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(conditions))
    schema = S({"dimensions": {"type": "array", "items": S(
        {"id": STR, "desc": STR, "scope": {"type": "string", "enum": ["global", "zone"]},
         "n_variables": {"type": "integer"}, "covers": {"type": "array", "items": {"type": "integer"}}},
        ["id", "desc", "scope", "covers"])}}, ["dimensions"])
    return sys, user, schema


def expand_values_prompt(factor: dict, effects_desc: str, assets: list[str], max_new: int):
    """M2 step 4 (agent expansion): propose values missing from the corpus."""
    vals = "; ".join(f"{k} = {v}" for k, v in factor["values"].items())
    sys = PREAMBLE + "\n\nEffects:\n" + effects_desc + "\nAssets: " + ", ".join(assets)
    user = (f"Context factor: {factor['name']} - {factor['semantics']}\nCurrent values: {vals}\n"
            f"Propose up to {max_new} ADDITIONAL values of this factor that occur in real homes and are missing above, "
            "where the new value changes whether some effect harms or benefits some asset compared with the closest existing "
            "value (e.g. cooking smoke vs fire for a hazard factor). For each give: value (snake_case), semantics, "
            "closest_existing (one of the current values), effect (an id from the list), asset, and why the label differs. "
            "Propose nothing if no value is missing.")
    schema = S({"candidates": {"type": "array", "items": S(
        {"value": STR, "semantics": STR, "closest_existing": STR, "effect": STR, "asset": STR, "why": STR},
        ["value", "semantics", "closest_existing", "effect", "asset", "why"])}}, ["candidates"])
    return sys, user, schema


def contrast_label_prompt(effect: dict, asset: dict, factor: dict, value: str, value_sem: str):
    """M2 step 4 check: label (e, v) when the factor takes one value, everything else ordinary."""
    sys = (PREAMBLE + f"\n\nEffect: {effect['id']} - {effect['desc']}\nAsset: {asset['id']} - {asset['desc']}")
    user = (f"Situation: in an ordinary home, {factor['name']} = {value} ({value_sem}); everything else is ordinary. "
            "Give the label of the effect on the asset and, for '-', the harm mechanism.")
    return sys, user, S({"label": LABEL, "harm": STR}, ["label", "harm"])


def elicit_prompt(effect: dict, asset: dict, dims_desc: str, max_items: int):
    """Phase B (effect-anchored elicitation): from one effect x asset, recall the situations
    that change its judgment ('noise is unacceptable while someone watches a movie')."""
    sys = (PREAMBLE + f"\n\nEffect: {effect['id']} - {effect['desc']}\nAsset: {asset['id']} - {asset['desc']}"
           "\n\nCurrent situation dimensions (name: meaning; values):\n" + dims_desc)
    user = (f"In real homes, in which concrete situations is this effect UNACCEPTABLE ('-') or clearly BENEFICIAL ('+') "
            f"for this asset? List up to {max_items} situation attributes that make the difference, each as a short "
            "world fact about people, time or environment (e.g. 'someone is watching a movie', 'a baby is asleep'), "
            "never a device setting. For each give: attribute, label (+ or -), dimension (the name of an existing "
            "dimension it is a value of, or NEW:<snake_case_name> if none fits) and why.")
    schema = S({"items": {"type": "array", "items": S(
        {"attribute": STR, "label": {"type": "string", "enum": ["+", "-"]}, "dimension": STR, "why": STR},
        ["attribute", "label", "dimension", "why"])}}, ["items"])
    return sys, user, schema


def canonicalize_prompt(dim: dict, phrases: list[str], n_existing: int = 0):
    """Phase C step 1: turn the phrases of one dimension into canonical, mutually exclusive
    states (paraphrases merged; semantic merging is left to label signatures). The first
    n_existing phrases are the dimension's EXISTING values; every state names its parent =
    the existing value whose meaning currently contains it (this yields the refines map)."""
    ex = ""
    if n_existing:
        ex = (f" Phrases 0..{n_existing - 1} are the EXISTING values of the dimension (format 'value: meaning'); keep each "
              "of them as its own state with exactly that name. For every state give parent = the existing value whose "
              "meaning currently includes it (the value itself for an existing state), or an empty string if no existing "
              "value covers it.")
    sys = PREAMBLE + ("\n\nTask: the phrases below were collected as values of ONE situation dimension. Rewrite them "
                      "as canonical STATES of that dimension: snake_case name + one-sentence semantics. Merge only "
                      "paraphrases of the same state; keep different states apart even if similar. States must be "
                      "mutually exclusive alternatives of the dimension. Put phrases that are not a state of this "
                      "dimension (device settings, other dimensions, events) into dropped." + ex)
    user = (f"Dimension: {dim['name']} - {dim.get('semantics', '')}\nPhrases:\n" +
            "\n".join(f"{i}. {s}" for i, s in enumerate(phrases)))
    schema = S({"states": {"type": "array", "items": S(
        {"name": STR, "semantics": STR, "members": {"type": "array", "items": {"type": "integer"}}, "parent": STR},
        ["name", "semantics", "members"])},
        "dropped": {"type": "array", "items": {"type": "integer"}}}, ["states"])
    return sys, user, schema


def property_assets_prompt(prop_text: str):
    """Stage A1 metadata: which protected assets a corpus property protects. Used only to
    report corpus coverage per asset (dimension discovery itself is asset-agnostic)."""
    sys = PREAMBLE + "\n\nTask: name the protected asset(s) that the property below protects or serves."
    user = f"Property: {prop_text}"
    schema = S({"assets": {"type": "array", "items": {"type": "string", "enum": [
        "life_health", "security", "property", "privacy", "resource", "comfort"]}}}, ["assets"])
    return sys, user, schema


def attribute_dimension_prompt(attribute: str, dims_desc: str):
    """Stage D3 (refinement): file the missing attribute named by a confirmed
    counterexample under an existing factor or a NEW dimension."""
    sys = PREAMBLE + ("\n\nTask: a situation attribute was found that the current context factors do not capture. "
                      "Decide which factor it is a (more specific) value of, or NEW:<snake_case_name> if it is a different "
                      "dimension of the situation. Also say whether it is a world state (people, time, environment, hazards) "
                      "rather than a device setting or a user preference.\n\nCurrent factors:\n" + dims_desc)
    user = f"Attribute: {attribute}"
    return sys, user, S({"dimension": STR, "is_world_state": BOOL, "reason": STR}, ["dimension", "is_world_state"])


def map_property_prompt(prop_text: str, vocab_desc: str, effects_desc: str):
    sys = PREAMBLE + "\n\nContext factors:\n" + vocab_desc + "\n\nEffect library:\n" + effects_desc
    user = (f"Property: {prop_text}\n"
            "Express this property as constraints over the given vocabulary: which effect(s) it restricts or requires, which asset it protects, "
            "the label ('-' if the effect must NOT occur in the pattern, '+' if the effect is desirable there), and the condition pattern. "
            "If the property cannot be expressed with these factors and values, set expressible=false and say what is missing.")
    schema = S({"expressible": BOOL, "effects": STRLIST, "asset": STR, "label": LABEL,
                "pattern": PATTERN, "missing": STR}, ["expressible"])
    return sys, user, schema


# ---------------------------------------------------------------- M3 bindings
def _ctx(vocab_desc, effect, asset):
    return (PREAMBLE + "\n\nContext factors:\n" + vocab_desc +
            f"\n\nEffect under study: {effect['id']} - {effect['desc']}\nAsset under study: {asset['id']} - {asset['desc']}")


def relevance_prompt(vocab_desc, effect, asset):
    sys = _ctx(vocab_desc, effect, asset)
    user = ("Is there ANY situation (any combination of factor values) in which this effect is labelled '+' or '-' for this asset? "
            "Answer can_affect=true unless the label is '0' in every situation.")
    return sys, user, S({"can_affect": BOOL, "reason": STR}, ["can_affect", "reason"])


def region_prompt(vocab_desc, effect, asset, region_desc, free_factors):
    sys = _ctx(vocab_desc, effect, asset)
    user = (f"Region of situations: {region_desc}\n"
            "Question: is the label of the effect on the asset THE SAME in every situation of this region?\n"
            "- If yes: consistent=true, give the label and (for '-') the harm mechanism.\n"
            f"- If no: consistent=false, pick ONE factor from {free_factors} whose values separate situations with different labels, "
            "and group its values (each group = values with the same label behaviour). Groups must partition the factor's values in the region.")
    schema = S({"consistent": BOOL, "label": LABEL, "harm": STR, "split_factor": STR,
                "value_groups": {"type": "array", "items": STRLIST}, "reason": STR}, ["consistent", "reason"])
    return sys, user, schema


def point_label_prompt(vocab_desc, effect, asset, cell_desc):
    sys = _ctx(vocab_desc, effect, asset)
    user = f"Situation: {cell_desc}\nGive the label of the effect on the asset in this situation and, for '-', the harm mechanism."
    return sys, user, S({"label": LABEL, "harm": STR}, ["label", "harm"])


def refute_prompt(vocab_desc, effect, asset, pattern_desc, label, harm, k):
    sys = _ctx(vocab_desc, effect, asset)
    user = (f"Claim: in EVERY situation matching [{pattern_desc}] the label is '{label}'. Mechanism: {harm}\n"
            f"Try to refute it. Describe up to {k} concrete real-world situations that (1) match the pattern when described with the given factors, "
            "but (2) an expert would label differently because of an attribute NOT captured by the factors. "
            "For each give: narrative, assignment (a value for EVERY factor), expected_label, missing_attribute. "
            "If the claim holds, return an empty list. Do not force counterexamples.")
    schema = S({"counterexamples": {"type": "array", "items": S(
        {"narrative": STR, "assignment": {"type": "object", "additionalProperties": STR},
         "expected_label": LABEL, "missing_attribute": STR},
        ["narrative", "assignment", "expected_label", "missing_attribute"])}}, ["counterexamples"])
    return sys, user, schema


def direct_constraints_prompt(vocab_desc, effect, asset):
    """Baseline B-K1: one-shot free generation of constraints (prior-work style)."""
    sys = _ctx(vocab_desc, effect, asset)
    user = ("List all condition patterns in which this effect harms ('-') or benefits ('+') this asset. "
            "Each item: pattern (factor -> allowed values), label, harm.")
    schema = S({"items": {"type": "array", "items": S({"pattern": PATTERN, "label": LABEL, "harm": STR},
                                                       ["pattern", "label"])}}, ["items"])
    return sys, user, schema


def judge_situation_prompt(effect, asset, narrative):
    """CEGAR spurious-counterexample check: an independent judge labels the concrete
    situation of a counterexample WITHOUT seeing the claim, the factors or the expected label."""
    sys = PREAMBLE.split("- Context factor")[0] + ("- Label of an effect on an asset in a situation:\n"
          "  \"+\" maintains or improves the asset; \"-\" UNACCEPTABLE harm (state the mechanism); \"0\" no relevant influence.\n"
          f"Answer with JSON only.\n\nEffect: {effect['id']} - {effect['desc']}\nAsset: {asset['id']} - {asset['desc']}")
    user = f"Concrete situation: {narrative}\nGive the label of the effect on the asset in this situation and, for '-', the harm mechanism."
    return sys, user, S({"label": LABEL, "harm": STR}, ["label", "harm"])


# ---------------------------------------------------------------- M5 grounding
GROUNDING_DSL = """Grounding DSL (JSON). A condition is one of:
 {"state": ENTITY, "eq": VALUE} | {"state": ENTITY, "in": [VALUES]}
 {"num": ENTITY, "gt": X} | {"num": ENTITY, "lt": X}
 {"held_min": N, "cond": CONDITION}    # CONDITION true and the entity unchanged for >= N minutes
 {"time": ["HH:MM", "HH:MM"]}           # local time inside the window (may wrap midnight)
 {"all": [CONDITIONS]} | {"any": [CONDITIONS]} | {"not": CONDITION}
Rules for a factor are ORDERED; the first rule whose condition is true gives the value.
A missing/unavailable entity makes its condition UNKNOWN (never guess). `default` is the value when every rule is false (null = unknown)."""


def grounding_prompt(factor, area, entities_desc):
    sys = PREAMBLE + "\n\n" + GROUNDING_DSL
    where = f" in area '{area['id']}' ({area['zone_type']})" if area else " for the whole home"
    vals = "; ".join(f"{v} = {factor.value_sem.get(v, v)}" for v in factor.values)
    user = (f"Factor: {factor.name} - {factor.semantics}\nValues: {vals}\nDetermine its value{where} from these Home Assistant entities "
            f"(use ONLY these ids):\n{entities_desc}\n"
            "Write ordered rules (most specific/most dangerous value first). Prefer corroboration by several independent entities. "
            "If no entity can support this factor, set groundable=false.")
    schema = S({"groundable": BOOL, "rules": {"type": "array", "items": S({"value": STR, "when": {"type": "object"}}, ["value", "when"])},
                "default": {"type": ["string", "null"]}, "note": STR}, ["groundable", "rules", "default"])
    return sys, user, schema


# ---------------------------------------------------------------- M6 intent
def intent_prompt(rule_yaml, action_type, effects_desc, vocab_desc):
    sys = PREAMBLE + "\n\nContext factors:\n" + vocab_desc
    user = (f"Automation rule (Home Assistant YAML):\n{rule_yaml}\nIts action type: {action_type}. Effects of this action:\n{effects_desc}\n"
            "Recover the designer's intent: rho = the condition pattern over context factors the designer wanted to respond to "
            "(the real-world situation, NOT the sensor event); intended_effect = one effect id from the list; served_asset = the asset it serves. "
            "Also list alternative_readings: other factor values that would produce the SAME trigger event but are NOT what the designer meant.")
    schema = S({"rho": PATTERN, "intended_effect": STR, "served_asset": STR,
                "alternative_readings": PATTERN, "confidence": {"type": "number"}}, ["rho", "intended_effect", "served_asset"])
    return sys, user, schema


# ---------------------------------------------------------------- B2 baseline
def llm_adjudicate_prompt(rule_yaml, states_desc, priority):
    sys = ("You are a safety guard for a smart home. A rule has just triggered. Decide what to do with its action. "
           f"Asset priority: {' > '.join(priority)}. Answer JSON only.")
    user = (f"Rule:\n{rule_yaml}\nCurrent Home Assistant states:\n{states_desc}\n"
            "disposition: execute | substitute | defer | block | ask. If substitute, name the alternative service call.")
    schema = S({"disposition": {"type": "string", "enum": ["execute", "substitute", "defer", "block", "ask"]},
                "alternative": STR, "reason": STR}, ["disposition", "reason"])
    return sys, user, schema
