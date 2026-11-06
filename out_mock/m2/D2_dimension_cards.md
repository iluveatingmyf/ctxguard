# Dimension cards — F-v0b

|Σ| = 20880; pairs with a non-0 label: 15/150; global core: ['home_occupancy', 'zone_occupant', 'vulnerable_present', 'active_hazard', 'outdoor_condition', 'indoor_air', 'indoor_thermal']; global reduct: ['home_occupancy', 'zone_occupant', 'vulnerable_present', 'active_hazard', 'outdoor_condition', 'indoor_air', 'indoor_thermal']; prime bindings 23/23, irredundant 23/23


## time_period  (global, corpus)
Period of day when the action executes.

- values: `day` Daytime hours when residents are normally active.; `night` Night hours when residents normally sleep.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = False, necessary for 0 pairs (0 conflict classes if removed)
- decision-preserving value groups: [['day', 'night']]
- rules: not annotated
- **verdict: DROP candidate: no label in K depends on this factor (not in the global core)**

## effect_zone  (action, derived)
Type of the area in which the action's effect acts.

- values: `bedroom` A sleeping room.; `kitchen` A cooking area.; `bathroom` A bathroom or shower room.; `living` Living room, dining room, study or hallway.; `entry` Front door, entrance hall, garage.; `utility` Laundry, storage, technical room.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = False, necessary for 0 pairs (0 conflict classes if removed)
- decision-preserving value groups: [['bedroom', 'kitchen', 'bathroom', 'living', 'entry', 'utility']]
- rules: not annotated
- **verdict: DROP candidate: no label in K depends on this factor (not in the global core)**

## home_occupancy  (global, corpus)
Whether any resident is at home.

- values: `occupied` At least one resident is at home.; `away` No resident is at home.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 3 pairs (8640 conflict classes if removed)
- decision-preserving value groups: [['occupied'], ['away']]
- rules: not annotated
- **verdict: KEEP**

## zone_occupant  (zone, corpus)
State of the people inside the effect zone.

- values: `empty` Nobody is in the zone.; `awake` Someone awake is in the zone.; `asleep` Someone is asleep in the zone and nobody awake is there.; `watching_a_movie` watching a movie
- corpus: 0 properties; elicitation: 1 items from 1 pairs; refutation counterexamples: 1
- stability: {}
- necessity: in global core = True, necessary for 2 pairs (6480 conflict classes if removed)
- decision-preserving value groups: [['empty', 'awake'], ['asleep'], ['watching_a_movie']]
- rules: not annotated
- **verdict: MERGE candidate: decision-preserving value groups [['empty', 'awake'], ['asleep'], ['watching_a_movie']]**

## vulnerable_present  (zone, corpus)
Whether a child, elderly or impaired person, or a pet is in the effect zone.

- values: `yes` At least one vulnerable person or a pet is in the zone.; `no` No vulnerable person or pet is in the zone.
- corpus: 0 properties; elicitation: 2 items from 2 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 5 pairs (22608 conflict classes if removed)
- decision-preserving value groups: [['yes'], ['no']]
- rules: not annotated
- **verdict: KEEP**

## active_hazard  (global, corpus)
The hazard currently present in the home and its source.

- values: `none` No hazard.; `cooking_fumes` Smoke, CO or heat produced by normal cooking (not a fire).; `fire` An uncontrolled fire.; `gas_leak` Fuel gas is leaking.; `water_leak` Water is leaking or flooding.; `intrusion` An unauthorized person is entering or inside.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 5 pairs (16920 conflict classes if removed)
- decision-preserving value groups: [['none'], ['cooking_fumes'], ['fire'], ['gas_leak'], ['water_leak'], ['intrusion']]
- rules: not annotated
- **verdict: KEEP**

## outdoor_condition  (global, corpus)
Outdoor weather/air relative to what is acceptable indoors.

- values: `benign` Mild, dry and clean outdoors.; `cold` Outdoor temperature is low enough to chill the home.; `hot` Outdoor temperature is high enough to overheat the home.; `rain_storm` Rain, storm or strong wind.; `polluted` Outdoor air quality is poor (smog, smoke, pollen).
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 3 pairs (11808 conflict classes if removed)
- decision-preserving value groups: [['benign'], ['cold'], ['hot'], ['rain_storm'], ['polluted']]
- rules: not annotated
- **verdict: KEEP**

## indoor_air  (zone, corpus)
Air quality in the effect zone.

- values: `good` CO2, humidity and pollutants are within healthy ranges.; `poor` CO2, humidity or pollutants exceed healthy ranges.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 1 pairs (3780 conflict classes if removed)
- decision-preserving value groups: [['good'], ['poor']]
- rules: not annotated
- **verdict: KEEP**

## indoor_thermal  (zone, corpus)
Thermal state of the effect zone.

- values: `cold` Too cold for comfort or health.; `comfortable` Within the comfort range.; `hot` Too hot for comfort or health.
- corpus: 0 properties; elicitation: 0 items from 0 pairs; refutation counterexamples: 0
- stability: {}
- necessity: in global core = True, necessary for 4 pairs (17400 conflict classes if removed)
- decision-preserving value groups: [['cold'], ['comfortable'], ['hot']]
- rules: not annotated
- **verdict: KEEP**
