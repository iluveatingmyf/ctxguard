# ctxguard results  (model: qwen2.5:7b-instruct, backend: mock, vocab: F-v0)

> **MOCK BACKEND - smoke test only, not results.**

## Vocabulary (RQ1)

|Sigma| = 16560 of product 25920; projected table max 120, mean 14.3 over 15 pairs.

| factor removed | conflict groups | pairs affected | necessary |
| --- | --- | --- | --- |
| time_period | 0 | 0 | NO |
| effect_zone | 0 | 0 | NO |
| home_occupancy | 8640 | 3 | yes |
| zone_occupant | 6480 | 2 | yes |
| vulnerable_present | 17928 | 5 | yes |
| active_hazard | 13536 | 5 | yes |
| outdoor_condition | 9360 | 3 | yes |
| indoor_air | 3024 | 1 | yes |
| indoor_thermal | 13800 | 4 | yes |


Held-out expressibility: {'n_test': 2, 'expressible': 1, 'ratio': 0.5, 'human_checked': False}; K recall: {'mean_cell_coverage': 0.667, 'share_covered_ge_0.9': 0.0}; novel '-' constraints: {'n_minus_bindings': 14, 'not_in_any_property': 11}


Groundability L2: 10/24 = 0.417; ungroundable: zone_occupant@bedroom, zone_occupant@kitchen, zone_occupant@bathroom, vulnerable_present@bedroom, vulnerable_present@living_room, vulnerable_present@kitchen, vulnerable_present@bathroom, vulnerable_present@entrance, indoor_air@living_room, indoor_air@kitchen, indoor_air@entrance, indoor_thermal@kitchen, indoor_thermal@bathroom, indoor_thermal@entrance


Groundability L3: 31/44 = 0.705; ungroundable: zone_occupant@garage, zone_occupant@laundry, indoor_air@grandma_room, indoor_air@kitchen, indoor_air@hallway, indoor_air@entrance, indoor_air@garage, indoor_air@laundry, indoor_thermal@hallway, indoor_thermal@bathroom, indoor_thermal@entrance, indoor_thermal@garage, indoor_thermal@laundry


## Specification quality vs human labels (RQ1)

| method | acc | kappa | P(-) | R(-) | F1(-) | F1(+) |
| --- | --- | --- | --- | --- | --- | --- |
| K | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| B-K1_direct | 0.344 | 0.017 | 1.0 | 0.033 | 0.065 | 0.0 |
| B-K2_per_cell | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |


## Runtime adjudication (RQ2), n = 154 scenarios {'random_rho': 60, 'targeted_A': 37, 'b1_forged': 23, 'missing': 30, 'b2_ambiguous': 4}

| method | accuracy | unsafe exec | over-intervention | purpose retention | factors | entities | latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0_execute | 0.851 | 1.0 | 0.0 | None | 0.0 | 0.0 | None |
| B1_property | 0.818 | 0.913 | 0.053 | None | 0.844 | 1.442 | 0.005 |
| B2_llm | 0.851 | 1.0 | 0.0 | None | None | 52.896 | 0.44 |
| abl_fill_default | 0.578 | 0.261 | 0.405 | None | 4.149 | 7.753 | 0.03 |
| abl_ground_all | 0.506 | 0.174 | 0.595 | None | 9.0 | 13.416 | 0.035 |
| abl_no_corroboration | 0.552 | 0.174 | 0.511 | None | 4.149 | 7.753 | 0.022 |
| abl_no_intent | 0.532 | 0.435 | 0.473 | None | 3.948 | 7.312 | 0.024 |
| ours | 0.506 | 0.174 | 0.595 | None | 4.149 | 7.753 | 0.045 |


## Robustness to missing sensors

| ours: missing ratio | acc | unsafe | intervene |
| --- | --- | --- | --- |
| 0.0 | 0.423 | None | 0.577 |
| 0.1 | 0.381 | None | 0.619 |
| 0.2 | 0.309 | None | 0.691 |
| 0.3 | 0.309 | None | 0.691 |
| 0.4 | 0.278 | None | 0.722 |
| 0.5 | 0.278 | None | 0.722 |

| abl_fill_default: missing ratio | acc | unsafe | intervene |
| --- | --- | --- | --- |
| 0.0 | 0.588 | None | 0.412 |
| 0.1 | 0.619 | None | 0.381 |
| 0.2 | 0.567 | None | 0.433 |
| 0.3 | 0.629 | None | 0.371 |
| 0.4 | 0.588 | None | 0.412 |
| 0.5 | 0.629 | None | 0.371 |

| B1_property: missing ratio | acc | unsafe | intervene |
| --- | --- | --- | --- |
| 0.0 | 0.928 | None | 0.072 |
| 0.1 | 0.959 | None | 0.041 |
| 0.2 | 0.969 | None | 0.031 |
| 0.3 | 0.938 | None | 0.062 |
| 0.4 | 0.979 | None | 0.021 |
| 0.5 | 0.99 | None | 0.01 |


## Policy compilation (RQ3)
rules 30, all equivalent: True, mean branches 12.97, max enumerated domain 127008, distinct abstract policies 25


## LLM cost

| module | class | calls | prompt tok | completion tok | s |
| --- | --- | --- | --- | --- | --- |
| B2/* | per-trigger (baseline LLM guard) | 154 | 104757 | 1694 | 0.0 |
| BASE/* | evaluation baselines | 105 | 96183 | 723 | 0.0 |
| M1/* | one-time knowledge | 116 | 70644 | 6180 | 0.0 |
| M2/* | one-time knowledge | 207 | 117994 | 1932 | 0.0 |
| M3/* | one-time knowledge | 944 | 888892 | 7420 | 0.0 |
| M5/* | per-home deployment | 68 | 104985 | 2141 | 0.0 |
| M6/* | per-home deployment | 30 | 30806 | 1164 | 0.0 |
