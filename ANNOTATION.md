# 标注手册（6 张表，总计约 10 小时）

通用规则：
- 只看表里给出的信息，**不要**看系统输出。
- 每张表都可以请一位同学独立重标 20% 的行，填进 ANNOTATOR_B 列，用来算 κ（评审一定会问）。实在找不到人，就在论文的 Threats to Validity 里说明“单人标注”。
- 拿不准的行，在 note 里写原因，不要硬标。

## 1. m1_effects_gold.csv（动作 → 效果，约 45 分钟）
每行一个动作，每列一个物理通道（含义见 `data/vocab/channels.yaml`）。每格填一个字母：
`u` = 该通道上升，`d` = 下降，`ud` = 视情况可升可降，**不影响就留空**。不要打勾，也不要用 + 或 -（Excel 会把它们当成公式）。
判断标准：在一个普通家庭里，这个动作本身会直接造成的物理变化，副作用也算（例如电机噪声、开窗进风进声）。
需要额外设备或特殊安装才成立的不算（例如“如果装了摄像头”“如果插座接的是暖气”）。
例：open_window → air_exchange `u`，passage_opening `u`，indoor_temperature `ud`，indoor_humidity `ud`，sound `u`，visibility `u`。

## 2. property_mapping_CHECK.csv（约 1 小时）
- `HUMAN_expressible`：这条 property 能否只用词表里的因子和取值表达。能填 1，不能填 0。
- `HUMAN_mapping_correct`：系统给的映射（效果、标签、利益、条件）对不对。对填 1，不对填 0。

## 3. novel_constraints_CHECK.csv（约 30 分钟）
这些是 K 里有、但种子 property 里没有的约束。合理且确实会造成损害的填 1，否则填 0。
这一项是“系统能提出还应该检查什么”这一论点的证据。

## 4. audit_K.csv（300 行，约 3 小时，最重要）
每行给出一个效果、一项利益和一个完整的抽象情境。
- 在这个情境下，这个效果对这项利益是：`+` 维护或改善；`-` 造成**不可接受**的损害；`0` 无关或影响可接受。
- 标 `-` 时，在 GOLD_harm 列写一句损害机制。写不出机制，就不要标 `-`。
- 判断时只依据情境里写出的内容，不要自行假设额外的条件。

## 5. intent_gold_L*.csv（每个 layout 约 20 分钟）
- `GOLD_rho`：设计者真正想响应的现实情境（不是传感器读数），格式为 `因子=取值|取值; 因子=取值`。例如 `indoor_air=poor`、`active_hazard=fire`。
- `GOLD_intended_effect`：从 effects_to_choose_from 里选一个。
- `GOLD_served_asset`：从 life_health / security / property / privacy / resource / comfort 里选一个。

## 6. scenario_gold_L*.csv（共约 200–300 行，约 3 小时）
你看到的是**真实情况**（true_situation）和被触发的规则。请判断以下每种处置是否可以接受（1 = 可以，0 = 不可以，可以有多个 1）：
- execute：照原规则执行
- substitute：写出可接受的替代动作（如 `exhaust_fan_on` 或实体名），没有就填 0；可选动作列在 possible_substitutes 列
- defer：暂缓，等情境变化后再判
- block：不执行，并通知用户
- ask：询问用户
判断标准：一个了解全部真实情况、谨慎的房主会接受什么。
触发信号是伪造的（B1）或情境被误读的（B2）时，原动作一般不应执行。


## 7. M2 核对表（共约 2 小时；论文里报告核对率和 κ）
所有表在 `out/annotate/`，填 1 或 0；填完跑 `python -m ctxguard.cli m2-check-score -g <这些 csv>`。
- `m2_A1_extraction_CHECK.csv`（50 行）：`HUMAN_faithful` 条件确实是原 property 说的或直接蕴含的；`HUMAN_world_state` 条件描述的是世界状态（人、时间、环境、危险），不是设备设置。
- `m2_A2_coding_CHECK.csv`（50 行）：`HUMAN_kind_ok` 类型（世界状态 / 传感读数 / 模式 / 设备状态 / 配置 / 事件 / 义务）判对了；`HUMAN_dimension_ok` 放进这个维度是对的；不对就在 `HUMAN_better` 写应该是什么。
- `m2_B1_elicit_CHECK_round1.csv`（60 行）：`HUMAN_world_state`、`HUMAN_label_ok`（该情境下 +/- 判断合理）、`HUMAN_dimension_ok`（归到的维度合理）。
- `m2_C2_signature_CHECK.csv`（60 行）：在 `HUMAN_label` 填你认为的标签 + / - / 0（只看 dimension=value、effect、asset，别看系统的 label 列）。
- `m2_rules_CHECK_F-v1.csv`（每个因子一行）：A、B 两人各自填 R1 世界状态、R2 单一变量、R3 互斥且完备、R4 原因而非读数。找不到第二个人就只填 A_ 列。
- `m2_D3_counterexample_CHECK.csv`：`HUMAN_real_counterexample` 这个反例在现实中成立且确实该判不同；`HUMAN_dimension_ok`。
