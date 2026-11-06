# ctxguard：一步一步在 Mac 上跑出全部结果

每一步都标注了对应的论文位置（Sec.）和模块（M1–M11、RQ3）。
所有结果都写进 `out/`。最后跑 `report`，把它们汇总成 `out/report.md`，里面的表可以直接放进论文。

## 0. 一次性安装（约 30 分钟，大部分时间在下载）

```bash
# 1) 安装 ollama：去 https://ollama.com 下载 Mac 版 App 并打开（菜单栏出现小羊驼即可）
# 2) 下载模型（14B 约 9GB；内存只有 16GB 的话改用 7B，并修改 config.yaml 里的 model）
ollama pull qwen2.5:14b-instruct
ollama pull nomic-embed-text
# 3) 解压 ctxguard.zip，进入目录
cd ctxguard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 4) 自检：确认 ollama 能连上、两个模型都已下载
python -m ctxguard.cli check
# 5) 离线冒烟测试（不调用模型，约 3 分钟），必须看到 SMOKE TEST PASSED
python tests/smoke_mock.py
```

以后每次新开终端，先 `cd ctxguard && source .venv/bin/activate`。

### 0b. 用 DeepSeek API 跑（可选，和 ollama 的结果分开存）

```bash
export DEEPSEEK_API_KEY=sk-...                       # 只放在环境变量里，不写进任何文件
python -m ctxguard.cli check --config config_deepseek.yaml   # 应显示 OK deepseek-chat
python -m ctxguard.cli m2-extract --config config_deepseek.yaml
python -m ctxguard.cli m2-dims --config config_deepseek.yaml
```

`config_deepseek.yaml` 的方法参数与 `config.yaml` 相同，结果写到 `out_deepseek/`；缓存按"后端 + 模型"区分，两边永不混。
后面用到 M1 效果库的步骤（m2-elicit、m2-ingest、m3 …）要先在这个配置下跑一次 `m1`。

### 0c. 新版 M2 A 阶段（2026-11-05）先跑这三步看结果

```bash
python -m ctxguard.cli m2-extract            # A1，不变
python -m ctxguard.cli m2-dims               # A2：提升 + 两两关系分槽（默认）
python -m ctxguard.cli m2-values             # A3，不变
```

看三处：`out/m2/A2_dims.yaml`（槽、每个槽的 subject 和事实、`lift_failed`、`relation_grouping`），
`out/annotate/m2_A2_relations_CHECK.csv`（抽检两两关系），`out/annotate/m2_A2_coding_CHECK.csv`（抽检提升）。
想复现旧做法做对比：在 config 里设 `m2_coding: question` 和 `m2_grouping: llm`，并把 `paths.out` 改成另一个文件夹。

### 0d. M1-v3 与 M2 lift-v4.1（2026-11-06）与实验记录规范

**改了什么（都有 config 开关，旧做法保留为消融，可从缓存零调用复现）：**

| 开关（`method.*`） | 默认（新） | 消融（旧） | 为什么 |
|---|---|---|---|
| `m1_direction_rule` | `unanimity_else_both` | `majority` | 开/关窗对室温、湿度的方向取决于室内外，旧规则按方向分别投票，把少数方向投掉（v2 结果 0 个双向效果）。新规则：通道按多数票保留；方向全体一致才取单向，否则（含 `either`）两个方向都保留 |
| prompt M1-v3 | `direction` 可取 `either` | — | 让"随情境而定"能直接表达 |
| `m2_lift_prompt` | `v4.1` | `v4` | v4 的 preamble 禁止"添加原因"，任务却要求写出读数背后的事实，二者矛盾 |
| `m2_lift_gate` | `evidence` | `counterfactual` | 反事实题对安全类 property 会得到"仍然该做"（谨慎原则），v4 因此拒掉 52 个正确提升；新门槛：事实不能是问句、不能含读数/模式/阈值等词、不能复述读数；反事实只记录 |
| `m2_codebook` | `a2_codebook_v1.yaml` | `a2_codebook.yaml` | v1 示例带 `reason`、`latent_fact`；**助手起草，必须人工核对后填 `reviewed_by`** |

**实验记录规范：**每次打分或运行都会追加一行到 `out/eval/experiment_log.csv`
（模型、后端、方法版本、prompts.py 哈希、method 配置哈希、码本版本、样本数、指标 JSON）。
**论文表格里的数字必须能在这个文件里找到对应行。**dev 分数另存到 `out/eval/dev_history/`，不会被覆盖。
prompt 或方法的改动只看 dev / 人工答案，**不看留出集**（56 条留出 property 只在最后 `m2-map` 时用一次）。

**跑之前先存快照（新的 m1 / m2-dims 会覆盖旧文件）：**
```bash
python -m ctxguard.cli snapshot --tag before_m1v3_m2v41
```
**M1：**
```bash
python -m ctxguard.cli m1 && python -m ctxguard.cli m1 --k 1        # 同时写出 *_alt-majority.yaml（消融，零额外调用）
python -m ctxguard.cli m1-score -g out/annotate/m1_effects_gold.csv  # 通道级、效果级 P/R/F1 + 双向召回
python -m ctxguard.cli m1-agree -g <A 的表> <B 的表>                 # 两位标注者的 κ
```
**M2 A2：**
```bash
python -m ctxguard.cli m2-dims && python -m ctxguard.cli m2-dev      # 先看 dev，再决定是否全量
python -m ctxguard.cli m2-dims --runs 3 && python -m ctxguard.cli m2-values
```
看到 `out/m2/A3_STALE.txt` 说明 A3 落后于 A2，必须重跑 `m2-values`。

## 1. 离线知识（RQ1，Sec. IV）：全部自动跑，只在 ★ 处需要你标注

| 顺序 | 命令 | 产出 | 预计耗时(14B) |
|---|---|---|---|
| 1 | `python -m ctxguard.cli m1` | `out/m1/effects.yaml`：效果映射 Eff | 10–20 分钟 |
| 2 | `python -m ctxguard.cli m1 --k 1` | 消融用的单次采样版本 | 5 分钟 |
| 3 | `python -m ctxguard.cli m1-gold-export` → ★ 填 `out/annotate/m1_effects_gold.csv` → `python -m ctxguard.cli m1-score -g out/annotate/m1_effects_gold.csv` | M1 的 P/R/F1 | 你约 1 小时 |
| 4 | M2 第一轮，见下面的「M2 详细步骤」：A1–A4 → B → C → 冻结 F-v1 | `out/m2/A*_… B1_… C*_…`，`data/vocab/factors_v1.yaml` | 7B 约 2–4 小时（`--runs 3` 时 A2 再加约 2 小时） |
| 5 | 把 `config.yaml` 的 `paths.factors` 改成 `data/vocab/factors_v1.yaml` | 冻结后的词表 F-v1 | 1 分钟 |
| 6 | `python -m ctxguard.cli m3 --pairs 10` | 试跑：先看 `out/m3/spec.yaml` 合不合理 | 20–40 分钟 |
| 7 | `python -m ctxguard.cli m3` | 规范 K（完整版） | **3–8 小时，放过夜跑**（有缓存，中断后重跑会接着用） |
| 8 | `python -m ctxguard.cli m2-audit` → `python -m ctxguard.cli m3-refute` → `python -m ctxguard.cli m2-refine` | 粗糙集审计 + 维度卡片；反驳检验（含"0"区域、独立裁判）；缺口转成证据 | 1–2 小时 |
| 9 | `python -m ctxguard.cli baseline-direct` | 基线 B-K1 | 20 分钟 |
| 10 | `python -m ctxguard.cli m2-map` → ★ 核对 `out/annotate/property_mapping_CHECK.csv` | 可表达性、B1 基线的输入 | 你约 1 小时 |
| 11 | `python -m ctxguard.cli vocab-metrics -g out/annotate/property_mapping_CHECK.csv` → ★ 核对 `novel_constraints_CHECK.csv` | 四项词表指标 | 1 分钟 |
| 12 | `python -m ctxguard.cli m4-export` → ★ 填 `out/annotate/audit_K.csv`（300 行） → `python -m ctxguard.cli m4-score -g out/annotate/audit_K.csv` | K 与人工标签的一致性，及两个基线 | 你约 3 小时 |

先跑第 4 步（M2），再跑第 6–7 步（M3）：K 建立在冻结后的词表上。
你要是想先拿到结果，可以先用 v0 词表把全流程跑一遍，再换成 v1 重跑。

### M2 详细步骤（方法见 docs/M2_方法与审计.md）

| 阶段 | 命令 | 你要做什么 | 留下的中间结果（都可进 evaluation） |
|---|---|---|---|
| A1 | `m2-extract` | 无 | `A1_conditions.yaml`、`A1_report.yaml`（按利益统计语料覆盖）、★`annotate/m2_A1_extraction_CHECK.csv` |
| A2 | ★先核对 `data/vocab/a2_codebook.yaml`（码本示例和 dev 集，约 20 分钟）→ `m2-dims --runs 3` → `m2-dev`（看 dev 分数） | 核对码本 | `A2_dims.yaml`、`A2_coding.yaml`、`A2_grouping.yaml`、`A2_runs/`、`A2_stability.yaml`、★`m2_A2_coding_CHECK.csv` |
| A3 | `m2-values` | 无 | `A3_factors_corpus.yaml`、`A3_value_assignment.yaml`（条件→取值）、`A3_values_report.yaml`（覆盖率、没分配上的条件） |
| A4 | ★人工审 `A3_factors_corpus.yaml`，按 R1–R4 写成 `data/vocab/factors_seed.yaml`（每个因子写 provenance）→ `m2-seed-check` | 约 1 小时 | `A4_seed_check.yaml`（审核改了什么） |
| B | `m2-elicit` | 无 | `B1_elicitation.yaml`（含被拒条目）、★`m2_B1_elicit_CHECK_round1.csv` |
| C | `m2-consolidate` | 无 | `C1_canonical.yaml`、`C2_signatures.csv`、`C3_factors_candidate.yaml`（含 refines）、`C3_report.yaml`、★`m2_C2_signature_CHECK.csv` |
| D1 | ★审 `C3_factors_candidate.yaml`（可以直接改）→ `m2-freeze --src out/m2/C3_factors_candidate.yaml --name v1 --parent data/vocab/factors_seed.yaml` | 约 30 分钟 | `data/vocab/factors_v1.yaml`、`m2/versions/F-v1.yaml`、★`m2_rules_CHECK_F-v1.csv`（两人各标） |
| D2 | 改 config → `m3` → `m2-audit` | 无 | `D2_decision_tables.csv.gz`、`D2_core_matrix.csv`、`D2_reducts.yaml`、`D2_value_partition.yaml`、`D2_prime_check.yaml`、`D2_dimension_cards.md/.yaml`、`D2_audit_summary.yaml` |
| D3 | `m3-refute` → `m2-refine` | 看 `D3_refinement.yaml` 的 next | `m3/refutation.yaml`、`D3_refinement.yaml`、★`m2_D3_counterexample_CHECK.csv` |
| 扩展：新 property | `m2-ingest --src <新 property 的 csv>` → `m2-consolidate --base current --dims <它给出的维度>` → 审 → 冻结 | 核对 ★`m2_I1_ingest_CHECK.csv` | `I1_ingest.yaml`（能表达的不动词表；不能表达的才变成证据） |
| 下一版 | `m2-consolidate --base current --dims <D3 给出的维度>` → 审 → `m2-freeze --src … --name v2` → 改 config → `vocab-migrate --old data/vocab/factors_v1.yaml` → `m3 --only-stale` → `m2-audit` | 同 D1 | `m3/spec_F-v1.yaml`（旧 K）、`m3/stale_pairs.yaml`、`m3/only_stale_stats.yaml` |
| 打分 | `m2-check-score -g <填好的 ★ 表…>` | 无 | `eval/m2_checks.yaml`（各项人工核对率、κ） |
| 快照 | `snapshot --tag m2_v1` | 无 | `runs/<日期>_m2_v1/` |

★ 表的填法见 ANNOTATION.md 第 7 节。每个文件写在哪一步、哈希、模型、提示词版本，都记在 `out/MANIFEST.csv`。

## 2. 部署与在线（RQ2，Sec. V；RQ3，Sec. V-D）

| 顺序 | 命令 | 产出 |
|---|---|---|
| 13 | `python -m ctxguard.cli m5 -l L1 L2 L3` | 落地配置 Γ + 可落地率 |
| 14 | `python -m ctxguard.cli m6 -l L1 L2 L3` → ★ 填 `out/annotate/intent_gold_L*.csv` → `python -m ctxguard.cli m6-score -l L1 L2 L3 -g <3 个 csv>` | 意图恢复准确率 |
| 15 | `python -m ctxguard.cli sim -l L1 L2 L3` → ★ 填 `out/annotate/scenario_gold_L*.csv` | 场景集（约 200–300 个） |
| 16 | `python -m ctxguard.cli eval -l L1 L2 L3` | 各方法的决策（含 B2 LLM 基线） |
| 17 | `python -m ctxguard.cli eval-score -l L1 L2 L3 -g <3 个场景 csv>` | 主结果表 |
| 18 | `python -m ctxguard.cli robust -l L1 L2 L3 -g <3 个场景 csv>` | 传感器缺失鲁棒性曲线 |
| 19 | `python -m ctxguard.cli rq3 -l L1 L2 L3` → `python -m ctxguard.cli rq3-verify -l L1 L2 L3` | 编译出的 HA 包 + 等价性验证 |
| 20 | `python -m ctxguard.cli report` | `out/report.md`，含 token 用量 |

## 你的 layout（L1）
照 `data/layouts/L3_family_house.yaml` 的格式写 `data/layouts/L1_<名字>.yaml`，规则写在 `data/rules/L1_rules.yaml`（HA 自动化 YAML 格式，每条规则一个动作）。
也可以直接把你现有的文件发给我，我来转换。

## 原则
- 所有 LLM 调用都缓存在 `out/llm_cache.sqlite`：重跑结果完全一致，不额外花 token。
- 每次调用都记录在 `out/ledger.csv`，`tokens` 命令按模块汇总。
- `--backend mock` 只用于冒烟测试，**不是实验结果**。
- 提示词全部在 `ctxguard/prompts.py`，论文附录直接引用这个文件。
