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

## 1. 离线知识（RQ1，Sec. IV）：全部自动跑，只在 ★ 处需要你标注

| 顺序 | 命令 | 产出 | 预计耗时(14B) |
|---|---|---|---|
| 1 | `python -m ctxguard.cli m1` | `out/m1/effects.yaml`：效果映射 Eff | 10–20 分钟 |
| 2 | `python -m ctxguard.cli m1 --k 1` | 消融用的单次采样版本 | 5 分钟 |
| 3 | `python -m ctxguard.cli m1-gold-export` → ★ 填 `out/annotate/m1_effects_gold.csv` → `python -m ctxguard.cli m1-score -g out/annotate/m1_effects_gold.csv` | M1 的 P/R/F1 | 你约 1 小时 |
| 4 | `python -m ctxguard.cli m2-extract` → `m2-dims` → `m2-values` → `m2-expand` | 维度从语料归纳（`dims.yaml`）→ 取值从语料归纳（`factors_corpus.yaml`）→ agent 扩展取值（`factors_candidate.yaml`、`expansion_report.yaml`）；消融：`m2-slots`、`m2-cluster` → `m2-name` | 7B 约 20–30 分钟 |
| 5 | ★ 审 `factors_candidate.yaml`：合并、改名、补可行性约束，另存为 `data/vocab/factors_v1.yaml`，并把 `config.yaml` 的 `paths.factors` 改成它；同时记下你改了几处（论文里要报告）。**与 v0 同义的因子保留 v0 的名字**（模拟器按这些名字生成世界） | 冻结后的词表 F-v1 | 你约 1 小时 |
| 6 | `python -m ctxguard.cli m3 --pairs 10` | 试跑：先看 `out/m3/spec.yaml` 合不合理 | 20–40 分钟 |
| 7 | `python -m ctxguard.cli m3` | 规范 K（完整版） | **3–8 小时，放过夜跑**（有缓存，中断后重跑会接着用） |
| 8 | `python -m ctxguard.cli m3-refute` | 反驳检验与缺口报告 | 1 小时 |
| 9 | `python -m ctxguard.cli baseline-direct` | 基线 B-K1 | 20 分钟 |
| 10 | `python -m ctxguard.cli m2-map` → ★ 核对 `out/annotate/property_mapping_CHECK.csv` | 可表达性、B1 基线的输入 | 你约 1 小时 |
| 11 | `python -m ctxguard.cli vocab-metrics -g out/annotate/property_mapping_CHECK.csv` → ★ 核对 `novel_constraints_CHECK.csv` | 四项词表指标 | 1 分钟 |
| 12 | `python -m ctxguard.cli m4-export` → ★ 填 `out/annotate/audit_K.csv`（300 行） → `python -m ctxguard.cli m4-score -g out/annotate/audit_K.csv` | K 与人工标签的一致性，及两个基线 | 你约 3 小时 |

先跑第 4 步（M2），再跑第 6–7 步（M3）：K 建立在冻结后的词表上。
你要是想先拿到结果，可以先用 v0 词表把全流程跑一遍，再换成 v1 重跑。

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
