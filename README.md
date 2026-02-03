# KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis

KORAL is a **two-stage pipeline** for SSD operational analysis:

- **Stage I (Literature KG):** Extract an **evidence-backed knowledge graph** from SSD research papers, aligned to a curated SSD taxonomy.
- **Stage II (Operational Analysis):** Summarize telemetry (SMART, workload, environment, etc.) using a rule base, retrieve relevant literature evidence from the Stage I KG, and call an LLM to perform SSD analysis (predictive / descriptive / prescriptive / what-if) with automatic evaluation.

---

## Repository layout

```text
KORAL/
├─ data_preparation/                  # Data prep scripts (Alibaba/Google/env/workload)
├─ dataset/                           # Datasets (Alibaba, Google, env, fio_workload, ...)
├─ stage_I/                           # Stage I: paper→KG pipeline
│  ├─ out/                             # default Stage I outputs (TTL/JSON/global KG)
│  ├─ __init__.py
│  ├─ ssd_cot_prompt.txt               # Stage I extraction prompt (strict JSON)
│  ├─ ssd_kg_pipeline.py               # Stage I pipeline (papers → TTL/JSON/global KG)
│  └─ taxonomy.json                    # SSD taxonomy (vocabulary)
├─ stage_II/                          # Stage II: operational pipeline + evaluation
└─ rule_base.json                     # Stage II rule base (summarization/mapping rules)
```

## Installation

Create a fresh environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install pandas numpy pyarrow fastparquet tqdm python-dateutil
pip install rdflib PyPDF2 python-dotenv openai requests
```

Set your OpenAI key (Stage I + Stage II use it):

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

---

# Stage I: Build the Literature Knowledge Graph (papers → TTL)

Stage I reads a **folder of papers** (`.pdf`, `.txt`, `.md`) and produces:

- **per-paper**: `*.ttl` and `*.kg.json`
- **merged**: `global_knowledge_graph.ttl` (accumulates across runs)
- **updated taxonomy**: `taxonomy.json` (if the model proposes new concepts)

## Inputs

- **papers folder**: a directory containing SSD papers (`.pdf`, `.txt`, `.md`)
- **taxonomy**: `stage_I/taxonomy.json`
- **prompt**: `stage_I/ssd_cot_prompt.txt`

The prompt enforces:
- taxonomy-first mapping,
- class vs instance distinction,
- explicit context (SSD ↔ environment/workload),
- directional effects (`improves` / `degrades`) with evidence and confidence,
- optional new concept proposals.  
(See `stage_I/ssd_cot_prompt.txt`.)

## Configure prompt paths

`stage_I/ssd_kg_pipeline.py` defaults to reading the prompt from `prompts/ssd_cot_prompt.txt`.
Since this repo keeps the prompt inside `stage_I/`, set:

```bash
export KG_PROMPT_PATH="stage_I/ssd_cot_prompt.txt"
export KG_PROMPT_ADDENDA_PATH="stage_I/out/ssd_prompt_addenda_auto.txt"
```

- `KG_PROMPT_ADDENDA_PATH` is optional. If enabled, Stage I appends “concept mapping hints” for future runs.

## Run Stage I

Example:

```bash
python stage_I/ssd_kg_pipeline.py \
  --papers_dir dataset/papers \
  --taxonomy stage_I/taxonomy.json \
  --out_dir stage_I/out \
  --model gpt-4o
```

### Outputs (Stage I)

After the run you should see:

```text
stage_I/out/
├─ <paper_slug>.ttl
├─ <paper_slug>.kg.json
└─ global_knowledge_graph.ttl
```

Stage I **merges** the current run into `stage_I/out/global_knowledge_graph.ttl` (it does not overwrite/erase prior knowledge).

---

# Data preparation

This repo includes scripts that prepare Alibaba and Google datasets and create test CSVs.
Place them under `data_preparation/` and keep datasets under `dataset/`.

## Alibaba (SMART)

Expected raw layout:

```text
dataset/alibaba/
├─ smartlog2018ssd/   # daily files
└─ smartlog2019ssd/   # daily files
```

Typical steps used in our pipeline:

1. Filter by model family (MA1/MA2/MB1/MB2/MC1/MC2).
2. Keep the 19 SMART features + `disk_id` + `ds`.
3. Drop model-specific missing attributes (keep `disk_id`, `ds`).
4. Build labeled test samples using `ssd_failure_tag.csv` and a 30-day failure window (for SMART-based datasets).

## Google (SMART)

Expected raw layout:

```text
dataset/google/raw_data/
```

We do not filter by model; we build labels + (optional) 30-day history windows following the dataset release and our pipeline assumptions.

## Environment and workload

- **Environment effects** (from papers): `dataset/env/env_effects.csv`
- **FIO workloads** (samples/configs): `dataset/fio_workload/`

---

# Stage II: Operational analysis + evaluation (telemetry → prompts → metrics)

Stage II consumes one **input CSV** (SMART-only or multi-modal) and produces:

- LLM prompts & responses (saved),
- extracted decisions/labels,
- evaluation metrics:
  - **Predictive**: precision / recall / accuracy
  - **Descriptive / Prescriptive / What-if**: B4, RL, FiP, CFV (as configured in evaluation modules)

## Stage II prerequisites: copy KG + taxonomy to repo root

By default, Stage II looks for these files in the **repo root**:

- `taxonomy.json`
- `global_knowledge_graph.ttl`
- `rule_base.json`

If you ran Stage I using `stage_I/taxonomy.json` and `stage_I/out/global_knowledge_graph.ttl`,
copy (or symlink) them:

```bash
cp stage_I/taxonomy.json taxonomy.json
cp stage_I/out/global_knowledge_graph.ttl global_knowledge_graph.ttl
```

## Dataset types

Stage II is driven by a `dataset_type` name. Common options:

- `SMART_ALIBABA` — SMART-only (Alibaba, no app)
- `SMART_GOOGLE` — SMART-only (Google)
- `ENV` — environment-only effects
- `SMART_WORKLOAD` — SMART + workload (Alibaba with `app`)
- `SMART_ENV` — SMART + env
- `WORKLOAD_ENV` — workload + env
- `SMART_ENV_WORKLOAD` — SMART + env + workload
- `SMART_FT` — SMART + flash-type column (e.g., SLC/TLC/QLC)
- `SMART_AL` — SMART + controller policy columns (GC algo, WL policy, etc.)
- `SMART_FT_ENV_WORKLOAD` — SMART + flash-type + env + workload

> You can generate these CSVs using the dataset generator script in `data_preparation/`
(e.g., `stage2_pair_dataset_generator.py`).

## Run Stage II (CLI)

Example: run all tasks on a given dataset:

```bash
python -m stage_II.cli \
  --dataset_type SMART_ALIBABA \
  --input_csv dataset/alibaba/test_data/smart.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --model gpt-4o \
  --out_name demo_smart_alibaba
```

Debug a small run:

```bash
python -m stage_II.cli \
  --dataset_type ENV \
  --input_csv dataset/env/env_effects.csv \
  --tasks descriptive,whatif \
  --limit_rows 50 \
  --out_name demo_env_50
```

### Stage II outputs

Each run writes a folder under:

```text
stage_II/runs/<run_name>/
├─ inputs/               # normalized rows used for prompts
├─ prompts/              # full prompts sent to the LLM
├─ responses/            # raw LLM responses (JSON where possible)
├─ parsed/               # parsed outputs (labels/decisions/justifications)
├─ metrics/              # metrics summaries (JSON/CSV)
└─ logs/                 # run logs
```

---

## Ontology & KG examples

Stage I and Stage II share a consistent “classes vs instances” design:

- **Classes** come from the taxonomy (e.g., `Temperature`, `IOPS`, `TLC`, `Garbage Collection`).
- **Instances** represent paper-specific or scenario-specific objects (e.g., `SSD_X`, `EC1`, `WP1`, `EXP1`).

Common relation patterns you’ll see in the Literature KG (Stage I) and Data KG (Stage II):

- `SSD_X operatesUnder EC1`
- `EC1 hasTemperature {"@value": 45, "unit": "C"}`
- `EC1 hasWorkloadProfile WP1`
- `WP1 hasReadWriteMix "Write-Heavy"`
- `Temperature degrades 99th Percentile Latency` (directional effect)
- `Workload impactsMetric Latency`
- Assertions always carry **evidence text** and a **confidence** score.

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@inproceedings{AkewarEtAl_IPDPS_2026,
  author    = {Akewar, Mayur and Madireddy, Sandeep and Luo, Dongsheng and Bhimani, Janki},
  title     = {KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis},
  booktitle = {IEEE International Parallel \& Distributed Processing Symposium (IPDPS)},
  year      = {2026},
  note      = {To appear}
}
```
