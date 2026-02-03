KORAL Stage II
================================================

This package provides Stage II pipeline that:
1) Reads a prepared input CSV (SMART / SMART+Workload / SMART+Env / etc.)
2) Builds an Intermediate Representation (IR) for SMART + optional modalities
3) Materializes a lightweight DataKG artifact per sample (TTL if rdflib is available)
4) Retrieves lightweight evidence from a Literature KG TTL (SPARQL via rdflib when available)
5) Calls GPT-4o (OpenAI Chat Completions) for:
     - predictive
     - descriptive
     - prescriptive
     - what-if
6) Records responses + computes metrics:
     Predictive: Precision/Recall/Accuracy (+ optional TTF_MSE, TL_MSE)
     Text: BLEU-4 (B4), ROUGE-L (RL)
     Grounding: FiP for descriptive/prescriptive, CFV for what-if

Where to put it in your repo
----------------------------
Copy the 'stage_II' folder into your project root so imports work:
  <repo_root>/stage_II/...

Your repo structure (as you described):
  dataset/
    alibaba/
    google/
    env/
    fio_workload/
  data_preparation/
  stage_II/   <-- copy here

How to run
----------
1) Install dependencies:
   pip install pandas numpy requests
   (optional but recommended) pip install rdflib

2) Export OpenAI key:
   export OPENAI_API_KEY="sk-..."

3) Run a Stage II job:
   python -m stage_II.cli --dataset_type SMART_ALIBABA --input_csv dataset/alibaba/test_data/smart.csv --tasks predictive,descriptive,prescriptive,whatif --limit_rows 100

Notes on input CSV schema
-------------------------
- SMART columns: any header matching r_<number> will be treated as SMART.
  Values can be scalar or a JSON list string like "[...]" for 30-day windows.
- Labels:
    - classification ground truth: 'failure' or 'label' (0/1)
    - optional regression: 'ttf_days' and 'tail_latency_ms'
- Optional modalities:
    - workload: 'app' (Alibaba) or 'fio_job' (text of an FIO job)
    - environment: columns like 'temperature_c', 'relative_humidity_pct',
      'vibration_freq_hz', 'vibration_amp_g', plus optional change fields.
    - flash type: 'flash_type' or 'ft'
    - algorithms/policies: 'algorithms' or 'policies' (semicolon-separated OK)

Reference text columns (optional)
--------------------------------
To compute BLEU/ROUGE, include any of:
  ref_descriptive, ref_prescriptive, ref_whatif

Outputs
-------
stage_II/runs/<RUN_NAME>/
  input_samples.csv
  responses.jsonl
  metrics_per_sample.csv
  metrics_summary.json
  data_kg_ttl/<sample_id>.ttl   (if rdflib available)

