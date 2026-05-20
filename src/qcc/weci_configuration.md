# WECI Configuration Guide

Weighted Evidence Convergent Index (WECI) is the Quantify-stage metric of the QCC framework.
It assigns importance weights to each evidence item and measures how well the system captures
decision-critical signals during interactive diagnosis.

---

## 1. Concept

```
WECI = Σ_{e ∈ E ∩ E*} w_e  /  Σ_{e ∈ E*} w_e
```

| Symbol | Meaning |
|--------|---------|
| `E` | Evidence items collected by the agent during the interaction |
| `E*` | Ground-truth evidence set (curated by specialist clinicians) |
| `w_e ∈ [0, 1]` | Importance weight of evidence item `e`, assigned by an LLM evaluator |

Unlike PC/EC/ICR (which treat all evidence as equally important), WECI accounts for heterogeneous
clinical significance: missing a pathognomonic finding (w=1.0) hurts more than missing a minor
demographic detail (w=0.1).

---

## 2. LLM Configuration

The evidence scorer uses CAMEL-AI's `ChatAgent`. Configure via environment variables or CLI flags.

### 2.1 Environment variables (`.env`)

```bash
OPENAI_API_BASE_URL=https://your-endpoint/v1
OPENAI_API_KEY=sk-...
```

### 2.2 CLI flags (override `.env`)

```
weci annotate --api-url https://other-endpoint/v1 --api-key sk-xxx ...
```

### 2.3 Model selection

```bash
weci annotate --model gpt-5-mini ...
# or
weci annotate --model qwen3-max ...
```

The model must support OpenAI-compatible chat completions. Default: `gpt-5-mini` with
`temperature=0.0` for deterministic scoring.

### 2.4 Scoring guidelines (built into the prompt)

| Weight | Clinical Meaning |
|--------|-----------------|
| 1.0 | Pathognomonic / gold-standard — alone strongly confirms the diagnosis |
| 0.7–0.9 | Highly discriminative — key finding that narrows the differential |
| 0.4–0.6 | Moderately helpful — supports diagnosis but non-specific |
| 0.1–0.3 | Marginally relevant — background context, demographics, weak signal |
| 0.0 | Irrelevant — provides no diagnostic signal |

---

## 3. RAG Backend Setup

The `weci.py` module provides a `KnowledgeRetriever` abstract base class.
The default `NoOpRetriever` returns empty context — this is a placeholder for future RAG integration.

### 3.1 Implementing a custom retriever

```python
from qcc.weci import KnowledgeRetriever

class PubMedRetriever(KnowledgeRetriever):
    def retrieve(self, diagnosis: str, evidence: str) -> str:
        # Query your vector DB / PubMed / MedCPT with (diagnosis + evidence)
        results = your_search_function(f"{diagnosis} {evidence}")
        return "\n".join(r.text for r in results[:5])
```

### 3.2 Wiring into the annotate command

```python
# In cmd_annotate() — replace NoOpRetriever() with your implementation
retriever = PubMedRetriever()
```

When RAG is enabled, the retrieved context is injected into the LLM prompt as:

```
ADDITIONAL MEDICAL KNOWLEDGE:
<retrieved context>
```

---

## 4. CLI Usage

### 4.1 Install

```bash
pip install -e .
# This registers the `weci` command via pyproject.toml entry point
```

### 4.2 `annotate` — Generate evidence weights

Annotate every sample in a segmented JSONL dataset with importance weights (0–1) using an LLM.

```bash
weci annotate \
  --input data/agentclinic_medqa_segmented.jsonl \
  --output results/weci_weights.jsonl \
  --model gpt-5-mini \
  --max-items 50 \
  --workers 5
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | Yes | — | Segmented JSONL dataset path |
| `--output` | Yes | — | Output JSONL with annotated weights |
| `--model` | No | `gpt-5-mini` | LLM model name |
| `--api-url` | No | `$OPENAI_API_BASE_URL` | API base URL override |
| `--api-key` | No | `$OPENAI_API_KEY` | API key override |
| `--max-items` | No | unlimited | Limit number of samples to process |
| `--workers` | No | `5` | Thread-parallel workers |
| `--rag-backend` | No | `None` | RAG backend (requires custom retriever) |

**Output format** (one JSON object per line):

```json
{
  "sample_id": "agentclinic_medqa_1",
  "correct_diagnosis": "Acute Myocardial Infarction",
  "patient_evidence": [
    {"index": 1, "text": "1. Patient has crushing chest pain...", "weight": 0.95},
    {"index": 2, "text": "2. Patient is 65 years old...", "weight": 0.3}
  ],
  "exam_evidence": [
    {"index": 1, "text": "1. ECG shows ST-segment elevation...", "weight": 1.0},
    {"index": 2, "text": "2. Troponin I elevated...", "weight": 0.9}
  ],
  "total_weight": 3.15
}
```

### 4.3 `compute` — Compute WECI from traces

Compute WECI from scenario result traces using pre-annotated evidence weights.

```bash
weci compute \
  --weights results/weci_weights.jsonl \
  --traces results/qcc/20260519/ \
  --verbose
```

| Flag | Required | Description |
|------|----------|-------------|
| `--weights` | Yes | Annotated weights JSONL (from `annotate`) |
| `--traces` | Yes | Directory containing trace JSON files from scenario runs |
| `--verbose` | No | Print per-case WECI, PC, EC |

**Output** (JSON to stdout):

```json
{
  "num_samples": 48,
  "avg_weci": 0.7234,
  "avg_patient_coverage": 0.8512,
  "avg_exam_coverage": 0.6430
}
```

`avg_patient_coverage` and `avg_exam_coverage` are raw unweighted coverage rates
(PC/EC) included for comparison against WECI.

---

## 5. End-to-End Workflow

```bash
# Step 1: Run QCC evaluation to collect traces
qcc-eval --datasets medqa --modes qcc \
  --doctor-model gpt-5-mini \
  --max-turns 16 --max-items 50

# Step 2: Annotate dataset with evidence importance weights
weci annotate \
  --input data/agentclinic_medqa_segmented.jsonl \
  --output results/weci_weights.jsonl \
  --model gpt-5-mini

# Step 3: Compute WECI from traces
weci compute \
  --weights results/weci_weights.jsonl \
  --traces results/qcc/latest/ \
  --verbose
```

---

## 6. Input Dataset Format

The `annotate` command expects a segmented JSONL file where each line contains:

```json
{
  "id": "case_001",
  "OSCE_Examination": {
    "Correct_Diagnosis": "Disease Name"
  },
  "atomic_facts": {
    "patient_facts": [
      "1. Chief complaint description...",
      "2. Past medical history..."
    ],
    "exam_facts": [
      "1. Physical exam finding...",
      "2. Lab result..."
    ]
  }
}
```

Evidence items MUST be numbered (`1.`, `2.`, ...) — the WECI matching logic uses these indices.
This format is the standard output of `qcc-segment`.

---

## 7. Trace Format

The `compute` command reads trace JSON files produced by scenario evaluation runs.
Each file contains `case_id` and `trace` (list of turn entries with `role_id` and `content`).
Evidence indices are extracted from:

- `role_id=patient` → `[REFERENCE]` section → patient fact indices
- `role_id=reporter` or `measurement` → content → exam fact indices

---

## 8. Notes

- **RAG is a placeholder**: The `NoOpRetriever` returns empty context. To enable RAG, subclass `KnowledgeRetriever` and wire it into `cmd_annotate()`.
- **Reproducibility**: temperature=0.0 ensures deterministic scoring for the same model.
- **Parallelism**: Thread-parallel via `ThreadPoolExecutor` — safe for I/O-bound LLM calls. Increase `--workers` for faster annotation on large datasets.
- **Full WECI integration** (φ_EA agent pipeline) is planned for a future release. See `ONHOLD.md` for details.
