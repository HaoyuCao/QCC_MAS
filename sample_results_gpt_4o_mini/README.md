# Sample Output — AgentClinic Dataset (gpt-4o-mini)

This directory contains the output from running QCC on **1 case** from the AgentClinic-MedQA dataset with `gpt-4o-mini`.

## Files

| File | Description |
|------|-------------|
| `result.json` | Summary output: ACC, ICR, PC, EC, WECI + per-case evidence weights |
| `agentclinic_medqa_0.json` | Full dialogue trace with all agent interactions |

## Case Summary

- **Case**: `agentclinic_medqa_0` — Myasthenia gravis (35F, diplopia, ptosis, fatigable weakness)
- **Prediction**: Myasthenia gravis (correct)
- **WECI**: 0.329
- **ICR**: 0.20
- **Exam Coverage**: 0.45

The `evidence_weights` section in `result.json` shows how ϕ_EA scored each of the 14 patient facts and 11 exam facts, with `collected: true/false` indicating whether the dialogue successfully acquired each item.

## Reproduce

```bash
qcc-eval --datasets medqa --modes qcc \
  --doctor-model gpt-4o-mini \
  --max-items 1 --max-turns 16
```
