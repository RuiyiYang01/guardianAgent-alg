# GuardianAgent

Official code for **GuardianAgent: Policy-Conditioned Risk-Adaptive
Anonymization with Verified Adversarial Escalation**.

**Accepted to Findings of EMNLP 2026.**

- [arXiv abstract](https://arxiv.org/abs/2608.29251)
- [Paper PDF](https://arxiv.org/pdf/2608.29251)

GuardianAgent combines policy matching and content transformation for privacy
protection. It uses an evidential fast path with an LLM fallback, the Adaptive
Multi-factor Risk Scoring Formula (AMRSF) to choose an
`allow`/`transform`/`deny` action and initial anonymization strength, and a
five-level anonymization hierarchy with verified adversarial escalation.

## Repository contents

This repository contains code associated with the EMNLP paper only:

- `guardian_policy_agent/models/`: evidential fast-path classifier and feature encoders;
- `guardian_policy_agent/retrieval/`: structured and lexical policy retrieval;
- `guardian_policy_agent/rag/`: slow-path prompting, model I/O, and response parsing;
- `guardian_policy_agent/service/decider.py`: AMRSF risk estimation and action decisions;
- `guardian_policy_agent/service/anonymizer.py`: five-level anonymization and verified escalation;
- `guardian_policy_agent/eval/`: evaluation implementations used by the paper; and
- `scripts/`: training, benchmark, ablation, robustness, and aggregation code.

Generated result files, raw model outputs, detailed result documents, datasets,
trained checkpoints, browser-extension code, deployment infrastructure, and
the separate system-paper implementation are intentionally not included.
Please obtain the public datasets from their original providers and follow
their licenses and terms of use.

## Setup

The code was developed with Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install `requirements-eval.txt` as well when reproducing comparisons that use
FLAIR, Microsoft Presidio, Hugging Face datasets, or transformer pipelines.

For experiments that use an LLM, configure an OpenAI-compatible local server,
OpenAI, or Gemini through environment variables. For example, for a local
OpenAI-compatible endpoint:

```bash
export LLM_PROVIDER=local
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLM_API_KEY=dummy-key
export LLM_JSON_MODE=true
```

The lexical verifier is the default. Set `VERIFIER_MODE=semantic` to use the
semantic-support variant reported in the paper. The default semantic threshold
is `0.25` and can be changed with `VERIFIER_SEM_THRESHOLD`.

## Running the code

A deterministic smoke run that does not call an LLM is:

```bash
PYTHONPATH=. python scripts/benchmark_anonymizer_paths.py \
  --no-llm --configs A --limit 5
```

The main anonymization benchmark accepts `tab`, `staab-synth`, and
`pii-masking` through `--corpus` once the corresponding datasets are available:

```bash
PYTHONPATH=. python scripts/benchmark_anonymizer_paths.py \
  --corpus staab-synth --limit 200 --seed 42
```

Additional scripts reproduce the paper's AMRSF studies, verifier ablations,
external attribute-inference evaluation, backbone checks, multi-seed runs, and
model-based diagnostics. Each script documents its expected inputs and command
line at the top of the file.

Run the included unit tests with:

```bash
PYTHONPATH=. pytest -q
```

## Citation

```bibtex
@article{yang2026guardianagent,
  title   = {GuardianAgent: Policy-Conditioned Risk-Adaptive Anonymization with Verified Adversarial Escalation},
  author  = {Yang, Ruiyi and Lihinikaduarachchi, Gayathri and Masood, Rahat and Salim, Flora D. and Kanhere, Salil S.},
  journal = {arXiv preprint arXiv:2608.29251},
  year    = {2026},
  note    = {Accepted to Findings of EMNLP 2026}
}
```
