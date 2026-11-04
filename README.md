# TherapyGym

TherapyGym is a research framework for evaluating and improving therapy-oriented language-model agents using clinically grounded skill and safety criteria.

> [!IMPORTANT]
> TherapyGym is a research artifact, not a medical device or clinical service. It must not be used to diagnose, treat, or replace professional mental-health care. Model outputs may be inaccurate or unsafe.

## Resources

- Project: https://therapygym.stanford.edu
- Paper: *TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots* (NeurIPS 2026)
- TherapyJudgeBench: https://huggingface.co/datasets/therapygym/therapyjudgebench
- TreeSynth patient profiles: https://huggingface.co/datasets/therapygym/treesynth-patient-profiles

## What is included

- multi-turn patient simulation and therapist-agent code;
- the TherapyJudge prompt and evaluation pipeline;
- CTRS-aligned skill and therapy-safety reward computation;
- GRPO training integration built on rLLM and verl;
- benchmark-runner configuration and prompts.

Datasets and third-party corpora are not redistributed in this code repository. TherapyJudgeBench and the TreeSynth patient profiles are available from the links above under their separately stated CC BY 4.0 dataset licenses. Other third-party assets retain their original licenses and terms.

## Installation

The reference environment uses Python 3.10, CUDA-capable GPUs, PyTorch 2.6, vLLM 0.8.4, and verl 0.3.x.

```bash
git clone https://github.com/fangruih/therapygym.git
cd therapygym
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Required only for GRPO training. Pin to the version used by the paper.
git clone --branch v0.3.1 https://github.com/volcengine/verl.git
pip install -e ./verl
pip install "flash-attn>=2.6.0,<2.7.0" --no-build-isolation
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for GPU environment notes.

## API credentials

Set only the credentials needed for the provider you use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export WANDB_API_KEY=...   # optional
```

Never commit credentials. The public release contains no Stanford-internal API configuration.

## Run the evaluator

The main evaluator is `therapygym.therapy.judge.TherapyJudge`. It accepts a multi-turn patient/therapist conversation and returns normalized CBT skill scores plus binary safety indicators. Public OpenAI, Anthropic, Amazon Bedrock, LiteLLM, and local vLLM adapters are available under `therapygym/rllm/llm_api/`.

For local inference, start an OpenAI-compatible vLLM server:

```bash
python -m therapygym.therapy.launch_patient_vllm \
  --model Qwen/Qwen3-4B \
  --port 8000
```

## GRPO training

After installing verl and preparing the separately distributed patient profiles:

```bash
python -m therapygym.therapy.train_therapy \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train_verl.parquet \
  data.val_files=/path/to/val_verl.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B
```

Large-scale training requires multiple GPUs. Exact hyperparameters and compute used for the paper are reported in the paper and its appendix.

## Repository layout

```text
therapygym/
  therapy/              Therapy agents, patient simulation, judge, and rewards
  rllm/                 Apache-2.0-derived rLLM integration
ENVIRONMENT.md          Reference environment notes
pyproject.toml          Python package metadata and dependencies
```

## Data and responsible use

Therapy conversations can contain sensitive subject matter. Do not use identifiable patient data with this software. The benchmark is intended for research evaluation and does not establish clinical efficacy or deployment readiness. See the paper for the safety taxonomy, limitations, and annotation protocol.

## License and attribution

TherapyGym code is released under the Apache License 2.0. Portions under `therapygym/rllm/` are derived from [rLLM](https://github.com/agentica-project/rllm); see [NOTICE](NOTICE). Datasets and model weights retain their own licenses and are not covered by the code license.

## Citation

See [CITATION.cff](CITATION.cff). A proceedings identifier will be added after publication.
