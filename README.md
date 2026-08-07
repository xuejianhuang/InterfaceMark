# InterfaceMark

Official implementation of the released **Terminal InterfaceMark** protocol:
training-free, one-bit watermark-presence detection for a frozen latent
diffusion model.

## Repository structure

```text
interfacemark-watermark/
├── configs/
│   ├── sd15_terminal_smoke.yaml
│   └── sd15_terminal_5000.yaml
├── prompts/
│   └── PartiPrompts.tsv
├── scripts/
│   └── run_full.sh
├── src/interfacemark/
│   ├── core.py        # carrier, tail transport, four displacement laws
│   ├── sd15.py        # frozen SD1.5/VAE channel
│   ├── prepare.py     # splits, carriers, and design-set controls
│   ├── generate.py    # resumable image generation and scoring
│   ├── evaluate.py    # audit, calibration, AUC/TPR/FPR, quality
│   └── records.py     # hashes and atomic records
├── tests/
│   └── test_core.py
├── pyproject.toml
└── requirements.txt
```

## Dependencies

The reported SD1.5 experiment was run on:

- Ubuntu 22.04;
- Python 3.12;
- PyTorch 2.8.0 + CUDA 12.8;
- one NVIDIA RTX 5090 (32 GB).

Other modern NVIDIA GPUs can be used with the corresponding PyTorch wheel.
The released protocol requires CUDA.

### 1. Clone the repository

```bash
git clone https://github.com/alllllinnrann/interfacemark-watermark.git
cd interfacemark-watermark
```

### 2. Create the environment

```bash
conda create -n interfacemark python=3.12 -y
conda activate interfacemark

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[test]"
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

Run the mathematical unit tests:

```bash
pytest -q
```

### 3. Download Stable Diffusion 1.5

Model weights are not redistributed by this repository.

```bash
hf auth login
mkdir -p models

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir models/sd15
```

For networks that require the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/cache/huggingface"
```

The default configurations expect the checkpoint at `models/sd15`. Edit the
`model` field in the YAML file if it is stored elsewhere.

## Smoke reproduction

The smoke configuration uses 8 design samples, 16 clean calibration images,
and 16 paired test cases. It checks the complete workflow but is not a
paper-scale statistical evaluation.

```bash
bash scripts/run_full.sh configs/sd15_terminal_smoke.yaml
```

Successful completion creates:

```text
runs/sd15_terminal_smoke/
├── PREPARE_COMPLETE
├── CALIBRATION_COMPLETE
├── TEST_COMPLETE
├── EVALUATION_COMPLETE
├── config.resolved.json
├── controls.json
├── carriers.pt
├── splits/
│   ├── calibration/00000/
│   │   ├── clean.png
│   │   └── summary.json
│   └── test/00000/
│       ├── clean.png
│       ├── tail_coupled.png
│       ├── shuffled_tail.png
│       ├── fixed_rms.png
│       ├── fixed_quality.png
│       └── summary.json
├── summary.json
└── summary.csv
```

Each sample record includes the prompt, seed, applied displacement, correct-
and wrong-key scores, paired PSNR/RGB MSE, runtime, and peak allocated GPU
memory.

## Formal 5,000/5,000 reproduction

The formal configuration exactly specifies:

- 640 independent design samples;
- 64 of them for Fixed-quality selection;
- 5,000 clean calibration images;
- 5,000 unseen clean/watermarked test pairs;
- 30 sampling steps and classifier-free guidance 7.5;
- tail probability \(p=2.5\times10^{-7}\);
- independent correct- and wrong-key carrier seeds;
- a target false-positive rate of 1%.

Run the resumable pipeline:

```bash
tmux new -s interfacemark
conda activate interfacemark
cd /path/to/interfacemark-watermark

bash scripts/run_full.sh configs/sd15_terminal_5000.yaml \
  2>&1 | tee runs/sd15_terminal_5000.log
```

Detach without stopping the run with `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t interfacemark
```

The wrapper is resumable. Existing shards are accepted only when their
configuration hash, metadata, and image inventory are complete.

## Run stages separately

The wrapper above executes these three stages.

### Stage A: prepare

```bash
interfacemark-prepare --config configs/sd15_terminal_5000.yaml
```

This command:

1. deterministically creates disjoint design/calibration/test manifests;
2. creates the correct and wrong secret carriers;
3. computes Tail-coupled displacements;
4. constructs the fixed Shuffled-tail permutation;
5. fits Fixed-RMS on all 640 design displacements;
6. fits Fixed-quality on 64 design images only;
7. stores hashes and resolved settings.

### Stage B: generate and score

```bash
interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split calibration

interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split test
```

For every test case, SD1.5 is sampled once to obtain the clean terminal latent.
The four watermark variants are then produced from that shared latent. This
keeps prompts, initial noise, and generated content paired across variants.

To process a selected resumable range:

```bash
interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split test \
  --start 1000 \
  --count 250
```

### Stage C: audit and evaluate

```bash
interfacemark-evaluate --input runs/sd15_terminal_5000
```

Evaluation fails closed if a sample, image, score, seed, or configuration hash
is missing or inconsistent. The threshold is derived only from the clean
calibration split. The test split reports:

- AUC;
- TPR at the independently calibrated 1% FPR;
- observed test FPR and Wilson 95% interval;
- wrong-key AUC;
- paired PSNR and RGB MSE;
- runtime and peak GPU memory.

## Reference SD1.5 results

The released 5,000-calibration/5,000-test run produced:

| Variant | AUC | TPR at calibrated 1% FPR | Wrong-key AUC | Mean PSNR |
|---|---:|---:|---:|---:|
| Tail-coupled | 0.999635 | 99.46% | 0.5064 | 35.04 dB |
| Shuffled-tail | 0.998765 | 98.58% | 0.5062 | 35.04 dB |
| Fixed-RMS | 0.999639 | 99.60% | 0.5064 | 34.80 dB |
| Fixed-quality | 0.999791 | 99.76% | 0.5068 | 34.32 dB |

The observed clean test FPR was 1.26%. The score used here is always the
training-free analytic carrier projection; no learned detector or voting
ensemble is involved.

## Reproducibility notes

- Keep calibration and test outputs separate.
- Do not tune the threshold, carrier, tail probability, or Fixed-quality
  displacement on the test split.
- Use `summary.json` and completion markers rather than counting files by hand.
- The prompt table is versioned and its SHA-256 is stored in every resolved
  configuration.
- The repository does not include attacks or robustness training; this release
  reproduces the clean terminal-interface study only.
