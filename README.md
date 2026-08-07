# InterfaceMark

Official implementation of **Terminal InterfaceMark**, a protocol for one-bit watermark-presence detection in frozen latent diffusion models.

## Repository Structure

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
│   ├── core.py        # carrier construction, tail transport, and four displacement laws
│   ├── sd15.py        # frozen SD1.5/VAE generation channel
│   ├── prepare.py     # data splits, carriers, and design-set controls
│   ├── generate.py    # resumable image generation and scoring
│   ├── evaluate.py    # auditing, calibration, AUC/TPR/FPR, and image quality
│   └── records.py     # hashes and atomic records
├── tests/
│   └── test_core.py
├── pyproject.toml
└── requirements.txt
```

## Dependencies

The reported SD1.5 experiments were conducted with:

* Ubuntu 22.04;
* Python 3.12;
* PyTorch 2.8.0 + CUDA 12.8;
* one NVIDIA RTX 5090 GPU (32 GB).

Other recent NVIDIA GPUs can be used with the corresponding PyTorch build. The released implementation requires CUDA.

### 1. Clone the Repository

```bash
git clone https://github.com/xuejianhuang/InterfaceMark.git
cd InterfaceMark
```

### 2. Create the Environment

```bash
conda create -n interfacemark python=3.12 -y
conda activate interfacemark

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[test]"
```
Run the mathematical unit tests:

```bash
pytest -q
```

### 3. Download Stable Diffusion 1.5

Model weights are not redistributed with this repository.

```bash
hf auth login
mkdir -p models

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir models/sd15
```

For networks that require a Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/cache/huggingface"
```

The default configurations expect the checkpoint at `models/sd15`. If the model is stored elsewhere, update the `model` field in the corresponding YAML file.

## Formal 5,000/5,000 Reproduction

The formal configuration specifies:

* 640 independent design samples;
* 64 design samples for Fixed-quality selection;
* 5,000 clean calibration images;
* 5,000 unseen clean/watermarked test pairs;
* 30 sampling steps with classifier-free guidance of 7.5;
* tail probability (p = 2.5 \times 10^{-7});
* independent carrier seeds for the correct and wrong keys;
* a target false-positive rate of 1%.

Run the resumable pipeline:

```bash
tmux new -s interfacemark
conda activate interfacemark
cd /path/to/interfacemark-watermark

bash scripts/run_full.sh configs/sd15_terminal_5000.yaml \
  2>&1 | tee runs/sd15_terminal_5000.log
```

Detach from the session without stopping the run using `Ctrl+B`, followed by `D`. Reconnect with:

```bash
tmux attach -t interfacemark
```

The pipeline is resumable. Existing shards are reused only when their configuration hashes, metadata, and image inventories are complete and consistent.

## Running Individual Stages

The wrapper above executes the following three stages.

### Stage A: Prepare

```bash
interfacemark-prepare --config configs/sd15_terminal_5000.yaml
```

This command:

1. deterministically constructs disjoint design, calibration, and test manifests;
2. generates the correct-key and wrong-key secret carriers;
3. computes the Tail-coupled displacements;
4. constructs the fixed Shuffled-tail permutation;
5. fits Fixed-RMS using all 640 design displacements;
6. fits Fixed-quality using only 64 design images;
7. stores hashes and fully resolved configuration settings.

### Stage B: Generate and Score

```bash
interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split calibration

interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split test
```

For each test sample, SD1.5 is sampled once to obtain the clean terminal latent. The four watermark variants are then generated from the same terminal latent, keeping the prompt, initial noise, and generated content paired across variants.

To process a selected resumable range:

```bash
interfacemark-generate \
  --input runs/sd15_terminal_5000 \
  --split test \
  --start 1000 \
  --count 250
```

### Stage C: Audit and Evaluate

```bash
interfacemark-evaluate --input runs/sd15_terminal_5000
```

Evaluation fails closed if any sample, image, score, seed, or configuration hash is missing or inconsistent. The detection threshold is determined exclusively from the clean calibration split.

The test split reports:

* AUC;
* TPR at the independently calibrated 1% FPR threshold;
* observed test FPR with a Wilson 95% confidence interval;
* wrong-key AUC;
* paired PSNR and RGB MSE;
* runtime and peak GPU memory usage.
