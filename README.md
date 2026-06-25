# SFP-DeFT

**SFP-DeFT** is a complete, code-integrated project based on the official SFP
repository. It adds **DeFT: Density-guided Feature Tracing** as a
training-free module for open-vocabulary semantic segmentation.

The central story is:

> Support density matters for training-free dense vision-language transfer.

DeFT does not train new parameters. It refines frozen SFP/CLIP visual tokens
after backbone feature extraction and before text-image similarity.

## Project Layout

```text
SFP-DeFT-complete/
  SFP_ICCV/                 # original SFP ICCV code, now integrated with DeFT
    eval.py                 # supports --deft and ablation flags
    SFP_segmentor.py        # calls DeFT before text-image similarity
    resfp/                  # DeFT implementation
    configs/                # original SFP dataset configs
  SFP-D_Journal/            # original journal extension folder
  paper/                    # AAAI-style LaTeX and Markdown draft
  experiments/              # ablation matrix
  results/                  # result log template
  tools/                    # predownload, result logging, visualization helpers
  scripts/                  # AutoDL setup and quick-run scripts
```

## Quick Start on AutoDL

```bash
cd /
mkdir -p /root/autodl-tmp/src /root/autodl-tmp/hf-cache /root/autodl-tmp/data
cd /root/autodl-tmp/src

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

git clone --depth 1 https://github.com/ybh1230/SFP-DeFT.git
cd SFP-DeFT

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache

bash scripts/bootstrap_autodl.sh
```

If GitHub clone fails:

```bash
cd /root/autodl-tmp/src
wget -O SFP-DeFT.zip https://codeload.github.com/ybh1230/SFP-DeFT/zip/refs/heads/main
unzip SFP-DeFT.zip
mv SFP-DeFT-main SFP-DeFT
cd SFP-DeFT
bash scripts/bootstrap_autodl.sh
```

## Environment

Use the original SFP environment:

```bash
cd /root/autodl-tmp/src/SFP-DeFT/SFP_ICCV
conda env create -f environment.yml
conda activate sfp_ovss
pip install -U huggingface_hub
```

If the conda file is slow or fails, follow the manual setup in
`SFP_ICCV/README.md`.

## HuggingFace Cache

```bash
cd /root/autodl-tmp/src/SFP-DeFT
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache

bash scripts/predownload_clip.sh
```

## Run Baseline and DeFT

Edit dataset paths in `SFP_ICCV/configs/*.py` first.

Baseline:

```bash
cd /root/autodl-tmp/src/SFP-DeFT/SFP_ICCV
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_sfp
```

DeFT:

```bash
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

Save raw DeFT visualization maps:

```bash
python eval.py \
  --config ./configs/cfg_voc21.py \
  --work-dir ./work_logs/voc21_deft_vis \
  --deft \
  --deft-return-maps \
  --deft-save-maps ./work_logs/voc21_deft_vis/deft_maps
```

## Ablations

```bash
cd /root/autodl-tmp/src/SFP-DeFT

python tools/run_eval_matrix.py \
  --workdir ./SFP_ICCV \
  --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \
  --work-root ./SFP_ICCV/work_logs/voc21_matrix
```

Manual ablations:

```bash
cd SFP_ICCV
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/no_graph --deft --deft-graph-weight 0.0 --deft-local-weight 1.0
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/no_local --deft --deft-graph-weight 1.0 --deft-local-weight 0.0
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/no_multiscale --deft --deft-multiscale-strength 0.0
```

## Paper Draft

- LaTeX: `paper/main.tex`
- Markdown preview: `paper/main.md`

The tables intentionally contain placeholders. Fill them after real experiments.

## Recommended GPU

RTX 4090 24GB is enough for pre-experiments, main inference, ablations, and
visualization. Start with batch size 1 and `num_workers=4`, then increase only
after the baseline is stable.
