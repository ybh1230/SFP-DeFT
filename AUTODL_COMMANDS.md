# AutoDL Commands

## Clone

```bash
cd /
mkdir -p /root/autodl-tmp/src /root/autodl-tmp/hf-cache /root/autodl-tmp/data
cd /root/autodl-tmp/src

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

git clone --depth 1 https://github.com/ybh1230/SFP-DeFT.git
cd SFP-DeFT
```

Fallback:

```bash
wget -O SFP-DeFT.zip https://codeload.github.com/ybh1230/SFP-DeFT/zip/refs/heads/main
unzip SFP-DeFT.zip
mv SFP-DeFT-main SFP-DeFT
cd SFP-DeFT
```

## Environment Variables

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
```

## Setup

```bash
bash scripts/bootstrap_autodl.sh

cd SFP_ICCV
conda env create -f environment.yml
conda activate sfp_ovss
pip install -U huggingface_hub
```

## Predownload CLIP

```bash
cd /root/autodl-tmp/src/SFP-DeFT
bash scripts/predownload_clip.sh
```

## Run VOC21 Baseline

```bash
cd /root/autodl-tmp/src/SFP-DeFT/SFP_ICCV
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_sfp
```

## Run VOC21 DeFT

```bash
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

## Run Matrix

```bash
cd /root/autodl-tmp/src/SFP-DeFT
python tools/run_eval_matrix.py \
  --workdir ./SFP_ICCV \
  --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \
  --work-root ./SFP_ICCV/work_logs/voc21_matrix
```

## Record Result

```bash
python tools/record_result.py \
  --dataset VOC21 \
  --method DeFT \
  --backbone CLIP-ViT-B/16 \
  --seed 0 \
  --mIoU VALUE \
  --mAcc VALUE \
  --boundary_F VALUE \
  --small_object_mIoU VALUE \
  --fps VALUE \
  --gpu RTX4090 \
  --notes "main result"
```
