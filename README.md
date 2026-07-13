# DriveWorld-lite V2

第一版实现目标：nuScenes `CAM_FRONT`，历史 8 帧 + future Ego trajectory，预测未来 16 帧，6 Hz，256×448。

## 当前实现

- dependency-free nuScenes/CAN 原始 JSON reader；
- 6 Hz clip sampling、相机最近邻匹配、CAN/ego_pose 插值和 anchor ego frame；
- train-scene-only Ego normalization、逐字段 valid mask、versioned JSONL manifest；
- 数据合同校验和 RGB + 数值 + future trajectory GIF；
- last-frame、无/有 Ego 3D U-Net baseline；
- frozen CogVideoX VAE adapter、时间 padding/crop、masked latent diffusion；
- Ego Fourier/Transformer encoder、temporal cross-attention 和 AdaFiLM；
- checkpoint/EMA/RNG 恢复、反事实轨迹编辑、评估与推理入口；
- `train.py` 必须显式传入 `--start-training`，避免误启动训练。

## 数据准备与校验

当前 mini 数据位于 `data/nuscenes-mini`（已加入 `.gitignore`）：

```bash
python -m scripts.build_front_clips \
  --config configs/data/nuscenes_front_8x16_6hz.yaml

python -m scripts.validate_dataset \
  artifacts/manifests/nuscenes-mini-front-8x16-6hz/train.jsonl \
  artifacts/manifests/nuscenes-mini-front-8x16-6hz/val.jsonl \
  --data-root data/nuscenes-mini

python -m scripts.render_clip \
  artifacts/manifests/nuscenes-mini-front-8x16-6hz/val.jsonl \
  --data-root data/nuscenes-mini --output artifacts/val_clip.gif
```

mini 只有 10 个 scene、约 200 秒视频。它适合数据测试、模型 smoke test 和 tiny overfit，**不够用于正式训练或泛化结论**。正式训练需要 nuScenes `v1.0-trainval` 和对应相机 sweeps。

## 测试

默认环境可运行数据测试；模型测试使用含 PyTorch 的环境：

```bash
python -m pytest -q
conda run -n DriveWorld python -m scripts.smoke_models
conda run -n driveworld-v2 python -m scripts.smoke_vae --frames 8 --height 64 --width 64
```

`smoke_models` 只使用极小随机张量检查 forward/backward/sample，不读取训练集，也不执行 optimizer step。

冻结 VAE 的真实 nuScenes 重建测试（同样不会训练）：

```bash
conda run -n driveworld-v2 python -m scripts.reconstruct_vae \
  --frames 8 --index 0 --output artifacts/vae_reconstruction_8f.png
```

## 基线评估

```bash
python evaluate.py --task last-frame --max-clips 100
```

## 训练（不会自动启动）

```bash
python train.py \
  --task baseline \
  --model-config configs/model/unet3d_baseline.yaml \
  --train-config configs/train/debug.yaml \
  --start-training
```

正式扩散配置已指向本地 `pretrained/vae`。该目录需要包含标准命名的
`config.json` 和 `diffusion_pytorch_model.safetensors`；权重目录已加入 `.gitignore`。
`identity_debug` VAE 仍可在临时配置中用于纯接口测试，但不能用于训练。

更完整的阶段设计和验收门禁见 [PLAN_TODO.md](PLAN_TODO.md)。

单帧 anchor、SingleViewSTDiT、Rectified Flow 和渐进训练的 V2 实现状态与命令见
[V2_IMPLEMENTATION.md](V2_IMPLEMENTATION.md)。

本机 16 GB 调试、latent cache、断点恢复和多 4090 DDP 操作见
[TRAINING.md](TRAINING.md)。

本机实测结果、已知问题与最终验收结论见
[LOCAL_ACCEPTANCE.md](LOCAL_ACCEPTANCE.md)。

当前已下载的partial trainval覆盖、manifest统计和对应训练配置见
[DATASET_TRAINVAL_PARTIAL.md](DATASET_TRAINVAL_PARTIAL.md)。
