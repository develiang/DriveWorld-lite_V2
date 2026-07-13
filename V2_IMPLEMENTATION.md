# DriveWorld-lite V2 实现说明

V2 面向单张 `CAM_FRONT` anchor image + future Ego trajectory → 16帧未来视频。
当前代码已完成可运行核心，但没有启动训练，也没有声称获得清晰生成质量。

## 已实现

- `SingleViewSTDiT`：factorized spatial/temporal attention；
- 动态2D spatial与1D temporal position embedding；
- 单帧/多帧 history mask 和 zero-init history control residual；
- future Ego 从16个RGB时间点 valid-aware 对齐到5个future latent时间点；
- masked conditional Rectified Flow，`t=0` noise、`t=1` clean；
- logit-normal timestep、Euler/Heun、CFG；
- 单帧anchor cache协议与VAE权重/版本/posterior/padding fingerprint；
- 逐future-latent loss、逐帧PSNR/MAE/edge retention；
- EMA warmup和预训练checkpoint shape/coverage审计；
- 本机、云端和接口smoke配置。

主要文件：

```text
driveworld/models/single_view_stdit.py
driveworld/diffusion/rectified_flow.py
driveworld/models/vae_protocol.py
driveworld/models/pretrained.py
driveworld/evaluation/horizon_metrics.py
configs/model/single_view_stdit_rf_v2_*.yaml
configs/train/v2_*.yaml
```

## 不训练的本机验证

```bash
taskset -c 0-7,10-27 conda run -n driveworld-v2 \
  python -m scripts.smoke_v2 --sample-steps 2
```

已实测：

```text
SingleViewSTDiT trainable parameters: 16,057,104
anchor RGB frames / latent frames: 1 / 1
future RGB frames / latent frames: 16 / 5
generated shape: [1,16,3,256,448]
finite: true
peak allocated VRAM: 9.483 GB
optimizer steps: 0
```

## V2 latent cache

V2 单帧模式不能复用第一版8帧history cache。必须用V2模型配置重新编码：

```bash
conda activate driveworld-v2
python -m scripts.cache_vae_latents \
  --data-config configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml \
  --model-config configs/model/single_view_stdit_rf_v2_local.yaml \
  --split train --output artifacts/latent_cache_v2_single_image

python -m scripts.cache_vae_latents \
  --data-config configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml \
  --model-config configs/model/single_view_stdit_rf_v2_local.yaml \
  --split val --output artifacts/latent_cache_v2_single_image
```

cache hash包含VAE权重SHA256、diffusers版本、posterior mode、padding协议和
`condition_history_frames=1`。训练时传入旧cache会因history latent shape不符而直接失败。

## 预训练权重审计

加载任何Video DiT/STDiT权重前先执行：

```bash
python -m scripts.audit_pretrained_stdit \
  --model-config configs/model/single_view_stdit_rf_v2_local.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output artifacts/pretrained_audit.json
```

只有参数名和shape匹配的权重会进入兼容集合。正式训练配置可增加：

```yaml
pretrained_denoiser: /path/to/checkpoint.pt
pretrained_min_coverage: 0.5
```

coverage低于阈值会终止，避免“看似加载成功，实际几乎随机初始化”。MagicDrive原始
STDiT参数命名与当前single-view实现不保证直接兼容，需要根据审计报告编写显式映射。

## 训练入口（仅供用户显式启动）

在线VAE本机配置：

```bash
python train.py \
  --task diffusion \
  --data-config configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml \
  --model-config configs/model/single_view_stdit_rf_v2_local.yaml \
  --train-config configs/train/v2_local_single_image.yaml \
  --run-steps 100 \
  --start-training
```

建议正式训练使用重新生成的V2 latent cache，并先完成16-clips/S1/S2门禁。云端配置为
`single_view_stdit_rf_v2_cloud.yaml` + `v2_multi_4090.yaml`；多卡、预训练映射和完整
nuScenes训练尚未实测，不能直接视为最终大规模训练配方。

## 仍待完成

- 取得并映射兼容的预训练Video DiT/STDiT权重；
- local Ego cross-attention与当前aligned AdaLN消融；
- camera intrinsics、depth和semantic可选条件；
- variable bucket和S1～S5渐进训练runner；
- LPIPS、flow warping、flicker、FVD与K-seed评估；
- 5070 Ti backward显存门禁及2×/4×4090 DDP/FSDP实测；
- 完整nuScenes trainval和最终清晰度/稳定性G0～G5验收。

