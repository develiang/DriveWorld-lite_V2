# Partial nuScenes Trainval 数据报告

日期：2026-07-12

## 下载覆盖

- 数据目录：`data/nuscenes-trainval`；
- 磁盘占用：约46 GB；
- metadata：完整 `v1.0-trainval`，850 scenes；
- CAM_FRONT：3376 keyframes + 16095 sweeps；
- CAN pose：979 scenes；
- 实际完整 CAM_FRONT scenes：85，其中官方train 62、官方val 23。

只有图像和metadata都存在的连续窗口会进入manifest；缺图窗口会标记为 `missing_image` 并拒绝。未下载scene不会产生训练样本。

## 8→16 / 256×448

配置：`configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml`

- train：4381 clips / 62 scenes；
- val：1624 clips / 23 scenes；
- total：6005 clips；
- 时间误差 p95：33.333 ms；
- 最大已接受时间误差：train 54.018 ms、val 51.666 ms；
- CAN steering有效率：100%；
- scene泄漏：0；
- 随机检查：train/val各100 clips，共4800张图像，错误0。

产物：

```text
artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/
artifacts/dataset_validation_trainval_partial_8x16.json
artifacts/trainval_partial_val_clip.gif
```

## 4→8 / 128×224

配置：`configs/data/nuscenes_front_4x8_6hz_trainval_partial.yaml`

- train：5586 clips / 62 scenes；
- val：2071 clips / 23 scenes；
- total：7657 clips；
- 时间误差 p95：33.333 ms；
- CAN steering有效率：100%；
- scene泄漏：0；
- 随机检查：train/val各100 clips，共2400张图像，错误0。

产物：

```text
artifacts/manifests/nuscenes-trainval-partial-front-4x8-6hz/
artifacts/dataset_validation_trainval_partial_4x8.json
```

## 推荐使用顺序

1. 本机先使用4→8和22.48M模型检查新数据训练曲线；
2. 通过后切换8→16，仍先使用22.48M模型；
3. 多4090服务器使用8→16和71.1M模型；
4. 三种训练均推荐先生成冻结VAE latent cache；
5. 当前数据比mini大约7.8倍，但仍只有完整trainval的10%，不能当作最终全量实验。

预计FP16 latent cache大小：4→8约0.55 GB；8→16约2.8 GB，不包含索引和少量序列化开销。

一键重新构建和验证manifest：

```bash
conda activate driveworld-v2
./scripts/prepare_partial_trainval.sh
```

构建可恢复的冻结VAE cache：

```bash
MODE=4x8 ./scripts/cache_partial_trainval.sh
MODE=8x16 ./scripts/cache_partial_trainval.sh
```

当前PyTorch/CUDA组合长时间连续调用CogVideoX VAE时可能发生底层崩溃。cache wrapper默认每个worker只新增100个文件后主动退出、冷却2秒并启动新进程；若worker仍异常则冷却5秒重试。已有文件通过原子rename保存，续跑不会重新解码或反序列化。

如果机器上仍出现段错误，可进一步缩短worker生命周期：

```bash
MODE=4x8 CHUNK_SIZE=25 COOLDOWN_SECONDS=10 ./scripts/cache_partial_trainval.sh
```

## 训练配置

| 场景 | Data | Model | Train |
|---|---|---|---|
| 本机快速调试 | `nuscenes_front_4x8_6hz_trainval_partial.yaml` | `latent_diffusion_local_quality.yaml` | `trainval_partial_local_4x8.yaml` |
| 本机正式调试 | `nuscenes_front_8x16_6hz_trainval_partial.yaml` | `latent_diffusion_local_quality.yaml` | `trainval_partial_local_8x16.yaml` |
| 多4090 | `nuscenes_front_8x16_6hz_trainval_partial.yaml` | `latent_diffusion_multi_4090.yaml` | `trainval_partial_multi_4090.yaml` |

本机4→8启动示例：

```bash
DATA_CONFIG=configs/data/nuscenes_front_4x8_6hz_trainval_partial.yaml \
MODEL_CONFIG=configs/model/latent_diffusion_local_quality.yaml \
TRAIN_CONFIG=configs/train/trainval_partial_local_4x8.yaml \
LATENT_CACHE=artifacts/latent_cache_trainval_partial_4x8/<config-hash> \
RUN_STEPS=1000 \
./scripts/launch_local_16gb.sh
```

本机8→16启动示例：

```bash
DATA_CONFIG=configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml \
MODEL_CONFIG=configs/model/latent_diffusion_local_quality.yaml \
TRAIN_CONFIG=configs/train/trainval_partial_local_8x16.yaml \
LATENT_CACHE=artifacts/latent_cache_trainval_partial_8x16/<config-hash> \
RUN_STEPS=1000 \
./scripts/launch_local_16gb.sh
```

四张4090启动示例：

```bash
NPROC_PER_NODE=4 \
DATA_CONFIG=configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml \
MODEL_CONFIG=configs/model/latent_diffusion_multi_4090.yaml \
TRAIN_CONFIG=configs/train/trainval_partial_multi_4090.yaml \
LATENT_CACHE=artifacts/latent_cache_trainval_partial_8x16/<config-hash> \
RUN_STEPS=1000 \
./scripts/launch_multi_4090.sh
```

首次执行建议保留 `RUN_STEPS=1000`。确认loss、验证、checkpoint和恢复正常后，再连续提交多个分段任务。

### 不使用cache：本机在线VAE训练

如果希望训练阶段直接读取RGB并调用冻结VAE，使用专用短进程wrapper：

```bash
conda activate driveworld-v2
SEGMENT_STEPS=100 ./scripts/train_online_vae_8x16.sh
```

该模式不读取latent cache。每个optimizer step会分别编码8帧history和16帧future；每100步保存 `last.pt`、退出并用新CUDA进程恢复，以规避当前环境下CogVideoX VAE长循环的原生段错误。在线VAE吞吐显著低于cached模式。
