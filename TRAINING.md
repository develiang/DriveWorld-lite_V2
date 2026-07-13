# DriveWorld-lite V2 训练手册

## 1. 已验证环境

- 本机实测 GPU：RTX 5070 Ti 16 GB；配置同样适用于 4070 Ti 16 GB；
- Conda：`driveworld-v2`；PyTorch 2.7.1 + CUDA 12.8；
- 冻结 CogVideoX VAE：`pretrained/vae`；
- 本机调试：4 history → 8 future、128×224；
- 云端训练：8 history → 16 future、256×448、cached latent + DDP。

所有训练入口都必须显式传入 `--start-training`。默认不会自动开始训练。

## 2. 本机 16 GB 调试

### 2.1 构建 4→8 manifest

```bash
conda run -n driveworld-v2 python -m scripts.build_front_clips \
  --config configs/data/nuscenes_front_4x8_6hz_128x224.yaml
```

### 2.2 在线 VAE 调试

在线模式每个 batch 读取 RGB 并执行冻结 VAE，便于检查完整链路：

```bash
conda activate driveworld-v2
DATA_CONFIG=configs/data/nuscenes_front_4x8_6hz_128x224.yaml \
MODEL_CONFIG=configs/model/latent_diffusion_local_16gb.yaml \
TRAIN_CONFIG=configs/train/local_16gb.yaml \
./scripts/launch_local_16gb.sh
```

在线模式已实测峰值显存约 1.7 GB（micro batch 1、小 denoiser），但吞吐受 VAE 限制。

当前 8→16、256×448 在线 VAE 使用可恢复的分段启动器：

```bash
SEGMENT_STEPS=100 MAX_FAILURES=20 COOLDOWN_SECONDS=10 \
./scripts/train_online_vae_8x16.sh
```

本机内核日志确认历史段错误集中在逻辑 CPU 8/9，并且 Python、PIL 和
系统崩溃收集器都曾在同一 CPU 对上随机地址崩溃，不是单一 VAE Python
调用栈。该启动器在 28 线程本机上会自动使用 `taskset` 排除 CPU 8/9，
且所有 dataloader worker 继承相同 affinity。其他机器默认不限制；可用
`TRAIN_CPUSET=...` 显式覆盖，或用 `TRAIN_CPUSET=''` 禁用。

不启动训练的在线 VAE 压力测试：

```bash
taskset -c 0-7,10-27 conda run -n driveworld-v2 \
  python -m scripts.stress_online_vae --iterations 100
```

此外，CogVideoX VAE adapter 会永久保持 `eval`，验证结束后同步 CUDA 并
清理 allocator cache。这与 MagicDrive 将冻结 VAE 独立于 trainable model、
在验证边界清理显存的处理一致。

### 2.3 推荐：cached latent 调试

分别缓存 train/val：

```bash
conda run -n driveworld-v2 python -m scripts.cache_vae_latents \
  --data-config configs/data/nuscenes_front_4x8_6hz_128x224.yaml \
  --model-config configs/model/latent_diffusion_local_16gb.yaml \
  --split train --output artifacts/latent_cache_4x8

conda run -n driveworld-v2 python -m scripts.cache_vae_latents \
  --data-config configs/data/nuscenes_front_4x8_6hz_128x224.yaml \
  --model-config configs/model/latent_diffusion_local_16gb.yaml \
  --split val --output artifacts/latent_cache_4x8
```

命令会打印带配置 hash 的目录，例如：

```text
artifacts/latent_cache_4x8/f95b1ced241b0243
```

使用缓存训练：

```bash
conda activate driveworld-v2
LATENT_CACHE=artifacts/latent_cache_4x8/f95b1ced241b0243 \
./scripts/launch_local_16gb.sh
```

cached 模式不会加载 VAE，checkpoint 也不会重复保存 VAE 权重。

### 2.4 Tiny-overfit 顺序

不要直接跑完整 mini：

```bash
conda run -n driveworld-v2 python train.py \
  --task diffusion \
  --data-config configs/data/nuscenes_front_4x8_6hz_128x224.yaml \
  --model-config configs/model/latent_diffusion_local_16gb.yaml \
  --train-config configs/train/local_16gb.yaml \
  --latent-cache artifacts/latent_cache_4x8/f95b1ced241b0243 \
  --overfit-clips 16 --max-steps 2000 \
  --start-training
```

先确认 16 clips loss 可明显下降、checkpoint 可恢复，再扩大到一个 scene。

### 2.5 恢复训练

```bash
RESUME=artifacts/runs/local-16gb-4x8/last.pt \
LATENT_CACHE=artifacts/latent_cache_4x8/f95b1ced241b0243 \
./scripts/launch_local_16gb.sh
```

恢复内容包括模型、优化器、LR scheduler、GradScaler、EMA 和随机状态。恢复时应保持原来的总训练步数配置，避免改变 cosine schedule 语义。

本机曾因 CPU 8/9 系统级不稳定而在单进程运行中随机崩溃。启动器现已隔离这两个逻辑 CPU；在完成 BIOS 默认设置、关闭 CPU/RAM 超频或硬件稳定性检查前，仍建议保持全局 `max_steps` 不变，用 `--run-steps` 分段运行：

```bash
python train.py ... --run-steps 1000 --start-training
python train.py ... --resume artifacts/runs/<run>/last.pt --run-steps 1000 --start-training
```

每次恢复都继续原始全局 cosine LR schedule，而不是重新开始学习率。

## 3. 多张 4090 单机训练

### 3.1 数据

编辑：

```text
configs/data/nuscenes_front_8x16_6hz_trainval.yaml
```

把 `data_root` 指向云端 nuScenes `v1.0-trainval`，并确保同目录含 `can_bus`。构建 manifest 需要 `nuscenes-devkit` 来解析官方 train/val scene split。

```bash
conda activate driveworld-v2
python -m scripts.build_front_clips \
  --config configs/data/nuscenes_front_8x16_6hz_trainval.yaml
```

### 3.2 多卡并行缓存

```bash
NPROC_PER_NODE=4 \
DATA_CONFIG=configs/data/nuscenes_front_8x16_6hz_trainval.yaml \
MODEL_CONFIG=configs/model/latent_diffusion_multi_4090.yaml \
OUTPUT=artifacts/latent_cache_trainval \
./scripts/cache_latents_multi_gpu.sh
```

每张 GPU 处理一个 shard，结束后自动合并 train/val index。记录脚本输出的最终 hash 目录。

### 3.3 DDP 启动

```bash
NPROC_PER_NODE=4 \
LATENT_CACHE=artifacts/latent_cache_trainval/<config-hash> \
./scripts/launch_multi_4090.sh
```

默认云端模型约 71.1M trainable parameters。配置为每卡 micro batch 4、梯度累积 2：

```text
effective batch = 4 × 2 × GPU 数
4 GPUs → 32
8 GPUs → 64
```

如果 4090 上 OOM，先把 `configs/train/multi_4090.yaml` 的 `micro_batch_size` 从 4 降到 2，同时把 accumulation 从 2 提到 4，保持 effective batch 不变。

### 3.4 多卡恢复

```bash
NPROC_PER_NODE=4 \
LATENT_CACHE=artifacts/latent_cache_trainval/<config-hash> \
RESUME=artifacts/runs/multi-4090/last.pt \
./scripts/launch_multi_4090.sh
```

当前实现覆盖单机多 GPU `torchrun`。跨多台物理服务器需要额外提供 `MASTER_ADDR`、`MASTER_PORT`、`NNODES` 和 `NODE_RANK`，不属于当前已实测范围。

## 4. 监控和产物

控制台与 TensorBoard 记录：

- train/validation loss；
- gradient norm；
- learning rate；
- peak allocated VRAM；
- optimizer steps/s。

```bash
tensorboard --logdir artifacts/runs
```

checkpoint 默认不包含冻结 VAE，因此云端和本机都必须保留相同的 `pretrained/vae`。推理默认加载 EMA denoiser。

## 5. 当前已验证结果

- 在线真实 VAE 2-step：loss `1.692 → 1.448`，峰值显存 1.71 GB；
- cached latent 50-step：loss `1.257 → 0.498`，无 NaN/Inf；
- 断点恢复：已从 step 2 恢复并继续执行 optimizer step；
- 完整 8→16 latent + 71.1M denoiser：单步 loss 1.532，val loss 1.294，峰值显存 1.64 GB；
- CPU 8/9 隔离后，真实 256×448、8+16 帧 BF16 在线 VAE 连续编码 30 clips 通过，峰值 allocated VRAM 5.061 GB，常驻 allocated 0.826 GB；
- 多 GPU DDP 代码和启动脚本已完成，但本机只有一张 GPU，尚未做真实双卡/NCCL 验收。
