# 本机阶段验收报告

日期：2026-07-12

## 1. 环境

- GPU：NVIDIA GeForce RTX 5070 Ti 16 GB；
- Python 环境：`driveworld-v2`；
- PyTorch：2.7.1 + CUDA 12.8；
- 冻结 VAE：CogVideoX VAE，16 latent channels，空间压缩 8，时间压缩 4；
- 本机任务：CAM_FRONT，4 history → 8 future，6 Hz，128×224。

## 2. 数据和缓存

- mini manifest：747 train / 216 val clips；
- 完整 latent cache：747 train / 216 val，共 72 MB；
- cache 目录：`artifacts/latent_cache_4x8_full/7f254652af22c136`；
- 16-clips latent 统计：mean=-0.0049、std=1.182、范围约 [-4.00, 3.88]；
- 所有缓存 tensor 有限，train/val index 完整。

长循环期间曾发生进程级段错误。内核日志显示崩溃也发生在 Python、PIL 和 apport，且集中于逻辑 CPU 8/9，因此根因指向该 CPU 对或 CPU/RAM 超频稳定性，而非 CogVideoX VAE 单一调用栈。在线训练启动器现会在本机自动排除 CPU 8/9；冻结 VAE 永久保持 eval，验证边界同步并清理 CUDA allocator。缓存文件仍使用原子写入并支持续跑。

## 3. 工程验收

- 单元/合同测试：11 passed；
- online RGB→VAE→denoiser optimizer smoke：通过；
- cached latent 训练：通过；
- bf16、gradient checkpointing、EMA、梯度累积：通过；
- NaN/Inf、梯度、LR、显存、吞吐监控：通过；
- 固定验证：通过；
- checkpoint model/optimizer/scheduler/scaler/EMA/RNG/config：通过；
- checkpoint 排除冻结 VAE：通过；
- checkpoint resume：通过；
- EMA/raw 推理加载：通过；
- Diffusers 官方 DDIM v-prediction 采样：通过；
- 分段训练 `--run-steps`：通过。

在当前驱动组合下，单进程连续运行到 10000 步后最终 CUDA 状态可能在保存阶段崩溃。使用 `--run-steps 500`～`2000` 分段保存/恢复可以稳定规避，已从 step 7500 恢复、运行100步并成功保存完整 `last.pt@7600`。

## 4. 模型结果

### 4.1 小模型管线验证

- 参数量：1.45M；
- 50-step cached 训练无 NaN/Inf；
- 峰值显存约0.05 GB；
- 能降低高噪声 loss，但低 timestep loss 约1.0，采样失败；
- 结论：只用于接口 smoke，不用于画质。

### 4.2 16-clips 质量过拟合

- 参数量：22.48M；
- 配置：`latent_diffusion_local_quality.yaml`；
- 训练：2000 steps、mixed-low timestep sampling；
- 峰值显存：0.53 GB；
- 低 timestep 明显改善：t=50 loss从约0.90降到0.31，t=100降到0.16；
- 固定 train loss：0.403@500 → 0.301@2000；
- 生成道路结构清晰连续，证明训练与逆扩散闭环成立；
- 五种 Ego 条件产生非零差异：相对 original 的 latent/video mean-abs 为 straight 0.0116、left 0.0124、right 0.0224、stop 0.0164；由于16条数据连续且以直行为主，视觉控制仍较弱。

推荐本机过拟合 checkpoint：

```text
artifacts/runs/local-quality-overfit16-4x8/last.pt
```

生成样例：

```text
artifacts/local_quality_original_raw.gif
artifacts/local_quality_counterfactual_raw/counterfactual_grid.gif
```

### 4.3 完整 mini 训练

- 数据：747 train / 216 val；
- 参数量：22.48M；
- 计算已执行到10000 steps，全程无 NaN/Inf，约79 optimizer steps/s；
- 峰值显存：0.54 GB；
- val loss 最佳点约在 step 2500（0.639），之后逐渐过拟合；
- 有效 checkpoint：2500、5000、7500，以及分段恢复保存的 last@7600；
- mini 的独立 val scene 采样仍不稳定，不能视为泛化完成。

推荐本机 mini checkpoint：

```text
artifacts/runs/local-mini-full-4x8/step-0002500.pt
```

短训练 checkpoint 使用 raw 权重通常优于 decay=0.9999 的滞后 EMA；本机配置现已改为 EMA decay=0.999。云端长训练仍可使用0.9999。

## 5. 验收结论

| 项目 | 状态 |
|---|---|
| 数据管线 | 通过 |
| 冻结VAE与latent cache | 通过，长缓存需可恢复执行 |
| 单卡稳定训练 | 通过 |
| 数值/显存监控 | 通过 |
| checkpoint与恢复 | 通过，推荐分段进程 |
| 逆扩散生成闭环 | 通过 |
| 16-clips overfit | 通过 |
| 完整mini泛化 | 未通过，数据不足且scene少 |
| 强Ego反事实控制 | 弱通过：有非零响应，视觉差异有限 |
| 多卡4090 | 代码就绪，等待服务器实机NCCL验收 |

本机目标“网络可调通、可稳定训练、可恢复、可采样”已经完成。下一阶段不应继续在mini上追求泛化画质，而应迁移到 `v1.0-trainval` 和多4090服务器。
