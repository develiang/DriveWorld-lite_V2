# DriveWorld-lite V2 第一版实现方案与 TODO

> 本计划将 `roadmap.md` 中的 M0～M3 落成一版可执行工程方案。第一版只做：
> **nuScenes / CAM_FRONT / 8 帧历史 → 16 帧未来 / 6 Hz / Ego 轨迹条件 / 256×448**。

## 0. 当前状态（2026-07-12）

状态约定：`[x]` 已实现并通过相应非训练测试；`[ ]` 尚未完成；“部分完成”表示已有代码，但还缺验收项。

- 工程代码：P0 基础骨架、M0 数据管线、M1/M2 模型骨架、M3 轨迹编辑与推理入口已落地；
- mini 数据：10 scenes、196.55 秒、2342 个 CAM_FRONT 帧，生成 train 574 / val 192 个高度重叠 clips；
- 数据验证：机械抽检 train/val 各 100 clips、共 4800 张图像，错误 0、scene 泄漏 0、时间误差 p95=33.333 ms；
- 单元测试：22 passed；覆盖真实 Dataset contract、DDPM/RF可逆性、future-only mask、单帧STDiT、Ego顺序响应、EMA兼容和逐帧质量指标；
- VAE：本地 `pretrained/vae` 的 CogVideoX VAE（16 latent channels、空间压缩 8、时间压缩 4）加载成功；
- VAE shape：1×256×448 → 1×16×32×56；8 帧 → 3 latent 帧；16 帧 → 5 latent 帧；padding/crop 重建测试通过；
- 真实图像重建：单帧 PSNR 36.81 dB、8 帧 28.27 dB、16 帧 26.79 dB；样例结构保持良好，仍需多 clip 统计；
- 训练验证：真实 RGB+VAE 2-step、cached latent 50-step、checkpoint resume、完整 8→16 latent + 71.1M denoiser 单步均通过；
- 本机收尾：22.48M 模型完成16-clips 2000-step质量过拟合；完整mini完成10000-step计算，推荐val checkpoint@2500；
- 本机生成：训练clip道路结构可连续生成；Ego条件产生非零响应但控制幅度较弱；完整mini对独立val scene泛化仍不足；
- 尚未进行：16 clips/单 scene 正式 overfit、trainval 完整训练、真实多 GPU NCCL 验收、学习后控制性评估；
- 数据结论：mini 足够做工程/数据/VAE smoke test，不足以正式训练或得出泛化结论；正式阶段需要 `v1.0-trainval`。
- Partial trainval：已准备85个完整scene（train 62 / val 23）；8→16为4381/1624 clips，4→8为5586/2071 clips，随机检查与scene隔离通过；
- V2 核心已落地：single-image anchor、SingleViewSTDiT、显式时空位置、逐 latent Ego 对齐、masked Rectified Flow、Euler/Heun、V2 cache fingerprint 和 EMA warmup；真实数据无训练 smoke 输出 `[1,16,3,256,448]`、峰值 9.483 GB；

## 1. 交付目标与边界

### 1.1 第一版交付物

- 可复现的 CAM_FRONT clip 索引与数据加载器；
- Last-frame、无 Ego 3D U-Net、有 Ego 3D U-Net 三个确定性基线；
- 冻结 Video VAE 的 masked latent video diffusion；
- 支持直行、左转、右转、减速/停车轨迹的反事实推理；
- 固定验证集上的画质、时序、控制性指标与四宫格视频报告；
- 支持断点续训、EMA、固定随机种子和配置快照。

### 1.2 暂不纳入第一版

- 多相机、Map、3D Box、LiDAR/Radar、文本；
- 17→48 直接生成和闭环仿真；
- MagicDrive-V2 主干改造；
- 原始 RGB 全量复制或逐样本保存 tensor 文件。

## 2. 固定技术决策

| 项目 | 第一版决策 |
|---|---|
| 数据 | nuScenes CAM_FRONT，按 scene 官方 split |
| 时间 | 目标 6 Hz，8 history + 16 future，窗口 stride=2 帧 |
| 图像 | 保持宽高比 resize/crop 到 256×448，归一化到 `[-1, 1]` |
| 坐标系 | 最后一帧 history 的 ego frame，`x` 向前、`y` 向左、yaw 逆时针为正 |
| Ego 条件 | `[x,y,yaw,vx,vy,ax,ay,yaw_rate,steering]` + 同维 valid mask |
| 数据存储 | JSONL/Parquet clip manifest + 原图路径；latent 单独 cache |
| 基线 | 小型 encoder-decoder 3D U-Net，未来轨迹通过 temporal encoder + FiLM 注入 |
| 生成模型 | 冻结 Video VAE + masked latent 3D U-Net（先跑通，再考虑 small DiT） |
| 扩散目标 | `v_prediction`，loss 只覆盖 future latent |
| 精度 | A800 使用 bf16；本地调试自动降分辨率和帧数 |
| 配置 | YAML + dataclass 校验；所有运行产物保存 resolved config |

## 3. 工程结构

```text
configs/{data,model,train,experiment}/
driveworld/
  data/           # scene 索引、时间对齐、坐标变换、Dataset
  models/         # baseline、VAE adapter、ego encoder、denoiser
  diffusion/      # noise scheduler、masked loss、sampler
  training/       # trainer、checkpoint、EMA、日志
  evaluation/     # 图像/时序/控制指标、报告
  visualization/  # clip 和轨迹渲染
scripts/          # 构建、校验、缓存、训练、推理入口
tests/            # unit + smoke + tiny overfit
artifacts/        # 本地产物，加入 gitignore
```

入口统一采用：

```bash
python -m scripts.build_front_clips --config configs/data/nuscenes_front_8x16_6hz.yaml
python -m scripts.validate_dataset artifacts/manifests/.../train.jsonl artifacts/manifests/.../val.jsonl
python train.py --task baseline --model-config configs/model/unet3d_baseline.yaml --start-training
python train.py --task diffusion --model-config configs/model/latent_diffusion_ego.yaml --start-training
python evaluate.py --task baseline --checkpoint ...
python inference.py --task diffusion --model-config configs/model/latent_diffusion_ego.yaml --checkpoint ... --trajectory left
```

## 4. 核心接口与数据契约

### 4.1 Clip manifest

manifest 每行只保存可复现索引和对齐后的数值，不复制 RGB：

```json
{
  "clip_id": "<scene_token>:<anchor_us>",
  "scene_token": "...",
  "split": "train",
  "anchor_timestamp_us": 0,
  "image_paths": ["24 relative paths"],
  "image_timestamps_us": [0],
  "target_timestamps_us": [0],
  "past_ego": [[0.0]],
  "future_ego": [[0.0]],
  "past_ego_valid": [[true]],
  "future_ego_valid": [[true]],
  "source_flags": ["ego_pose", "can"],
  "max_time_error_ms": 0.0
}
```

约束：数组长度必须分别为 24、8、16；所有路径相对 `data_root`；manifest 记录 schema version、构建配置 hash 和 nuScenes version。

### 4.2 Dataset 输出

```python
batch = {
    "past_rgb": FloatTensor[B, 8, 3, H, W],
    "future_rgb": FloatTensor[B, 16, 3, H, W],
    "past_ego": FloatTensor[B, 8, 9],
    "future_ego": FloatTensor[B, 16, 9],
    "past_ego_raw": FloatTensor[B, 8, 9],
    "future_ego_raw": FloatTensor[B, 16, 9],
    "past_ego_valid": BoolTensor[B, 8, 9],
    "future_ego_valid": BoolTensor[B, 16, 9],
    "timestamps": Int64Tensor[B, 24],
    "clip_id": list[str],
}
```

训练集图像增强只允许整段 clip 共享同一组空间参数；禁止逐帧随机 crop/flip。第一版默认不开水平翻转，避免 steering/yaw 符号处理错误。

### 4.3 时间对齐

1. 在每个 scene 内构造严格递增的 CAM_FRONT 时间序列；
2. 以 anchor（history 最后一帧）为中心生成间隔 `1/6 s` 的 24 个目标时间；
3. 每个目标时间匹配最近相机帧，当前 mini 实测阈值为 55 ms，超过则整段无效；
4. 连续量线性插值；yaw 先 unwrap，再插值；离散/不可插值字段保留 mask；
5. 禁止跨 scene，禁止重复选择同一图像帧，禁止时间倒序；
6. split 在生成窗口前按 scene 决定，避免泄漏。

> 风险：nuScenes 相机名义频率约 12 Hz，但实际时间抖动会使固定阈值过严。构建脚本必须输出误差分布，再最终锁定阈值。

### 4.4 Ego 坐标变换

以 anchor pose `T_world_anchor` 为基准：

```text
T_anchor_i = inverse(T_world_anchor) @ T_world_i
x, y       = translation(T_anchor_i)[0:2]
yaw        = yaw(rotation(T_anchor_i))
```

速度、加速度从 world frame 旋转到 anchor frame。若 CAN 缺失：位置/yaw 使用 ego_pose；速度/加速度使用带时间间隔的中心差分并做轻量平滑；steering 填 0 且 mask=false。训练前只用 train split 统计 mean/std，验证集复用。

### 4.5 模型接口

```python
pred = baseline(past_rgb, future_ego, future_ego_valid)

loss = diffusion.training_loss(
    past_rgb=past_rgb,
    future_rgb=future_rgb,
    future_ego=future_ego,
    future_ego_valid=future_ego_valid,
)

torch.manual_seed(seed)
video = diffusion.sample(
    past_rgb=past_rgb,
    future_ego=edited_trajectory,
    num_steps=steps,
)
```

所有模型 forward 都要断言 shape、dtype、有限值和时间长度，尽早暴露数据错误。

## 5. 模型实现细节

### 5.1 确定性 3D U-Net

- 输入 history RGB；先用 2D stem 逐帧降采样，再进行 3D residual blocks；
- temporal bottleneck 将 8 帧特征映射到 16 帧，可用 temporal upsample + causal/non-causal blocks；
- Ego encoder：Fourier feature → 2 层 MLP → 2～4 层 temporal Transformer；
- 每个尺度使用 pooled ego token 产生 FiLM `(scale, shift)`；
- 输出 `tanh` RGB，loss 为 Charbonnier + `0.1 LPIPS + 0.1 temporal L1`；
- LPIPS 输入按 `[B*T,3,H,W]` 计算，可在 debug 配置中关闭以提速。

基线的作用是验证管线与条件，不把大量时间投入极致画质。

### 5.2 Masked latent diffusion

VAE adapter 对外统一为 `[B,T,C,H,W] ↔ [B,T,Cz,h,w]`，内部负责第三方 VAE 的维度顺序、缩放因子和 temporal padding。

训练过程：

1. 分别无梯度编码 history/future RGB，以保留明确 latent 边界；若启用 cache，则按 `clip_id + vae_id + preprocess_hash` 读取；
2. 对不满足 CogVideoX `4k+1` 的 8/16 帧分别 pad 到 9/17 帧，得到 3/5 latent 帧，解码后裁回原长度；
3. history latent 保持干净，只对 future latent 采样噪声和 timestep；
4. 输入通道拼接 `[z_mixed, z_history_masked, history_mask]`；
5. denoiser 使用 temporal/spatial attention；future Ego tokens 通过 cross-attention 注入；
6. 用 timestep embedding 和逐时刻 Ego embedding 做 AdaLN；
7. loss mask 严格为 `[0×history, 1×future]`，预测目标为 velocity；
8. 10% ego condition dropout，为后续 classifier-free guidance 保留能力。

必须增加两项单测：history 区域 loss 恒为 0；同 seed 下仅改变 Ego 条件会改变 denoiser 输出。

### 5.3 反事实轨迹构造

第一版不允许任意手画不符合动力学的轨迹。由真实 future trajectory 生成四类编辑：

- straight：逐步衰减横向位移和 yaw；
- left/right：叠加平滑曲率 profile，并约束速度/横向加速度；
- stop：使用非负减速度 profile，使速度平滑降至 0；
- 所有轨迹在 anchor 处位置、yaw、速度连续，输出同时更新 `x/y/yaw/v/a/yaw_rate`。

保存编辑参数和原始轨迹，保证实验可复现。

## 6. 评估与验收

### 6.1 固定评估协议

- 固定 val clip 子集和 seed 列表；
- 确定性指标：PSNR、SSIM、LPIPS；
- 时序指标：frame-difference error、光流方向/幅值误差、history→future 边界误差；
- 控制指标：condition sensitivity、left/right motion sign accuracy、stop flow reduction；
- diffusion 每个 clip 用相同 seed 比较不同 Ego 条件，避免采样噪声混淆；
- 指标按预测时距分桶：0～1 s、1～2 s、2～2.67 s，而不只报均值。

Ego motion alignment 依赖视觉里程计，容易受动态目标影响。第一版将它作为诊断指标，不作为唯一验收门槛；控制验收同时查看光流、车道横移和人工视频。

### 6.2 阶段门禁

- M0：随机 100 clips 可视检查通过；scene 泄漏为 0；时间误差和缺失率有报告；
- M1：16 clips 明显过拟合；有 Ego 基线优于 last-frame 且输出非静止；
- M2：16 clips、单 scene 依次过拟合；采样无首帧跳变/NaN；可恢复训练；
- M3：相同历史和 seed 下，左右轨迹产生方向一致且可观察差异；打乱 Ego 后质量或对齐指标下降。

任何门禁失败时不得直接扩大训练规模。

## 7. 分阶段 TODO

### P0：工程骨架（0.5～1 天）

- [x] `P0-01` 创建 Python package、目录、`.gitignore`、依赖文件；
- [x] `P0-02` 实现 YAML 配置加载、data dataclass 和 resolved config 保存；
- [x] `P0-03` 实现统一 seed、run directory、控制台和 TensorBoard 结构化日志；
- [ ] `P0-04` 部分完成：已有 pytest、compileall、数据与模型 smoke test；尚未实际接入 Ruff CI；
- [x] `P0-05` 写数据根目录、nuScenes version、CAN 路径的环境配置说明。

### M0：数据管线（3～5 天）

- [x] `M0-01` 扫描 scene/CAM_FRONT，建立逐 scene 的有序帧表；依赖 P0；
- [x] `M0-02` 实现 6 Hz 目标时间网格和最近相机帧匹配；
- [x] `M0-03` 实现 clip sampler（8+16、stride=2、禁止跨 scene/缺口）；
- [x] `M0-04` 解析 ego_pose 和 CAN，统一第一版所需字段及单位；
- [ ] `M0-05` 部分完成：连续量/unwrap 角度插值已完成；异常值统计与按字段裁剪尚未实现；
- [x] `M0-06` 实现 anchor ego frame 变换和 train-only 标准化统计；
- [x] `M0-07` 实现 CAN fallback、逐字段 valid mask 和 source flags；
- [x] `M0-08` 输出 versioned manifest、构建统计和 split 泄漏检查；
- [x] `M0-09` 实现 Dataset/DataLoader 和整段一致 resize/crop；
- [x] `M0-10` 实现 RGB + 数值 + BEV future path 检查 GIF；
- [ ] `M0-11` 部分完成：坐标、yaw、插值和真实 Dataset contract 单测已有；需补窗口边界、split 泄漏失败用例；
- [ ] `M0-12` 部分完成：train/val 各机械抽检 100 clips 并输出 JSON，人工目视仅完成样例 GIF，尚无 HTML/100 clips 人工验收。

### M1：确定性基线（4～7 天）

- [x] `M1-01` 实现 last-frame baseline 与统一 evaluate API；依赖 M0；
- [x] `M1-02` 实现可关闭 Ego 的 3D U-Net；
- [x] `M1-03` 实现 Ego temporal encoder 和 FiLM 注入；
- [x] `M1-04` 实现 Charbonnier、可选 LPIPS、temporal loss；
- [ ] `M1-05` 部分完成：trainer、autocast、梯度裁剪、TensorBoard 和固定验证已有；缺自动视频回调；
- [x] `M1-06` checkpoint 保存/恢复 model、optimizer、LR scheduler、GradScaler、EMA、RNG 和 config；
- [ ] `M1-07` 依次完成 16 clips、单 scene、10% 数据门禁；
- [ ] `M1-08` 比较 last-frame / no-Ego / Ego，输出视频与分时距指标。

### M2：Latent diffusion（7～14 天）

- [ ] `M2-01` 部分完成：已锁定本地 CogVideoX VAE，权重完整、真实 1/8/16 帧重建与 shape/padding 通过；CPU 8/9 隔离后完成 30 clips、256×448、8+16 帧 BF16 在线编码压力测试（峰值 5.061 GB）；待做多 clip 重建质量统计和确认权重许可；
- [x] `M2-02` 实现 CogVideoX VAE adapter、local-only 加载、确定性 posterior mode、时间 padding/crop、永久 eval、验证边界显存清理和重建/压力 smoke test；
- [x] `M2-03` 实现带 cache key/version 的 latent cache 和原子写入；尚未执行全量 cache；
- [x] `M2-04` 实现 noise scheduler、v target 和 future-only masked loss；
- [x] `M2-05` 实现小型 latent 3D U-Net denoiser；
- [x] `M2-06` 接入 Ego temporal tokens、cross-attention、AdaFiLM；
- [ ] `M2-07` 部分完成：已有 DDIM-like 采样和 CFG；需用 Diffusers scheduler 校验公式并补 DPM-Solver；
- [x] `M2-08` 接入 bf16、gradient checkpointing、EMA 和梯度累积；
- [x] `M2-09` 加入 NaN/Inf、梯度范数、吞吐、学习率和峰值显存监控；
- [x] `M2-10` 完成真实 VAE + 128×224、4→8 端到端训练；并验证完整 8→16 latent + 71.1M denoiser；
- [ ] `M2-11` 部分完成：本机22.48M模型已完成16 clips/2000-step过拟合并生成清晰道路；A800与单scene独立门禁尚未执行；
- [ ] `M2-12` 锁定配置后训练 10% 数据，确认无系统性首帧跳变；
- [ ] `M2-13` 完整数据训练，并自动生成固定验证视频。
- [ ] `M2-14` 部分完成：实现 cached-latent Dataset、分片缓存、单机 `torchrun` DDP 和 4/8×4090 配置；本机只有单 GPU，尚未真实双卡/NCCL 验收；

### M3：Ego 可控性（4～7 天）

- [x] `M3-01` 实现动力学连续的 straight/left/right/stop 编辑器；依赖 M2；
- [ ] `M3-02` 部分完成：推理 CLI 支持预设轨迹和固定 seed；尚未支持外部 JSON/CSV 自定义轨迹；
- [ ] `M3-03` 完成 no-Ego、history-only、future-Ego、shuffled-Ego 消融；
- [ ] `M3-04` 部分完成：condition sensitivity 已实现；optical-flow alignment 尚未实现；
- [ ] `M3-05` 接入车道线横移诊断；视觉里程计作为可选项；
- [ ] `M3-06` 对低/中/高曲率与不同速度场景分桶评估；
- [ ] `M3-07` 生成人工盲评清单，记录控制方向、结构稳定性和失败类型；
- [x] `M3-08` 输出 history/GT/original/straight/left/right/stop 动态网格和逐轨迹 GIF；
- [ ] `M3-09` 根据门禁决定进入 Map 条件，或回退修正数据/条件注入。

### 工程收尾

- [x] `R-01` README/TRAINING 文档覆盖安装、数据、缓存、训练、恢复、评估、推理和多卡启动；
- [ ] `R-02` 保存软件版本、GPU、配置、commit/hash（无 Git 时保存源码快照 hash）；
- [ ] `R-03` 部分完成：16 GB 本机配置已实测；多 4090 配置完成且单卡容量测试通过，待服务器真实 DDP 验收；
- [x] `R-04` 使用相对路径、local-only VAE 配置并忽略数据/权重/产物；
- [ ] `R-05` 汇总已知失败模式与后续 M4 接口预留。

## 8. 推荐执行顺序与并行项

关键路径：

```text
P0 → M0-01..09 → M0 门禁 → M1 门禁 → M2-01..11 → M2 门禁 → M3 门禁
```

可并行：

- M0 数据构建期间可同时实现可视化和单元测试；
- M1 训练运行时可开发 VAE adapter，但 M2 训练必须等 M0/M1 数据门禁；
- M2 完整训练期间可实现轨迹编辑器、指标和报告模板；
- 不建议让多个实现同时修改 Dataset schema、checkpoint schema 或模型主干。

## 9. 下一步执行顺序

### N1：先完成非训练门禁

1. 将 VAE 重建测试扩展到固定 20 个 val clips，统计 1/8/16 帧 PSNR、SSIM、LPIPS 和峰值显存；
2. 人工检查重建 GIF 的运动连续性、细小车辆/车道线和 history/future 边界；
3. [已完成] 使用真实 VAE latent 完成 128×224、4→8 diffusion 端到端 optimizer smoke test；
4. [已完成] 校验真实 latent 数值分布、noise scheduler 和 v-target 可逆性；
5. [已完成] 接入 gradient checkpointing、NaN/Inf/梯度/显存监控、固定验证和完整恢复参数；下一项是自动验证视频回调。

### N2：数据准备

- mini 保留用于 smoke test 和 tiny overfit；
- [部分完成] nuScenes `v1.0-trainval` metadata/CAN完整，CAM_FRONT约下载10%，已有85个完整scene；
- [已完成] 对partial trainval重新生成官方scene split manifest和train-only Ego mean/std；
- 若暂时没有 trainval，不进入 10%/完整训练，只做 16 clips 和单 scene 调试。

### N3：训练门禁（必须由用户明确启动）

1. M1 baseline：16 clips overfit → 单 scene overfit → no-Ego/Ego 对比；
2. M2 diffusion：16 clips overfit → 单 scene overfit，确认首帧连续、可采样、可恢复；
3. 只有上述门禁通过且 trainval 到位后，才运行 10% 数据；
4. 完整训练前锁定 A800 配置、日志系统和 checkpoint/验证频率。

`train.py` 继续保留 `--start-training` 显式开关；任何 smoke/data/VAE 测试都不会自动跨入训练。

## 10. V2 方案：单帧条件的清晰、稳定未来视频预测

### 10.1 问题定义与第一版结论

V2 的核心任务调整为：

```text
单目 CAM_FRONT
1 张 anchor image + future Ego trajectory
    → 未来 16 帧，6 Hz，256×448
```

同时保留 `8 history → 16 future` 作为增强模式和上限对照。训练和模型接口必须支持
`history_frames ∈ {1, 8}`，但 V2 的正式验收必须单独报告单帧输入结果，不能用 8 帧历史
结果代替。

当前 step 11200 的 22.48M 3D U-Net 已学到 stop/straight/left/right 条件响应，但生成
纹理随预测时距明显退化：样例中生成帧边缘强度从第 1 帧的 `7.23` 降至第 16 帧的
`5.45`，与 GT 的像素 MAE 从 `16.33` 增至 `32.85`。这说明控制通路已经生效，主要
瓶颈转为高噪声生成能力、远期历史约束、显式时间建模、数据规模和视频生成先验。

单帧未来本质上是多解问题：单张图像无法唯一确定前车速度、遮挡后物体和未来动态。
因此 V2 的“稳定”定义为场景结构、静态背景、物体身份和运动方向稳定，而不是要求每次
随机采样都逐像素复现唯一 GT。正式评估必须同时报告固定 seed 确定性和多 seed 多样性。

### 10.2 从 MagicDrive-V2 借鉴与不照搬的部分

借鉴以下已经在 `~/code/MagicDrive-V2` 中落地的设计：

- 冻结 CogVideoX VAE，训练主模型时只在 latent space 学习；
- 以预训练 Video DiT/STDiT 为生成主干，而不是从零训练小型 3D U-Net；
- spatial block、temporal block 和条件控制分支分离；
- Rectified Flow、logit-normal timestep sampling 和训练/推理一致的 timestep transform；
- image → short video → long video → high resolution 的渐进训练；
- 按分辨率、帧长、FPS 建 bucket，长视频使用 temporal chunk；
- bf16、gradient checkpointing、EMA、ZeRO/FSDP 和 sequence parallel 的云端训练路径。

V2 不直接照搬以下部分：

- 不做六相机，不需要 cross-view attention 和六视角 channel packing；
- 第一阶段不依赖 HD map、3D box、LiDAR/Radar 或 T5 文本；
- 本机 16 GB 不做 STDiT 全参数大规模训练，只做小配置、adapter/LoRA 和推理 smoke；
- 不直接复用 MagicDrive checkpoint，必须先验证 VAE latent scaling、patch shape、位置编码、
  scheduler 和条件接口完全兼容。

参考实现位置：MagicDrive 的在线冻结 VAE 在
`~/code/MagicDrive-V2/scripts/train_magicdrive.py`，渐进配置在
`configs/magicdrive/train/stage1_*`、`stage2_*`、`stage3_*`，Rectified Flow 在
`magicdrivedit/schedulers/rf/rectified_flow.py`。

### 10.3 V2 固定技术决策

| 项目 | V2 决策 |
|---|---|
| 输入 | 主模式：1 张 anchor；增强模式：8 帧 history；均带 future Ego + valid mask |
| 输出 | 16 future frames，6 Hz，256×448；后续再扩到 32 帧 |
| VAE | 继续冻结 CogVideoX VAE，16 channels、空间 `/8`、时间约 `/4` |
| 主干 | 预训练 single-view Video DiT/STDiT；保留当前 U-Net 作为消融基线 |
| token | latent patch `(1,2,2)`；显式 temporal position/RoPE + 2D spatial position |
| 条件 | anchor latent、逐 latent 对齐的 future Ego、FPS、相机内参；可选 anchor depth/semantic |
| 生成目标 | masked conditional Rectified Flow velocity；loss 仅覆盖 future latent |
| timestep | 默认 logit-normal；必须覆盖高噪声，移除当前 `mixed_low=0.5` 默认策略 |
| 推理 | 30～50 步 RF Euler/Heun；固定 history/anchor，每步重注入已知 latent；支持 CFG |
| 训练 | image → short video → target video 的 progressive curriculum + variable bucket |
| 本机 | 5070 Ti 16 GB：batch 1、checkpoint、LoRA/adapter、短视频 smoke |
| 云端 | 多 4090：DDP 起步；模型扩大后切 FSDP/ZeRO-2 + sequence parallel |

### 10.4 VAE 与 latent 协议改进

VAE 不是当前远期崩解的唯一根因，但它决定可达到的质量上限，V2 必须先锁定以下协议：

- anchor/history 与 future 分段编码，禁止训练 target 通过非因果 VAE 路径读取未来之外的信息；
- 训练、cache、验证和推理统一使用同一种 posterior 策略；第一阶段保持确定性 `mode`；
- 对照实验再评估 MagicDrive 使用的 posterior `sample`，不能混用 mode cache 和 sample online；
- 明确 `1 → 1 latent`、`8 → 3 latents`、`16 → 5 latents` 的 padding 和时间戳映射；
- future latent 解码必须与训练 target 的分段协议一致；另做“history+future 联合解码”消融，
  但不得引入训练/推理不一致；
- cache key 增加 VAE 权重 hash、diffusers 版本、posterior mode、padding protocol 和 dtype；
- 每个固定 val clip 保存 RGB GT、VAE oracle reconstruction 和生成结果三列对照；
- 按未来帧统计 VAE PSNR/SSIM/LPIPS/edge retention，确认 VAE oracle 不存在同样的远期坍塌。

TODO：

- [x] `V2-VAE-01` 实现单帧 anchor 和 1/8-history 的统一 latent/time index contract；
- [x] `V2-VAE-02` 为 cache 增加权重 hash、diffusers、posterior、padding 和 history 长度 fingerprint，旧 cache 不允许静默复用；
- [ ] `V2-VAE-03` 固定 100 个 val clips，输出逐帧 VAE oracle 指标及置信区间；
- [ ] `V2-VAE-04` 比较 separate decode 与 context decode，锁定无泄漏且边界最稳定的方案；
- [ ] `V2-VAE-05` 比较 posterior mode/sample；只有 val 质量稳定提升才允许切换 sample；

### 10.5 预训练 Video DiT/STDiT 主干

当前 U-Net 只有一次空间下采样、局部 Conv3D 和轻量条件 attention，缺少强视频先验。V2
新增 `SingleViewSTDiT`，结构要求：

```text
noisy future latent patches
      + anchor/history control branch
      + aligned Ego/FPS/camera tokens
                 ↓
spatial attention ↔ temporal attention
                 ↓
future velocity / flow prediction
```

- 采用 factorized spatial/temporal blocks，避免把所有时空 token 做全量 attention；
- 每个 latent time index 都有显式 temporal position 或 RoPE；支持不同 history/future 长度；
- anchor/history 使用独立 patch embedder/control branch，并以 zero-init residual 注入主干；
- 不只在输入 channel 拼接 history，深层 block 也能访问 anchor 的多尺度 spatial tokens；
- Ego token 先按真实时间戳重采样到 5 个 future latent time points，再做一对一或局部窗口
  cross-attention；禁止所有 latent 无约束地全局关注全部 Ego token；
- temporal block 同时建模静态背景保持、动态目标运动和遮挡出现；
- 从兼容的 image/video DiT 权重初始化 spatial/base blocks，新 temporal/control blocks 使用
  zero-init 或小方差初始化，避免一开始破坏预训练画质；
- 本机使用 LoRA/adapter 验证接口，云端再决定解冻 temporal blocks、control blocks 或全参数训练。

TODO：

- [x] `V2-MOD-01` 写预训练 checkpoint 参数名/shape/coverage 兼容性检查器；
- [x] `V2-MOD-02` 实现显式 temporal/spatial sin-cos position，并补 Ego 帧顺序响应测试；
- [x] `V2-MOD-03` 实现 anchor/history control branch 和逐层 zero-init residual；
- [x] `V2-MOD-04` 实现 Ego frame rate → latent rate 的 valid-aware 确定性对齐器；
- [ ] `V2-MOD-05` 部分完成：aligned Ego 已通过逐时间 AdaLN 注入；待实现 local cross-attention 并与 global attention 消融；
- [x] `V2-MOD-06` 支持 `history_frames={1,8}` 和 variable future latent length；
- [ ] `V2-MOD-07` 部分完成：已有按参数名/shape的部分加载与审计；待取得兼容预训练权重并验证 spatial/base 映射；
- [ ] `V2-MOD-08` 部分完成：已有16.06M本机STDiT配置和真实数据显存报告；待补U-Net/预训练STDiT同条件对照；

### 10.6 Rectified Flow 与清晰度目标

当前 `mixed_low` 过多采样低噪声，而推理从高噪声开始，容易出现 loss 下降但纯噪声生成
能力不足。V2 默认切换为与 MagicDrive 类似的 Rectified Flow：

- 使用 `x_t = (1-t) * noise + t * clean` 的统一约定，并将方向、训练 target、采样方向写入
  单元测试，避免 timestep 正反定义混乱；
- timestep 使用 logit-normal，保证中高噪声覆盖；根据分辨率/帧长启用经过验证的 timestep
  transform，训练和推理必须共享配置；
- history/anchor 始终保持 clean，future 才参与加噪和 flow loss；
- loss 输出 overall、逐 future latent、逐 timestep bucket 三套统计；
- 使用 min-SNR 或显式 bucket weighting 平衡极高/极低噪声，但不得再次让低噪声占一半以上；
- 可选低频率 quality fine-tune：从预测 flow 还原 `x0`，在随机小批次上加入 latent Charbonnier、
  temporal-gradient loss；VAE-decode LPIPS/edge loss 仅作为后期实验，先评估显存和稳定性；
- 禁止把 LPIPS 直接作为唯一生成目标，避免提高单帧锐度却破坏时序和多样性。

TODO：

- [x] `V2-RF-01` 实现 Rectified Flow scheduler、future-only masked loss、Euler 和 Heun sampler；
- [x] `V2-RF-02` 补 `t=0/1`、velocity 可逆、history mask、单帧前向/采样一致性测试；
- [ ] `V2-RF-03` 对比 uniform、logit-normal、当前 mixed-low 的 timestep-bucket val loss；
- [x] `V2-RF-04` 增加逐 future latent loss 与 TensorBoard 记录；待训练后确认最后一个 latent 趋势；
- [ ] `V2-RF-05` 部分完成：20/30/50步及Euler/Heun接口已支持；待训练checkpoint质量/耗时比较；
- [ ] `V2-RF-06` 在门禁通过后实验 latent x0/temporal-gradient 辅助 loss；

### 10.7 单帧条件与几何/运动先验

单帧输入无法直接观测速度和遮挡变化。V2 最小条件仍是 future Ego，但为提高稳定性，允许加入
只从推理时可获得信息计算的先验：

- 相机内参和固定外参，用于区分几何透视与普通图像运动；
- anchor 单目 depth，作为可选 frozen teacher feature，不要求训练时 GT depth；
- anchor semantic/instance feature，帮助保持道路、车辆和树木身份；
- 8-history 模式可使用历史 optical flow/track feature；单帧模式必须关闭，避免接口泄漏；
- future Ego 保持 9D 连续量和 valid mask，并增加相对时间 `Δt`；
- stop/left/right 编辑必须受训练分布约束；默认 60° 转向需要标记为 OOD，不作为主要质量结论；
- classifier-free dropout 分别作用于 Ego、geometry 和 anchor control，支持独立 guidance scale。

TODO：

- [ ] `V2-COND-01` 将相机内参与 `Δt/FPS` 加入条件 contract；
- [ ] `V2-COND-02` 统计真实轨迹曲率/速度范围，为反事实轨迹增加 OOD score；
- [ ] `V2-COND-03` 接入可关闭的 frozen monocular depth feature，并做有/无消融；
- [ ] `V2-COND-04` 接入可关闭的 semantic/instance feature，并检查动态物体身份保持；
- [ ] `V2-COND-05` 实现按条件类型独立 dropout/CFG，防止 guidance 放大纹理噪声；

### 10.8 渐进训练与数据策略

参考 MagicDrive 的 stage1/2/3，V2 不允许从随机初始化直接在小数据上训练完整 1→16：

| Stage | 任务 | 分辨率/长度 | 主要目标 | 解冻范围 |
|---|---|---|---|---|
| S0 | VAE oracle + 16 clips overfit | 256×448，1/8→16 | 协议和边界正确 | 不训练 VAE |
| S1 | image/anchor reconstruction | 224×400，T=1 | 保留预训练空间画质 | control/embedder |
| S2 | short I2V | 128×224，1→4/8 | 学运动和 anchor 保持 | temporal+control |
| S3 | target I2V | 256×448，1→16 | 清晰远期预测 | temporal+control+部分 base |
| S4 | mixed history | 256×448，{1,8}→16 | 单帧与多帧统一 | 按门禁决定全参 |
| S5 | cloud quality FT | 多分辨率/长度 bucket | 泛化和长时稳定 | 多 4090/FSDP |

数据要求：

- partial 62 train scenes 只用于工程和消融，不能作为 V2 最终质量训练集；
- 至少使用完整 nuScenes train split，并按 scene 去重统计有效视频时长，而不只统计重叠 clip 数；
- 对 stop/turn/high-curvature、动态目标、夜间、雨天做 bucket balance，禁止只靠窗口重复提高样本数；
- 所有 crop/color augmentation 必须整段共享；几何翻转必须同步修改 yaw/steering/轨迹；
- 混合 `history=1/8` 时使用显式 bucket sampler，确保单帧任务不会被更容易的 8-history 淹没；
- 可引入其他驾驶视频做无标签视频预训练，但 nuScenes fine-tune 和 val scene 必须严格隔离。

TODO：

- [ ] `V2-DATA-01` 下载并验收完整 nuScenes CAM_FRONT trainval；
- [ ] `V2-DATA-02` 报告独立 scene、有效分钟数、动态/天气/轨迹 bucket，而非只报 clips；
- [ ] `V2-DATA-03` 实现 1/8-history、4/8/16-future、分辨率/FPS variable bucket sampler；
- [ ] `V2-DATA-04` 实现 stop/turn/dynamic/night/rain 的可重复平衡采样；
- [ ] `V2-TRN-01` 完成 S0→S4，每一级通过门禁后才进入下一级；
- [ ] `V2-TRN-02` 保存每个 stage 的初始化来源、解冻参数列表和 optimizer reset 策略；
- [ ] `V2-TRN-03` 比较从零、image pretrained、video pretrained 三种初始化；

### 10.9 训练稳定性与多卡方案

- EMA 不能从第一步固定使用 `0.9999`；采用早期 `0.99/0.999`、后期逐步升高，或在 warmup
  后初始化 EMA，并同时保存 raw/EMA 固定验证结果；
- bf16、gradient clipping、NaN/Inf、梯度范数、逐 timestep loss、逐 horizon loss 必须记录；
- 本机继续使用 CPU affinity guard、短 segment checkpoint 和原子保存，但硬件稳定性问题应从
  BIOS 默认设置、CPU/RAM 超频和内存测试上根治；
- 本机只做 adapter/LoRA 和最多 100～500 step smoke，不以本机吞吐决定云端模型结构；
- 多 4090 先用 DDP；单卡放不下 optimizer/model 时切 FSDP 或 ZeRO-2；token 数成为瓶颈时再启用
  spatial sequence parallel；
- VAE 编码可预缓存；online/cache 必须通过同 clip latent `allclose` 或统计等价检查；
- checkpoint 保存模型、EMA、optimizer、scheduler、sampler bucket、RNG、stage 和数据 fingerprint。

TODO：

- [x] `V2-SYS-01` 实现 EMA warmup decay 和向后兼容 checkpoint state；待训练后输出 raw/EMA 对比；
- [ ] `V2-SYS-02` checkpoint 增加 stage、bucket sampler 和数据/VAE fingerprint；
- [ ] `V2-SYS-03` 部分完成：单张5070 Ti真实VAE+STDiT forward/2-step Heun sample通过，峰值9.483GB；未启动backward/optimizer；
- [ ] `V2-SYS-04` 2×/4×4090 完成 DDP 吞吐、恢复和数值一致性测试；
- [ ] `V2-SYS-05` 模型扩大后完成 FSDP/ZeRO-2 与 sequence parallel 验收；

### 10.10 清晰度、稳定性和控制性门禁

V2 禁止只依据总 diffusion loss 判断质量。固定 val 集按预测时距输出：

- 图像：PSNR、SSIM、LPIPS、DISTS、edge retention/high-frequency energy；
- 时序：warping error、temporal LPIPS、flow consistency、flicker、anchor→future boundary error；
- 结构：道路/车道区域稳定、静态背景漂移、车辆 identity/size consistency；
- 控制：left/right motion sign、stop flow reduction、轨迹条件 sensitivity 和 shuffled-Ego 消融；
- 生成：FVD 或等价 video feature distance；固定 seed、raw/EMA、不同采样步数；
- 多解：每个条件生成 K=4 seeds，报告 mean、best-of-K、diversity，防止模型退化成模糊均值；
- 指标按 `0～1 s / 1～2 s / 2～2.67 s` 和逐帧两种方式报告。

阶段门禁：

- G0 VAE：16 帧 oracle PSNR 中位数不低于 25 dB，且末帧 edge retention 不比首帧下降超过 10%；
- G1 过拟合：16 clips 上单帧输入可保持道路/车辆结构，末帧生成 edge 不低于首帧的 85%；
- G2 高噪声：最高噪声 bucket 的 val loss 持续下降，纯噪声起步采样不出现统一纹理崩解；
- G3 泛化：独立 val scene 的末帧/首帧 edge ratio ≥ 0.85，且远期 LPIPS 不再单调失控；
- G4 控制：相同 anchor/seed 下方向指标正确，shuffled Ego 明显降低控制对齐；
- G5 多样性：K=4 结果保持结构稳定且存在合理动态差异，不允许四个 seed 完全相同或全面崩坏。

TODO：

- [ ] `V2-EVAL-01` 部分完成：实现逐帧 edge/PSNR/MAE/edge-retention 并写入反事实metadata；待接LPIPS和分时距聚合；
- [ ] `V2-EVAL-02` 实现 optical-flow warping、flicker 和静态背景漂移指标；
- [ ] `V2-EVAL-03` 实现 train/val、raw/EMA、U-Net/STDiT 的固定网格对照；
- [ ] `V2-EVAL-04` 实现 K-seed mean/best/diversity 报告；
- [ ] `V2-EVAL-05` 将 G0～G5 做成可失败的自动 gate，不通过时禁止扩大训练；

### 10.11 V2 推荐实施顺序

```text
V2-VAE-01..04 + V2-EVAL-01
        ↓
当前 U-Net 上完成 timestep/temporal-position 快速消融
        ↓
V2-RF-01..05
        ↓
V2-MOD-01..08（预训练 single-view STDiT）
        ↓
S0 16 clips → S1 image → S2 short I2V
        ↓
完整 nuScenes + S3 1→16 → S4 {1,8}→16
        ↓
多 4090 S5 quality fine-tune → G0..G5 最终验收
```

在进入 STDiT 云端训练前，先用当前 U-Net 做三个低成本判因实验：

1. train clip 与 val clip、raw 与 EMA 的逐帧对比；
2. uniform/logit-normal 与 mixed-low 的高噪声 bucket 对比；
3. 显式 temporal position + aligned Ego 与当前 global Ego attention 对比。

这三个实验用于确认问题来源和建立 V2 baseline，不将当前 22.48M U-Net 继续训练更多步视为
V2 的主要质量方案。所有 V2 训练仍必须由用户显式传入 `--start-training`，文档和测试变更
不得自行启动训练。
