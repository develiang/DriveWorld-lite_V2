# DriveWorld-lite V2 第一版实现方案与 TODO

> 本计划将 `roadmap.md` 中的 M0～M3 落成一版可执行工程方案。第一版只做：
> **nuScenes / CAM_FRONT / 8 帧历史 → 16 帧未来 / 6 Hz / Ego 轨迹条件 / 256×448**。

## 0. 当前状态（2026-07-13）

状态约定：`[x]` 已实现并通过相应非训练测试；`[ ]` 尚未完成；“部分完成”表示已有代码，但还缺验收项。

- 工程代码：P0 基础骨架、M0 数据管线、M1/M2 模型骨架、M3 轨迹编辑与推理入口已落地；
- mini 数据：10 scenes、196.55 秒、2342 个 CAM_FRONT 帧，生成 train 574 / val 192 个高度重叠 clips；
- 数据验证：机械抽检 train/val 各 100 clips、共 4800 张图像，错误 0、scene 泄漏 0、时间误差 p95=33.333 ms；
- 单元测试：63 passed；覆盖真实 Dataset contract、DDPM/RF 可逆性、future-only mask、
  单帧 STDiT、MDD 17→5 VAE/RF/CFG、权重严格加载、adapter-only checkpoint/resume、
  Ego 顺序响应、EMA 兼容和逐帧质量指标；
- VAE：本地 `pretrained/vae` 的 CogVideoX VAE（16 latent channels、空间压缩 8、时间压缩 4）加载成功；
- VAE shape：1×256×448 → 1×16×32×56；8 帧 → 3 latent 帧；16 帧 → 5 latent 帧；padding/crop 重建测试通过；
- 真实图像重建：单帧 PSNR 36.81 dB、8 帧 28.27 dB、16 帧 26.79 dB；样例结构保持良好，仍需多 clip 统计；
- 训练验证：真实 RGB+VAE 2-step、cached latent 50-step、checkpoint resume、完整 8→16 latent + 71.1M denoiser 单步均通过；
- 本机收尾：22.48M 模型完成16-clips 2000-step质量过拟合；完整mini完成10000-step计算，推荐val checkpoint@2500；
- 本机生成：训练clip道路结构可连续生成；Ego条件产生非零响应但控制幅度较弱；完整mini对独立val scene泛化仍不足；
- 尚未进行：16 clips/单 scene 正式 overfit、trainval 完整训练、真实多 GPU NCCL 验收、学习后控制性评估；
- 数据结论：mini 足够做工程/数据/VAE smoke test，不足以正式训练或得出泛化结论；正式阶段需要 `v1.0-trainval`。
- Partial trainval：已准备85个完整scene（train 62 / val 23）；8→16为4381/1624 clips，4→8为5586/2071 clips，随机检查与scene隔离通过；
- V2-Lite 调试基线已落地：single-image anchor、自研 `SingleViewSTDiT`、显式时空位置、逐 latent Ego 对齐、masked Rectified Flow、Euler/Heun、V2 cache fingerprint 和 EMA warmup；真实数据无训练 smoke 输出 `[1,16,3,256,448]`、峰值 9.483 GB。该 16M 级模型只保留作接口/数据/RF 回归基线，正式 V2 主线改为适配 MagicDrive Stage-3 EMA 的 `V2-MDDiT`；
- V2-MDDiT 权重适配已扩展到完整单视角 control 主干：Stage-3 `ema.pt`（SHA256
  `0806b23...83334`）直接从 mmap FP32 materialize 为 BF16，主干/control 1489 keys、
  1,806,221,732 参数严格加载；camera/frame/null-bbox 条件 50 keys、38,513,077 参数严格
  加载，只有 4 个 kinematics adapter 张量为新增参数。修复了 control temporal skip 重复相加
  和 PyTorch meta `assign=True` 间歇段错误；
- V2-MDDiT 本机 S2 adapter 训练和用户侧 100-step/推理已跑通。正式主线现为 B2 静态图
  control + temporal/cross-attention LoRA + AdaLN + action adapter，共 12,441,472 个可训练参数；
  256×448 真实 backward loss=0.999229，allocated/reserved 峰值 7.798/8.823 GiB，梯度有限；
  30-step B1/B2 推理均有限且 B2 对真实地图有非零响应，但零样本画质仍差，必须经过训练和
  固定验证集评测后才能判断清晰度收益；
- 静态地图数据链已落地：manifest schema v3 包含 location/map pose，nuScenes expansion JSON
  生成 MagicDrive 同顺序 8×200×200 BEV；partial train/val 的 bit-packed mmap 缓存为
  4381/1624 条，跨分段抽查逐像素一致。缓存生成按隔离 worker 可恢复原生崩溃，训练阶段
  只读 mmap，不再在线调用 Shapely/GEOS；
- 4×5090 配置已提供 12 Hz（50k steps）→6 Hz（100k steps）两阶段 DDP。`--resume` 只用于
  同阶段恢复；新增 `--init-checkpoint` 只迁移 LoRA/AdaLN/action delta 并重置 optimizer、
  scheduler、step 和 RNG。单 GPU 真实 dry-run 与 `torchrun --nproc_per_node=1` 均通过，
  未自动启动训练；
- 当前工作站 RGB 只有 metadata 预期帧的约 9.8%，full YAML 的 99% 图像完整率 gate 会主动
  拒绝生成伪“全量” manifest。全量 manifest、4×GPU NCCL/DDP 和正式训练必须到用户的
  4×5090 全量数据服务器上验收；

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
- M0～M3 第一版不改 MagicDrive-V2 主干；正式 V2-MDDiT 适配单列于第 10 节；
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

## 10. V2-MDDiT：基于 MagicDrive Stage-3 的单视角世界模型适配计划

> 本节于 2026-07-13 重写。设计依据是本机
> `/home/liang/code/MagicDrive-V2`（commit
> `4ed72c60e5e73e4fa6072a7321fcc2ed9668edee`）的实际源码和 Stage-3 配置，
> 不是按论文名称或模型外观猜测。用户给出的 `~/code/MagicDrive` 实际对应本机
> `~/code/MagicDrive-V2`。
>
> 约束：本节只定义实现、测试和训练方案；任何正式训练都必须由用户显式启动，代码准备、
> checkpoint 转换、单元测试和无反向 smoke test 不得自动进入训练。

### 10.1 路线重定义

V2 的正式质量主线改为 **V2-MDDiT**：保留 MagicDrive Stage-3
`MagicDriveSTDiT3-XL/2` 的 CogVideoX latent 接口、空间/时间 block、AdaLN、
cross-attention、RF scheduler 与 EMA 权重，删除多相机拓扑依赖，并把条件改造成
“单张 CAM_FRONT + 已知未来 Ego 动作/轨迹 → 16 帧未来”。

当前已经实现的 `driveworld/models/single_view_stdit.py` 不删除，但重命名为概念上的
**V2-Lite**。它只负责数据 contract、mask、RF、Ego 对齐、checkpoint resume 和小显存
回归，不再作为预期获得 MagicDrive 画质的主模型。原因不是参数量本身，而是它的维度、
block 实现和参数命名均不能高覆盖加载 Stage-3 EMA。

正式目标固定为：

- RGB 序列：1 张 anchor + 16 张未来，共 17 帧；
- 第一阶段复现 checkpoint 分布：224×400、12 Hz、17 帧；
- 项目目标：256×448、6 Hz、17 帧，其中只输出/评分后 16 帧；
- 视角数：`NC=1`，只使用 `CAM_FRONT`；
- VAE：CogVideoX-2b VAE，16 latent channels，空间压缩 8、时间压缩 4；
- 生成主干：`hidden_size=1152`、`depth=28`、`num_heads=16`、
  `patch_size=(1,2,2)`、`pred_sigma=False`；
- 训练目标：与 MagicDrive 相同的 rectified-flow velocity
  `x_start - noise`，loss 只覆盖待生成 latent；
- 本机只做转换、推理、forward/backward smoke 和 adapter/LoRA 调通；
  4×5090 使用冻结 BF16 主干的 DDP 做 LoRA/AdaLN/action-adapter 正式训练。只有后续扩大
  解冻范围、单卡不再容纳完整主干时，才切 FSDP/ZeRO，而不是在当前 1244 万可训练参数
  阶段先引入额外分片复杂度。

### 10.2 已核对的 MagicDrive 源码契约

| 源码位置 | 实际实现 | V2 结论 |
|---|---|---|
| `configs/magicdrive/train/stage3_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py` | Stage-3 使用 16-channel CogVideoX latent、XL/2、bf16、SP=4、12 Hz、logit-normal RF、timestep transform、30-step val/CFG=2 | 新配置先逐项复现，不能直接沿用当前小 DiT YAML |
| `magicdrivedit/models/magicdrive/magicdrive_stdit3.py::MagicDriveSTDiT3` | `PatchEmbed3D`、28 组 spatial block、28 组 temporal block、13 层 control branch、frame/camera/box/map embedder、`T2IFinalLayer` | 复用同构类或抽取同构最小实现，不用 PyTorch MHA 近似替代 |
| `magicdrive_stdit3.py::MultiViewSTDiT3Block` | spatial block 默认执行 cross-view attention；temporal block自动跳过；空邻接表会在拼接邻居 token 时失败 | `NC=1` 时构造 spatial block 必须显式 `skip_cross_view=True`，不能传 `{0: []}` |
| `magicdrive_stdit3.py::forward` | 当前代码通过 `len(mv_order_map)` 推断 `NC`，并强制调用 `encode_map` | 单视角 wrapper 必须显式固定 `NC=1`，并让 map/control branch 可选 |
| `magicdrive_stdit3.py::encode_cond_sequence` | 每个时间位置组合 frame relative pose、首帧 camera、T5 文本和可选 bbox token | 保留 per-time cross-attention 契约；替换/裁剪条件，不重写整套 block |
| `magicdrivedit/datasets/nuscenes_t_dataset.py::obtain_next2top` | 从 lidar/ego/global 标定计算 4×4 `next2top`，`frame_emb="next2top"` 送入 frame embedder | 项目必须按同方向生成 4×4 pose，并以数值测试对齐，不能只把 `x,y,yaw` 随意塞入 |
| `magicdrivedit/models/vae/vae_cogvideox.py::VideoAutoencoderKLCogVideoX` | posterior 用 `sample() * scaling_factor`；`micro_frame_size=8`、`micro_batch_size=1`；encode 后清 causal cache | VAE wrapper、scaling、随机性和清 cache 行为均需一致 |
| `magicdrivedit/utils/train_utils.py::MaskGenerator` | `image_head` 可将第一个 latent 标为 False，但源码在 `num_frames//4<=1` 时直接返回全 True，因此 17 RGB→5 latent bucket 实际不会触发 image-head；33 帧等长 bucket 才提供该训练信号 | V2 的 5-latent I2V mask 是对原 mask 机制的显式适配，不冒充 Stage-3 的 17 帧原始分布；不得给 latent 拼第 17 个 mask channel |
| `magicdrivedit/schedulers/rf/rectified_flow.py` | `t=0` 为干净端、`t≈1000` 为噪声端；训练 target 为 `x_start-noise` | 当前 V2-Lite 的归一化时间定义不可原样送入 Stage-3 |
| `magicdrivedit/schedulers/rf/__init__.py::RFLOW.sample` | 从 1000 向 0 Euler 更新，支持 clean-frame mask 和 CFG | MDDiT 路线优先移植/封装该 sampler，而不是复用语义不同的现有 sampler |
| `scripts/train_magicdrive.py` | VAE `eval()` 且 `no_grad()` 在线编码；RGB 从 `[B,T,NC,C,H,W]` 变为 `[B*NC,C,T,H,W]`；模型与 EMA 单独保存 | 训练时 VAE 冻结但可在线编码；VAE 不进入 optimizer/checkpoint |
| `magicdrivedit/utils/ckpt_utils.py` | `ema.pt` 是直接 `torch.save(ema.state_dict())` 的 FP32 state dict | 下载完成后先离线审计/转换一次，训练时不反复读取约 8 GB FP32 文件 |

### 10.3 预训练权重与可加载边界

`pretrained/MDDiT/ema.pt` 已完整落盘（8,152,555,582 bytes），并已完成只读结构审计、
SHA256 和单视角 base-only 映射报告；来源 revision/许可证元数据仍需补齐。

- [x] `V2-CKPT-01` `ema.pt` 已完整落盘，记录 8,152,555,582 bytes 与 SHA256；来源
  MagicDrive-V2 commit 已固化。gated model 页面来源和发布许可证仍列入 `V2-SYS-05`；
- [x] `V2-CKPT-02` 用 `torch.load(..., mmap=True, weights_only=True)` 只读审计 key、
  shape、dtype；禁止在 4070Ti 上先构造两份 FP32 全模型；
- [x] `V2-CKPT-03` 输出 `artifacts/reports/mdd_stage3_checkpoint.json`，包含
  matched / dropped / missing / shape-mismatch 的参数个数、元素数和占比；
- [x] `V2-CKPT-04` 已实现从 mmap FP32 EMA 直接把保留参数 materialize 到目标
  device/BF16，真实峰值 2.403 GB；独立 BF16 `safetensors` 降为可选部署优化，不再是前置步骤；
- [x] `V2-CKPT-05` 正式路径不再生成第二份转换权重；加载器按明确白名单从 mmap checkpoint
  直接 materialize 目标 BF16 参数，单视角删除项、matched/missing/shape mismatch 均进入
  report，避免 8 GB FP32 + 完整 BF16 文件同时常驻；
- [x] `V2-CKPT-06` 已保存带 upstream commit/config SHA 的 Stage-3 source snapshot；
  本项目 resolved config 会写入 load report。加载前校验 snapshot/report SHA、checkpoint
  size/SHA 记录、`in_channels=16`、1152/28/16、patch 1×2×2 和 final 64×1152
  （即 `pred_sigma=False`）；
- [ ] `V2-CKPT-07` 在 MagicDrive 原仓库用原 config 跑一个官方 checkpoint 的无训练
  推理/forward 基线，保存 shape、显存和固定 seed 输出摘要，作为适配前参照。

权重分类必须明确：

| 权重组 | 处理 |
|---|---|
| `x_embedder`、`t_embedder`、`t_block`、`fps_embedder` | 原 shape 严格加载 |
| `base_blocks_s`、`base_blocks_t`、`final_layer` | 原 shape 严格加载；这是视频生成能力主干 |
| spatial block 的 `cross_view_attn/norm3/mva_proj/scale_shift_table_mva` | 单视角不实例化或转换时丢弃，并单独统计 |
| `y_embedder` | 初期保留冻结，用预计算空/固定文本 token；不在线加载 T5-XXL |
| `camera_embedder`、`frame_embedder`、`base_token` | 保留并严格加载；Ego pose 优先复用 frame embedder |
| `bbox_embedder` | 禁止输入未来真值 box；保留并严格加载其 learned null-box 时序 token 路径，使 cross-attention 序列与 Stage-3 一致 |
| `controlnet_cond_embedder*`、`control_blocks_*`、`before_proj`、`x_control_embedder` | B2 静态 map 正式启用并严格加载；已修复 temporal skip 重复注入，不允许“加载了但 forward 没走” |
| 新增 kinematics/action adapter | 新参数，零初始化输出投影，单独报告和训练 |

验收标准：主干组必须 **100% shape match**；总体覆盖率同时报告“相对完整 Stage-3”
和“相对精简单视角目标模型”两个分母，不能用删除大量 key 后的虚高百分比。

### 10.4 单视角 MDDiT 的代码适配

新增模型建议命名为 `MagicDriveSingleViewSTDiT`，其实现以 MagicDrive 的类为来源，
但 forward contract 改成项目原生 tensor，不把整个 mmdet/ColossalAI 数据栈引进来。

- [x] `V2-MOD-01` 固化与 Stage-3 同构的 `MagicDriveSTDiT3Config` 子集：
  16→16 channels、patch 1×2×2、1152、28、16、MLP ratio 4、qk norm；
- [x] `V2-MOD-02` 已移植 `PositionEmbedding2D`、RoPE、`PatchEmbed3D`、
  `MultiViewSTDiT3Block`、`T2IFinalLayer` 的 Stage-3 同构子集，并以 PyTorch SDPA 提供
  fused-kernel fallback；
- [x] `V2-MOD-03` 单视角 base/control spatial block 都不实例化 cross-view 分支，
  `skip_cross_view=True` 固化在 NC=1 构造路径；
  禁止用空 `mv_order_map` 伪装单视角；
- [x] `V2-MOD-04` forward 内固定 `NC=1`，输入保持 `[B,16,Tz,H/8,W/8]`，
  移除 view/channel pack-unpack 和 `len(mv_order_map)` 依赖；
- [x] `V2-MOD-05` 已保持动态 2D position、temporal RoPE、spatial→temporal block 顺序、
  AdaLN、cross-attention 和 final layer 数值路径；
- [x] `V2-MOD-06` 已实现 base-only/zero-map 与
  `encode_map → x_control_embedder → control blocks → skip` 的静态 map 路径；完整 control
  1489 keys 严格加载，temporal skip 只注入一次；
- [ ] `V2-MOD-07` 已保存 B1 zero-map 和 B2 static-map 的 256×448、30-step 输出及显存，
  B2/B1 平均像素差为 10.398/255；还缺训练后的同 seed 定量画质对照与 history-control，
  因此不把零样本“有响应”误写成质量验收完成；
- [x] `V2-MOD-08` 支持 gradient checkpoint、BF16、PyTorch SDPA 自动选择
  Flash/memory-efficient/math backend；该修改已消除 256×448 显式 attention backward 的
  native segfault；
  单卡不依赖已初始化的 sequence-parallel process group；
- [x] `V2-MOD-09` 加载器拒绝 silent shape mismatch，尤其拒绝把当前 256/512 hidden
  的 V2-Lite checkpoint 当成 Stage-3；
- [ ] `V2-MOD-10` NC=1 小模型、真实 28+28+13 control 层 BF16 forward/backward、
  256×448 完整 30-step 已通过；
  固定 seed 数值 golden、真实 224×400 latent 和 chunk 前后一致性仍待完成。

control branch 不预先拍脑袋删除，按下列源码对应关系做消融：

1. **B0 base-only**：只执行 28 组 base spatial/temporal block，验证主干生成先验；
2. **B1 history-control**：保留 `x_control_embedder`、13 层 control block 和 skip，
   以 masked 17-frame latent 作为 control 输入，检查 anchor 保真是否改善；
3. **B2 map-control**：只有在项目能从 nuScenes 样本稳定构造与训练/推理同源的静态 BEV
   map 时启用原 map encoder；动态未来 box 不作为输入。

B0/B1/B2 必须用相同 checkpoint、seed、采样步数和验证 clips 报告，之后才能决定正式主线。

### 10.5 17 帧 / 5 latent 的 I2V 协议

MagicDrive 的 CogVideoX wrapper 对 17 RGB 帧整体编码后得到 5 个 latent 时间位置。
因此正式 V2 不再采用“anchor 单独编码成 1 latent + future 16 帧单独编码成 5 latent =
6 latents”的旧协议；该形状偏离 Stage-3 的 17-frame 训练分布。

目标协议为：

```text
RGB [anchor, future_1 ... future_16]  --joint VAE encode--> z [B,16,5,H/8,W/8]
latent mask = [False, True, True, True, True]
False: t=0 clean / fixed / no loss
True : RF noisy / generated / loss
joint VAE decode(5 latents) -> 17 RGB frames -> 丢弃 anchor，只输出 future 1..16
```

- [x] `V2-VAE-01` 以 MagicDrive wrapper 的 `latent_dist.sample()*scaling_factor`、
  `micro_frame_size=8`、`micro_batch_size=1` 为唯一正式协议；
- [x] `V2-VAE-02` encode 后按 MagicDrive wrapper 调用
  `_clear_fake_context_parallel_cache()`；decode 不额外假定源码没有的清理接口，并用本地
  diffusers 版本的独立测试决定是否需要进程隔离，避免之前 online VAE 的 native segfault；
- [ ] `V2-VAE-03` 实测 1/9/17 帧的 latent shape，并将 17→5 写成测试；
- [ ] `V2-VAE-04` 先用 posterior mean 检验
  `encode(anchor)[:,:,0]` 与 `encode(anchor+future)[:,:,0]` 的 causal 一致性，再对
  posterior sample 做多 seed 统计；报告 max/mean error；
- [ ] `V2-VAE-05` 做 VAE oracle：joint encode/decode 真实 17 帧，逐帧统计
  PSNR/SSIM/LPIPS/edge，生成模型不得被要求超过 VAE 上限；
- [ ] `V2-VAE-06` 训练时可在线 VAE 或使用 **17-frame joint cache**；
  旧的 1+16 分段 cache 必须因 fingerprint 不兼容而被拒绝；
- [x] `V2-VAE-07` 在线 VAE 始终 `eval()+requires_grad_(False)+no_grad`，
  不进入 optimizer、EMA 或 denoiser checkpoint；
- [x] `V2-VAE-08` 推理 clean anchor latent 必须来自真实 anchor；每个 Euler step 后
  恢复 clean 区域，不能让第一 latent 漂移。

如果 `V2-VAE-04` 不满足数值一致，不能悄悄改回 6 latents；需保留 joint 17-frame
协议，并在文档中说明训练时 clean latent 含有 causal joint encode 的差异，另外做
anchor-only 初始化对照实验。

### 10.6 Mask、RF 与 timestep 必须完全对齐

MagicDrive 的训练/采样语义固定如下：

- `t=0`：`add_noise` 返回 clean latent；
- `t=1000`：接近纯 noise；
- velocity target：`x_start - noise`；
- 采样：`1000 → 0`，更新 `z = z + v_pred * dt`；
- clean history：`x_mask=False`，block 和 final layer 使用 `t0` modulation；
- future：`x_mask=True`，参与加噪和 loss；
- Stage-3：logit-normal timestep sampling，`use_timestep_transform=True`，
  `cog_style_trans=True`；
- CogVideoX 帧数变换：17 RGB 帧按源码映射为 5 latent 帧，再参与分辨率/时长 transform。

- [x] `V2-RF-01` 新增 `MagicRectifiedFlowScheduler`，逐项适配 MagicDrive
  `RFlowScheduler/RFLOW`，不与现有 RF 用同名而语义不同的 `t`；
- [x] `V2-RF-02` 单元测试 clean/noise 两端、target 和 1000→0 timestep 顺序；
  完整真实模型 Euler loop 已通过 1-step CFG smoke；
- [x] `V2-RF-03` 对 `mask=[0,1,1,1,1]` 测试 clean latent 不变、future 才计 loss；
- [x] `V2-RF-04` 训练和推理共用同一 `timestep_transform`，传入真实
  `height/width/num_frames=17`，禁止以 latent T=5 冒充 RGB num_frames；
- [x] `V2-RF-05` scheduler 默认复现 Stage-3 的 logit-normal；uniform 只能作为显式 ablation；
- [x] `V2-RF-06` 默认配置和推理入口已固定 30-step Euler、CFG=2，并用 unit loop 与
  真实模型 256×448 完整 30-step CFG 验证更新/解码全有限；Heun、50 steps 和其他 CFG 作为后续
  评测变量，不改变 checkpoint 兼容基线；
- [x] `V2-RF-07` CFG 的 camera/rel_pos 使用 MagicDrive embedder 的 learned null，
  kinematics 使用 valid-mask null；当前文本固定为同一个空/base token，因此 cond/uncond
  不引入不同文本，不用全零 token 冒充 learned null；
- [x] `V2-RF-08` V2-Lite 与 V2-MDDiT scheduler/config 分开命名，checkpoint 中保存
  scheduler family 和 timestep direction，resume 时严格校验。

### 10.7 条件适配与未来信息泄漏

MagicDrive 是条件视频生成器，训练时使用 map、camera、relative pose 和未来帧 box。
世界模型推理时拿不到未来真值 box，因此不能原样把所有 Stage-3 条件喂进去。

条件按可用性分组：

| 条件 | V2 输入策略 | 原因 |
|---|---|---|
| anchor RGB | clean 首 latent | 推理时真实可用 |
| CAM_FRONT 内外参 | 保留首帧 `camera_embedder` | 推理时已知，且有 Stage-3 权重 |
| future relative ego pose | 用项目轨迹构造与 `obtain_next2top(v2=True)` 同方向的 4×4 矩阵，送 `frame_embedder` | 直接复用 Stage-3 的 per-time pose token |
| vx/vy/ax/ay/yaw_rate/steering | 新增 kinematics adapter，输出 per-time token并拼入原 cross-attention sequence | 原 frame embedder 只编码 4×4 pose，不能覆盖这些量 |
| 文本 | 使用 Stage-3 checkpoint 中已保存的空文本 `base_token`；`y_embedder` 不实例化 | 保持空文本条件且避免本机常驻 T5-XXL |
| 静态 HD/BEV map | 固定为 B2：nuScenes expansion map → 8×200×200 bit-packed cache | 推理时可获得，训练/推理同源且不包含未来动态真值 |
| 未来 3D box | 禁止作为输入 | 推理不可得，会造成未来泄漏 |
| future RGB/latent | 只作为训练 target；clean 区仅 anchor | 防止 target 泄漏 |

- [x] `V2-COND-01` 从 anchor-relative Ego `[x,y,yaw]` 构造 17 个 4×4
  anchor→current (`next2top v2`) transform，第一帧为单位阵；manifest 另保存全局
  location/map pose 供静态 BEV 使用；
- [ ] `V2-COND-02` 用 MagicDrive `obtain_next2top` 对相同标定样本做方向/数值 golden
  test，覆盖直行、左转、右转和平移；
- [x] `V2-COND-03` 保留原 `CamEmbedderTemp(time_downsample_factor=4.5)` 时间契约：
  pose/action 先形成 17 个 RGB-time token，再用两次源码同构 `cog_temp_down` 对齐到
  latent T=5；只有不匹配的其他长度才允许进入模型的插值 fallback；
- [x] `V2-COND-04` 新增 `KinematicsEmbedder`，输入
  `[vx,vy,ax,ay,yaw_rate,steering] + valid mask`，输出 1152 维 per-time token；
- [x] `V2-COND-05` kinematics adapter 的最后投影零初始化，并以 frame-token residual
  注入，使初始网络等价于未加入
  新条件的已加载 checkpoint；
- [x] `V2-COND-06` 训练 condition dropout 同步替换 pose/action/camera 为各自 null，
  loss 返回并记录 `condition_drop_mask`；推理 CFG 使用相同 null contract；
- [x] `V2-COND-07` 世界模型 forward 使用显式白名单；未来 box 禁止进入，static map
  只允许来自 anchor 时刻全局 pose 的固定 8 通道缓存，future RGB 只作为训练 target；
- [ ] `V2-COND-08` 增加 zero/shuffle/counterfactual Ego 三组测试：输出应有限且 shape
  不变，训练后轨迹方向响应必须显著高于未训练 adapter；
- [ ] `V2-COND-09` 左/右转变换测试同时检查坐标系符号，不仅检查输出“有差异”。

### 10.8 数据频率、分辨率与样本准备

Stage-3 的视频训练 bucket 主要是 12 Hz，分辨率包含 224×400、424×800、
848×1600，长度包含 17 帧。当前项目 256×448 / 6 Hz 虽然 shape 可被动态位置编码处理，
但 6 Hz 是真实分布迁移，不能只依靠 `fps_embedder` 假设自动解决。

- [x] `V2-DATA-01` manifest schema v3 已包含 image path、target/image timestamp、
  3×7 camera 参数、anchor map pose/location、CAN action、valid mask；
- [ ] `V2-DATA-02` 已提供 `256×448 / 12 Hz / 1→16` 时间频率适配配置和
  `256×448 / 6 Hz / 8→16` 目标配置，官方 scene split 隔离；当前工作站只有约 9.8% RGB，
  99% 完整率 gate 会在写 manifest 前终止，需在全量数据服务器执行准备脚本后勾选；
- [x] `V2-DATA-03` build/validate report 已分别记录实际 camera 对齐误差 mean/p95/max、
  scene 可用帧数和拒绝原因；full 报告待服务器数据生成；
- [ ] `V2-DATA-04` 将 resize/crop 后的 camera intrinsic 同步变换并做投影测试；
- [x] `V2-DATA-05` 12/6 Hz 都只从真实 CAM_FRONT sample_data 时间轴做最近邻门限匹配，
  超过 45/55 ms 即拒绝，不伪造重复帧；
- [ ] `V2-DATA-06` 17-frame joint latent cache 使用原始数据、VAE config、scaling、
  sample seed、resize/crop、fps 和 commit 的完整 fingerprint；
- [ ] `V2-DATA-07` 统计 straight/left/right/stop、速度、yaw-rate、昼夜和天气分布，
  sampler 对稀有动作做可追踪加权；
- [ ] `V2-DATA-08` 固定一组 train/val/counterfactual clips 和 seeds，贯穿所有阶段。

### 10.9 分阶段训练与冻结策略

#### S0：原模型可复现性（不改结构）

- MagicDrive 原仓库、原 Stage-3 config、原 VAE、原 checkpoint；
- 只做推理/forward，不训练；
- 验收：checkpoint 可读、无 missing 主干 key、输出视频有限、固定 seed 可复现。

#### S1：单视角结构等价与权重转换（不训练）

- 建立 `MagicDriveSingleViewSTDiT`；
- `NC=1`、禁用 cross-view，先用空文本/真实 camera/relative pose；
- 验收：主干 100% shape match、B0/B1/B2 forward 均有清晰的 key 覆盖报告；
- 验收：17→5→17 VAE、mask、RF 端点全部通过。

#### S2：4070Ti 稳定调通

- 224×400 / 17 帧，batch=1，BF16，gradient checkpoint；
- 冻结 VAE、全部 MDDiT 主干和已加载 embedder；
- 只训练 kinematics/action adapter；若显存仍不足，对 denoiser CPU offload，
  此阶段允许慢但要求连续 100 step 无 NaN/OOM/native crash；
- optimizer 只包含新 adapter，EMA 也只跟踪 trainable 参数或使用差分 adapter EMA；
- 验收：保存/resume 后下一步 loss、optimizer/scaler/sampler state 正常。

#### S3：单 4090 adapter/LoRA 训练

- 先在 12 Hz reproduction split，再切 6 Hz target split；
- 训练 action adapter + temporal/cross-attention LoRA；base spatial 默认冻结；
- B0/B1/B2 中只选择 S1/S2 证据最好的分支；
- 显式比较 raw/EMA、CFG 1/2、30 steps，不同时扩大多个变量；
- 验收：固定 val 的末帧质量不再单调崩坏，Ego counterfactual 有方向一致性。

#### S4：多 4090 partial fine-tune

- 当前 4×5090 第一个正式版本使用 DDP、BF16、activation checkpoint，冻结完整 Stage-3
  主干，只训练 action adapter + temporal/cross-attention LoRA + AdaLN；本机实测完整模型
  backward reserved 8.823 GiB，复制主干在 32 GB/卡内有足够余量；
- 只有后续依次解冻 temporal blocks → cross-attn/AdaLN → 高层 spatial blocks，导致单卡
  optimizer/EMA 不再容纳时，才新增 FSDP/ZeRO-2/sequence parallel；
- 学习率分组：新 adapter > temporal/cross-attn > pretrained spatial；
- 每次解冻前后记录 trainable params、峰值显存、吞吐和验证退化；
- 不默认全量 Adam fine-tune；先用 LoRA/partial 证明收益。

#### S5：长时 rollout fine-tune

- 第一 chunk 真实 anchor→16 future；
- 后续 chunk 使用上一段生成末帧/latent 作为 anchor；
- 从 teacher-forced 逐步增加 generated-history 概率；
- 每段的 action/pose 时间轴必须连续，clean mask 重新锚定当前 anchor；
- 只在单段画质和控制性通过 gate 后进入。

- [ ] `V2-TRAIN-01` S2 adapter、S3/S4 12Hz LoRA、S3/S4 6Hz LoRA 已独立 YAML；S0/S1
  使用无训练脚本而非 YAML，S5 rollout 尚未实现；
- [x] `V2-TRAIN-02` LoRA YAML 固定冻结策略、rank/alpha/target、三组学习率和预计设备；
- [ ] `V2-TRAIN-03` 已实现轻量 adapter/EMA/optimizer/scheduler/scaler/RNG、resolved config、
  upstream/audit SHA 的保存和恢复；S2 每 25 step 会先原子写 `step-*.pt`/`last.pt` 再验证，
  防止 validation native crash 丢失已完成区间。尚缺精确 dataloader cursor，因此暂不勾选；
- [x] `V2-TRAIN-04` MDD resume 会校验 checkpoint SHA 记录、VAE/RF/fps、17→5
  latent mask/protocol 和训练 stage，不兼容即终止；
- [x] `V2-TRAIN-05` `scripts/train_mdd_adapter_smoke.sh` 以 25-step 分段运行，
  只对 native abort/segfault（134/139）从原子 `last.pt` 恢复；其他 Python/data/config
  错误立即退出，不并发写 manifest/cache；
- [x] `V2-TRAIN-06` 提供 `--dry-run`、`--run-steps N` 和显式
  `--start-training`；默认命令只做配置/数据/权重验证。
- [x] `V2-TRAIN-07` 新增 12Hz→6Hz `--init-checkpoint`：只迁移声明完整的
  LoRA/AdaLN/action delta，允许 fps/window/cache/training schedule 改变，重置 optimizer、
  scheduler、step、RNG；`--resume` 仍严格限制同阶段，二者互斥。
- [x] `V2-TRAIN-08` 新增 `prepare_mdd_full_trainval.sh`、`validate_mdd_4x5090.sh`、
  `launch_mdd_4x5090.sh`；launcher 默认拒绝训练，只有 `START_TRAINING=1` 才执行。

### 10.10 评测与放大训练 Gate

画质差和越往后越模糊必须拆成 VAE 上限、单段生成误差、条件失配和 rollout 漂移四类，
不能只看一个总 val loss。

- [ ] `V2-EVAL-01` VAE oracle：逐帧 PSNR/SSIM/LPIPS/edge；
- [ ] `V2-EVAL-02` 生成画质：每帧 LPIPS、DISTS 或等价感知指标、edge energy、
  temporal warping error，至少报告 frame 1/4/8/12/16；
- [ ] `V2-EVAL-03` 时序：静态背景 flicker、光流一致性、相邻帧 latent delta；
- [x] `V2-EVAL-04` 控制评测工具：固定噪声的 straight/left/right/stop/hold、
  zero-kinematics/invalid/shuffle Ego、逐帧 pair、速度趋势和方向 proxy 已落地；具体
  checkpoint 是否通过 G4 仍必须由多 clip/seed 的 step-zero/raw/EMA 报告判定；
- [ ] `V2-EVAL-05` 泄漏审计：删除未来 box/map 动态通道后指标不得异常塌陷；
- [ ] `V2-EVAL-06` checkpoint：raw 与 EMA、固定 K seeds 的 mean/best/diversity；
- [ ] `V2-EVAL-07` rollout：1/2/3 chunks 的逐帧曲线和 anchor seam；
- [ ] `V2-EVAL-08` B0/B1/B2、12/6 Hz、224×400/256×448 使用相同固定样本表；
- [ ] `V2-EVAL-09` 输出 HTML/视频报告，同时保存机器可读 JSON，禁止只挑最佳视频。

进入 4×5090 **首个 1000-step LoRA pilot** 前必须满足 G0/G1/G2；pilot 完成并验证断点恢复
后满足 G6。只有 pilot 的固定验证样本出现可测画质/控制收益（G3/G4），才把相同配置放大到
12Hz 50k → 6Hz 100k；G5 是进入多 chunk rollout 前置，不阻塞单 chunk LoRA pilot。

- **G0 权重**：主干严格加载，转换 report 无未解释 shape mismatch；
- **G1 数值**：VAE/RF/mask/pose golden tests 全过，100-step 无 NaN/native crash；
- **G2 数据**：scene 隔离、时间误差、缺帧、条件 valid mask 通过；
- **G3 I2V**：anchor 保持，末帧画质显著优于 V2-Lite 同预算基线；
- **G4 控制**：Ego shuffle/反事实测试证明模型使用条件且方向正确；
- **G5 rollout**：2/3 chunk 不出现立即崩溃、冻结帧漂移或时间轴错位；
- **G6 可恢复**：真实中断后 resume 结果与连续训练在容差内一致。

### 10.11 显存和部署边界

Stage-3 EMA 是完整 FP32 大模型 state dict，不能按当前 16M 模型的显存经验估算。

| 设备 | 允许任务 | 不作为目标 |
|---|---|---|
| 本机单卡 16 GB | checkpoint mmap；BF16 推理；adapter/LoRA forward/backward；极小 batch smoke | 完整 MDDiT Adam 全参训练 |
| 单台 4090/5090 | BF16 推理；冻结主干的 adapter/LoRA 稳定训练 | 未测算就全参 optimizer+EMA 常驻 |
| 4×5090 32 GB | 当前 DDP LoRA/AdaLN/action pilot 与正式 trainval；后续扩大解冻时再切 FSDP/ZeRO | 无数据/质量 gate 直接全参训练 |

- [x] `V2-SYS-01` mmap→目标参数逐张量 materialize，不再调用 meta
  `load_state_dict(assign=True)`；主干 BF16 估算 3.612 GB，避免 FP32+BF16+模型三份常驻；
- [x] `V2-SYS-02` VAE 冻结/no_grad 且不进入 checkpoint/optimizer；文本固定使用 checkpoint
  `base_token`，无需加载 T5；
- [x] `V2-SYS-03` 已提供无 optimizer 的真实数据 backward smoke，记录
  allocated/reserved/host RSS 和按 action/AdaLN/LoRA-A/LoRA-B 分组梯度；B2 LoRA
  256×448 实测 7.798/8.823 GiB、host RSS 约 8.2 GB，梯度全部有限；同分辨率 30-step
  CFG B1/B2 decode 峰值约 9.45 GiB；
- [ ] `V2-SYS-04` DDP torchrun launcher 和单进程 TCPStore/GPU dry-run 已通过；真实
  4×5090 NCCL、rank 数据分片、一次无 optimizer/all-reduce smoke 和 resume barrier 仍需在
  服务器执行，当前不声称本机单卡已验证四卡；
- [ ] `V2-SYS-05` 记录 MagicDrive checkpoint/license 的使用约束，发布产物前复核。

### 10.12 实施顺序和首批交付

```text
下载完成
  → V2-CKPT-01..07（原模型可复现 + 权重审计/转换）
  → V2-VAE-01..08 + V2-RF-01..08（17→5、mask、时间方向）
  → V2-MOD-01..10（同构 NC=1 主干）
  → V2-COND-01..09 + V2-DATA-01..08（pose/action、无未来泄漏）
  → S1 无训练 forward
  → S2 本机 adapter-only 100-step（仅用户显式启动）
  → G0/G1/G2/G6
  → 单 4090 S3
  → G3/G4
  → 多 4090 S4
  → G5 后再做 S5 rollout
```

接下来代码实现的首批文件计划：

- [x] `driveworld/models/magicdrive_single_view_stdit.py`：同构单视角 base 主干；
- [x] `driveworld/models/mdd_condition_adapter.py`：pose/action/null 条件；
- [x] `driveworld/diffusion/magic_rectified_flow.py`：Stage-3 同语义 RF；
- [x] `driveworld/models/magic_cogvideox_adapter.py`：17-frame joint VAE contract；
- [x] `scripts/audit_mdd_stage3_checkpoint.py`：mmap 审计；
- [ ] `scripts/convert_mdd_stage3_singleview.py`：BF16 safetensors + report；
- [x] `scripts/test_mdd_singleview_load.py`、`test_mdd_end_to_end.py`：无训练、固定 seed smoke；
- [x] `scripts/inference_mdd.py`：30-step Euler/CFG 推理与 GIF/JSON 输出；
- [x] `scripts/smoke_mdd_adapter_backward.py`：无 optimizer 的梯度/显存 smoke；
- [x] `scripts/train_mdd_adapter_smoke.sh`：仅 native crash 可恢复的 25-step S2 watchdog；
- [x] `scripts/cache_static_maps.py`：manifest-SHA 校验、bit-packed mmap、隔离分段续跑；
- [x] `scripts/prepare_mdd_full_trainval.sh`：12/6 Hz full manifest、静态地图缓存和数据验证；
- [x] `scripts/validate_mdd_4x5090.sh`、`launch_mdd_4x5090.sh`：DDP dry-run 与显式训练门禁；
- [x] `configs/model/v2_mdd_stage3_singleview.yaml`；
- [x] `configs/model/v2_mdd_stage3_singleview_lora_{12hz,6hz}.yaml`；
- [x] `configs/train/v2_mdd_local_adapter_smoke.yaml`；
- [x] `configs/train/v2_mdd_4x5090_lora_{12hz,6hz}.yaml`：DDP、batch/accumulation、
  action/LoRA/AdaLN 分组学习率和独立输出目录；
- [x] `tests/test_mdd_checkpoint_*`、`test_mdd_vae_contract.py`、
  `test_mdd_rf_contract.py`、`test_mdd_condition_adapter.py`、
  `test_mdd_world_model.py`、`test_mdd_adapter_checkpoint.py`。

本机实现里程碑已经完成：Stage-3 EMA 审计/直接 BF16 materialize、17 帧 joint VAE、
`[False,True,True,True,True]` mask、MagicDrive RF/CFG、`NC=1` B2 control、LoRA backward、
地图缓存、增量 checkpoint 与单进程 torchrun dry-run 全部通过。下一项不是在当前 partial
数据上继续堆步数，而是在全量数据服务器依次执行 prepare → 4-rank dry-run → 1000-step
12Hz pilot；仍然只有显式 `START_TRAINING=1` 才会执行 optimizer step。

### 10.13 4×5090 执行顺序（训练不自动启动）

```bash
conda activate driveworld-v2
cd /home/liang/code/DriveWorld-lite_V2

# 1. 全量数据：构建 12/6 Hz manifest、bit-packed static-map cache，并验证图片/scene。
./scripts/prepare_mdd_full_trainval.sh

# 2. 四卡只读加载和 DDP contract；没有 optimizer step。
NPROC_PER_NODE=4 MODE=12hz ./scripts/validate_mdd_4x5090.sh

# 3. 用户明确启动的 12 Hz 1000-step pilot。
START_TRAINING=1 NPROC_PER_NODE=4 MODE=12hz RUN_STEPS=1000 \
  ./scripts/launch_mdd_4x5090.sh
```

1000-step pilot 必须先检查 loss/grad/显存、`last.pt`、固定 val 推理和一次 `RESUME` 小段。
确认后，用同阶段 `RESUME=artifacts/runs/v2-mdd-4x5090-lora-12hz/last.pt` 继续到 50k。
12Hz 完成后切 6Hz 时不得 `RESUME`；先验证并只迁移 delta：

```bash
INIT_CHECKPOINT=artifacts/runs/v2-mdd-4x5090-lora-12hz/last.pt \
  NPROC_PER_NODE=4 MODE=6hz ./scripts/validate_mdd_4x5090.sh

START_TRAINING=1 NPROC_PER_NODE=4 MODE=6hz RUN_STEPS=1000 \
  INIT_CHECKPOINT=artifacts/runs/v2-mdd-4x5090-lora-12hz/last.pt \
  ./scripts/launch_mdd_4x5090.sh
```

6Hz pilot 通过后，后续进程使用
`RESUME=artifacts/runs/v2-mdd-4x5090-lora-6hz/last.pt`，不要再次使用
`INIT_CHECKPOINT`，否则会把 global step 和 optimizer 日程重新置零。
