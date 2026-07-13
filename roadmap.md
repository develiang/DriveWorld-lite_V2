DriveWorld-lite V2 Roadmap

结合你现在已经完成的内容：

* nuScenes 数据下载与基础处理；
* CAM_FRONT 连续帧提取；
* Ego 条件 3D U-Net 基础实验；
* MagicDrive-V2 环境搭建和推理；
* 本地 RTX 5070 Ti 16GB；
* 云端 多4090/5090训练

下一步不应该直接挑战“六路相机、20→50 帧、文本+地图+LiDAR 全条件”，而是先完成一个真正可训练、可评估、可控制的：

单前视角、Ego 轨迹条件、潜空间视频预测模型。

⸻

一、最终项目路线

整个项目可以拆成八个里程碑：

阶段	目标	输出
M0	单前视角数据管线	CAM_FRONT 时序训练集
M1	确定性视频预测基线	3D U-Net 未来帧预测
M2	单前视角 Latent Diffusion	可生成多种合理未来
M3	Ego 轨迹可控预测	左转、右转、减速等控制
M4	Map、3D Box、LiDAR BEV 条件	几何结构控制
M5	文本控制	文本描述交通行为
M6	17→48 长时预测	约 4 秒未来视频
M7	三视角、六视角扩展	多相机一致性
M8	BEV Dynamics + 闭环仿真	完整世界模型雏形

第一阶段的重点是 M0～M3。

建议用大约 4～6 周完成一个质量合格的单前视角版本。

⸻

二、第一版目标定义

不要一开始使用 20→50，第一版固定为：

输入：
CAM_FRONT 历史 8 帧
历史 Ego 状态 8 帧
未来 Ego 轨迹 16 帧
输出：
CAM_FRONT 未来 16 帧
采样帧率：
6 Hz
历史时长：
约 1.33 秒
未来时长：
约 2.67 秒
分辨率：
256 × 448

nuScenes 相机原始采集频率约为 12 Hz，但带完整人工标注的关键帧是 2 Hz。因此，第一版可以从原始相机流中下采样到 6 Hz，同时对 Ego 位姿和 CAN 信号进行时间插值。 

为什么选 8→16：

1. 24 帧总长度在 A800 80GB 上容易训练；
2. 2.67 秒未来已经可以观察转向、跟车、刹车；
3. 比 4→4 更接近真正的视频世界模型；
4. 后面很容易扩展到 12→24、17→32；
5. 先解决可控性，再解决超长视频。

⸻

三、模型任务定义

模型学习：

p\left(
X_{t+1:t+16}
\mid
X_{t-7:t},
E_{t-7:t},
A_{t+1:t+16}
\right)

其中：

* X：CAM_FRONT 图像；
* E：历史 Ego 状态；
* A：未来 Ego 轨迹；
* 输出是未来 16 帧 CAM_FRONT 图像。

注意，未来 Ego 轨迹应该从第一版就加入。

只输入过去 Ego 状态时，模型无法知道车辆接下来要左转、直行还是右转。训练阶段可以使用 nuScenes 中真实发生的未来 Ego 轨迹作为条件。

⸻

四、M0：重构单前视角数据管线

预计时间：3～5 天。

4.1 样本结构

每个训练样本建议保存为：

{
    "past_rgb":       [8, 3, 256, 448],
    "future_rgb":     [16, 3, 256, 448],
    "past_ego":       [8, D],
    "future_ego":     [16, D],
    "timestamps":     [24],
    "scene_token":    str,
    "start_timestamp": int,
    "sample_valid":   bool
}

第一版 Ego 特征定义为：

relative_x
relative_y
relative_yaw
velocity_x
velocity_y
acceleration_x
acceleration_y
yaw_rate
steering

不建议把绝对经纬度输入模型。

所有轨迹都转换到：

最后一帧历史图像对应的 Ego 坐标系。

也就是：

最后一帧历史位置：
x = 0
y = 0
yaw = 0

未来轨迹表示自车相对于这一时刻如何运动。

⸻

4.2 数据采样

建议配置：

camera: CAM_FRONT
fps: 6
history_frames: 8
future_frames: 16
stride: 2
resolution:
  height: 256
  width: 448

需要严格保证：

* 不跨 scene 切片；
* 不穿过数据缺失区间；
* 相邻图像时间差正常；
* CAN/Ego 插值覆盖完整；
* 训练集和验证集按 scene 划分；
* 同一 scene 不能同时出现在训练和验证中。

⸻

4.3 CAN 数据处理

nuScenes CAN 扩展包含位置、速度、加速度、转向等低层车辆信号，但官方也提示该部分数据具有一定实验性质，因此必须处理缺失值和异常值。 

建议处理流程：

CAN 原始时间戳
      ↓
统一到相机时间戳
      ↓
线性插值连续量
      ↓
角度 unwrap 后插值
      ↓
异常值裁剪
      ↓
转换到相对 Ego 坐标

对于缺少 CAN 的 scene：

* 优先使用 ego_pose 计算位置、速度和 yaw；
* steering 等字段可以填零并增加 valid_mask；
* 不要直接删除所有缺 CAN 的样本。

⸻

4.4 数据可视化验收

训练之前必须生成一个数据检查视频：

左侧显示 CAM_FRONT，右侧显示：

当前速度
转向角
yaw rate
未来 Ego 轨迹
当前帧编号
scene token

未来 Ego 轨迹可以画成简单的鸟瞰折线：

       future path
           /
          /
ego ----●

M0 通过条件

随机抽取 100 个 clip，确认：

* 没有跨场景；
* 视频连续；
* 轨迹与车辆运动方向一致；
* 左转场景的未来轨迹确实向左弯；
* 刹车场景速度确实下降；
* 图像与 CAN 没有明显错位。

⸻

五、M1：确定性预测基线

预计时间：4～7 天。

你之前已经完成过简单 3D U-Net 和 Ego condition ablation，因此这一阶段不需要继续长期优化，而是把它变成规范的基线。

5.1 必须保留的三个基线

Baseline 0：Last Frame

把最后一帧重复 16 次：

future[i] = past[-1]

它可以检验指标是否只是偏爱静态视频。

Baseline 1：无 Ego 的 3D U-Net

历史 8 帧
    ↓
3D U-Net
    ↓
未来 16 帧

Baseline 2：有 Ego 的 3D U-Net

历史 8 帧 + 未来 Ego 轨迹
                ↓
             3D U-Net
                ↓
           未来 16 帧

Ego 条件可以先通过：

未来 Ego [16, D]
       ↓
Temporal MLP / Transformer
       ↓
Ego Embedding
       ↓
FiLM / AdaLN

注入网络。

⸻

5.2 损失函数

确定性模型可以使用：

L =
L_{\text{Charbonnier}}
+
0.1L_{\text{LPIPS}}
+
0.1L_{\text{temporal}}

其中时序损失比较相邻帧变化：

L_{\text{temporal}}
=
\left\|
(\hat X_{t+1}-\hat X_t)
-
(X_{t+1}-X_t)
\right\|_1

这一阶段的目标不是生成高清图像，而是验证：

* 数据管线正确；
* 模型能够预测明显运动；
* Ego 条件确实有效；
* 训练集能够被拟合。

⸻

5.3 分级训练

按照下面的顺序执行：

Test A：过拟合 16 个 clip

样本数：16
训练步数：2000～5000

必须能够明显过拟合。

Test B：过拟合一个 scene

样本数：一个完整 scene
训练步数：5000～10000

Test C：训练 10% 数据

训练步数：20000～40000

Test D：完整数据训练

训练步数：50000～100000

任何模型如果无法过拟合 16 个 clip，都不要进入完整训练。

⸻

六、M2：单前视角 Latent Video Diffusion

预计时间：1～2 周。

这一阶段才是 DriveWorld-lite V2 的核心。

6.1 第一版不要直接改 MagicDrive-V2

MagicDrive-V2 已经支持文本、道路地图、3D Box、相机参数和时空条件，并采用渐进式训练提升分辨率和视频长度，但它原本主要是多视角条件视频生成，不是“历史视频→未来视频预测”。直接修改它，会同时遇到多视角、条件编码、分布式训练和历史帧注入问题。 

更适合的第一版是：

使用预训练视频 VAE，但自己训练一个较小的未来帧 Diffusion Backbone。

⸻

6.2 推荐架构

历史 RGB 8 帧
      ↓
冻结的 Video VAE
      ↓
历史 Latent
      ─────────────────┐
                       │
未来 RGB 16 帧         │
      ↓                │
冻结的 Video VAE       │
      ↓                │
未来 Latent            │
      ↓                │
添加噪声               │
      ↓                │
Noisy Future Latent ───┼──→ Video DiT / 3D U-Net
                       │
未来 Ego Trajectory ───┤
                       │
Diffusion Timestep ─────┘
                              ↓
                         预测噪声 / v

建议复用 CogVideoX 体系中的 3D VAE。CogVideoX 使用时空压缩的 3D VAE，Diffusers 也提供了官方训练和 Image-to-Video LoRA 脚本，可以作为第二阶段迁移预训练模型的参考。 

⸻

6.3 历史帧注入方式

第一版推荐使用 Masked Video Diffusion。

构造完整的 24 帧 latent：

[历史 8 帧 | 未来 16 帧]

其中：

历史 latent：始终保持干净
未来 latent：执行加噪和去噪

增加一个 mask：

历史帧位置：mask = 1
未来帧位置：mask = 0

模型输入：

noisy_latent
known_history_latent
history_mask
ego_embedding
timestep

损失只计算未来帧：

L =
\left\|
M_{\text{future}}
\odot
(\epsilon-\epsilon_\theta)
\right\|_2^2

这种方式比“把最后一帧作为普通图片条件”更符合视频预测任务，因为模型能够看到完整历史运动。

⸻

6.4 Backbone 规模

第一版建议：

model_type: latent_3d_unet_or_small_dit
parameters: 200M-400M
latent_channels: follow_vae
hidden_dim: 768
num_layers: 16-24
attention:
  spatial: true
  temporal: true
cross_attention:
  ego: true
text_condition: false

不要从 1B、2B 参数起步。

nuScenes 的独立场景规模不足以支撑大型视频模型随机初始化训练；视频生成研究也普遍依赖图像预训练、视频预训练和高质量数据微调等多个阶段。 

⸻

6.5 Ego 条件编码

未来 Ego 条件不要压缩成一个全局向量，而应该保留时间结构：

future_ego: [B, 16, D]
          ↓
位置 Fourier Embedding
          ↓
Temporal Transformer
          ↓
ego_tokens: [B, 16, C]

注入方式：

* Cross-Attention：视频 latent 查询 Ego token；
* AdaLN：用 Ego token 调制对应未来时刻；
* Temporal Alignment：第 i 个未来 latent 重点关注第 i 段 Ego 轨迹。

推荐组合：

Cross-Attention + AdaLN

⸻

6.6 训练配置

A800 80GB 推荐起始配置：

resolution: [256, 448]
history_frames: 8
future_frames: 16
fps: 6
precision: bf16
gradient_checkpointing: true
flash_attention: true
micro_batch_size: 2
gradient_accumulation_steps: 16
effective_batch_size: 32
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
warmup_steps: 2000
training_steps: 100000
ema: true
diffusion_prediction: v_prediction
condition_dropout:
  ego: 0.1

实际 batch size 需要根据 VAE latent 尺寸和 Backbone 调整。

本地 5070 Ti 调试配置

resolution: [128, 224]
history_frames: 4
future_frames: 8
micro_batch_size: 1
model_parameters: 50M-100M

本地卡只用于：

* 检查数据；
* 跑通 forward；
* 检查 loss；
* 过拟合小数据；
* 测试推理脚本。

完整训练放到 A800。

⸻

七、M3：验证 Ego 可控性

预计时间：4～7 天。

模型能生成未来视频，不代表它真的使用了 Ego 条件。

必须做反事实控制实验。

7.1 同一历史，不同未来轨迹

固定同一个历史视频：

历史视频完全相同

分别输入：

轨迹 A：直行
轨迹 B：向左偏移
轨迹 C：向右偏移
轨迹 D：减速停车

生成四段视频。

理想结果：

* 左转时画面消失点和道路结构向对应方向变化；
* 右转时相反；
* 减速时前方车辆和道路的光流幅度减小；
* 停车时背景变化逐渐趋近于零。

⸻

7.2 Ego 消融实验

至少训练或推理比较：

Model A：无 Ego
Model B：只有历史 Ego
Model C：历史 Ego + 未来 Ego 轨迹
Model D：打乱未来 Ego 轨迹

关键结论应该是：

C 优于 A 和 B
D 明显劣于 C

否则模型可能只是在依赖历史图像生成最常见未来。

⸻

7.3 控制指标

除了 PSNR、SSIM 和 LPIPS，还需要加入控制指标：

Ego Motion Alignment

从生成视频估计自车相机运动，与输入 Ego 轨迹比较：

输入 yaw 变化
vs
生成视频估计 yaw 变化

Optical Flow Alignment

比较生成视频和真实视频的全局光流方向及幅度。

Lane Motion Alignment

使用车道线检测模型，比较车道线在未来帧中的横向变化。

Condition Sensitivity

计算改变 Ego 轨迹后，生成视频发生了多大变化：

S =
\frac{
D(V_{\text{left}},V_{\text{right}})
}{
D(A_{\text{left}},A_{\text{right}})
}

如果不同轨迹生成的视频几乎一样，说明模型忽略了条件。

⸻

八、第一阶段验收标准

单前视角模型满足以下条件，才进入 Map、LiDAR 和文本阶段。

数据层

* 能连续可视化 100 个有效 clip；
* 图像和 Ego 轨迹时间同步；
* 无 scene 泄漏；
* 缺失数据有明确 mask。

模型层

* 能过拟合 16 个 clip；
* 训练 loss 稳定下降；
* 输出不是简单重复最后一帧；
* 车辆、车道线不会在前几帧立刻变形。

预测层

* 未来 1 秒相对稳定；
* 未来 2～3 秒保持基本道路结构；
* 前方车辆具有连续运动；
* 历史帧与第一帧未来图像没有明显跳变。

控制层

* 改变未来 Ego 轨迹会改变生成视频；
* 左右转控制方向正确；
* 减速和停车控制可以观察到；
* 打乱 Ego 条件会降低预测质量。

工程层

* 单卡训练可恢复；
* checkpoint 包含模型、优化器、EMA 和随机状态；
* 固定验证集和随机种子；
* 每个 checkpoint 自动生成对比视频；
* 推理脚本支持自定义 Ego 轨迹。

⸻

九、单前视角完成后的扩展顺序

阶段 1：加入 HD Map

输入局部 BEV Map：

drivable area
lane divider
road divider
crosswalk

先使用 Raster Map + CNN Encoder，不急着使用 Vector Map Transformer。

⸻

阶段 2：加入 3D Boxes

输入动态目标：

class
position
size
yaw
velocity
track_id

目标是提升：

* 车辆身份一致性；
* 运动轨迹一致性；
* 前景目标可控性。

⸻

阶段 3：加入 LiDAR/Radar BEV

先加入历史 LiDAR occupancy/depth BEV，再加入 Radar velocity BEV。

不要直接输入原始点云。

⸻

阶段 4：加入文本

文本分两类。

场景描述文本

An urban road with several vehicles ahead.
The ego vehicle is approaching an intersection.

可以直接通过文本编码器和 Cross-Attention 输入。

行为控制文本

自车在路口左转
前车突然刹车
左侧车辆切入

建议先转换为结构化条件：

文本
 ↓
事件解析
 ↓
Ego trajectory / Agent trajectory
 ↓
视频生成

需要注意：nuScenes 没有同一历史场景下的多种反事实未来。因此，仅靠真实数据不能严格训练：

同一场景：
晴天 → 雨天
直行 → 左转
无行人 → 行人突然出现

第一版文本应该描述真实发生的行为，而不是任意修改世界。真正的反事实文本控制需要：

* 数据增强；
* 仿真数据；
* 伪标签；
* 结构化轨迹编辑；
* 预训练视频生成模型。

⸻

十、长时预测路线

单前视角的长度建议逐级增加：

阶段 A：
8 → 16，6 Hz
1.33 秒 → 2.67 秒
阶段 B：
12 → 24，6 Hz
2 秒 → 4 秒
阶段 C：
17 → 32，8 Hz
2.1 秒 → 4 秒
阶段 D：
17 → 48，12 Hz
1.4 秒 → 4 秒

不要从 8→16 直接跳到 20→50。

对于 50 帧预测，可以先采用滚动生成：

历史 8 帧
   ↓
生成未来 16 帧
   ↓
取最后 8 帧作为新历史
   ↓
继续生成未来 16 帧

随后加入：

* 4～8 帧 overlap；
* latent blending；
* shared noise initialization；
* trajectory continuity；
* history memory tokens。

直接 17→48 留到滚动预测跑通之后。

⸻

十一、建议的代码结构

DriveWorld-lite-V2/
├── configs/
│   ├── data/
│   │   └── nuscenes_front_8x16_6hz.yaml
│   ├── model/
│   │   ├── unet3d_baseline.yaml
│   │   └── latent_diffusion_ego.yaml
│   └── train/
│       ├── debug.yaml
│       └── a800.yaml
│
├── driveworld/
│   ├── data/
│   │   ├── nuscenes_front_dataset.py
│   │   ├── can_interpolator.py
│   │   ├── ego_transform.py
│   │   └── clip_sampler.py
│   │
│   ├── models/
│   │   ├── video_vae.py
│   │   ├── video_dit.py
│   │   ├── ego_encoder.py
│   │   ├── masked_diffusion.py
│   │   └── unet3d_baseline.py
│   │
│   ├── diffusion/
│   │   ├── scheduler.py
│   │   ├── loss.py
│   │   └── sampler.py
│   │
│   ├── evaluation/
│   │   ├── image_metrics.py
│   │   ├── temporal_metrics.py
│   │   ├── ego_alignment.py
│   │   └── generate_report.py
│   │
│   └── visualization/
│       ├── render_clip.py
│       └── render_ego_path.py
│
├── scripts/
│   ├── build_front_clips.py
│   ├── validate_dataset.py
│   ├── cache_vae_latents.py
│   ├── train_baseline.py
│   ├── train_diffusion.py
│   └── inference_counterfactual.py
│
├── tests/
├── train.py
├── evaluate.py
└── inference.py

⸻

十二、推荐时间表

第 1 周：数据与确定性基线

* CAM_FRONT 6 Hz 数据集；
* Ego/CAN 时间对齐；
* 8→16 clip；
* 数据可视化；
* Last Frame baseline；
* Ego 3D U-Net baseline。

第 2 周：Latent Diffusion 跑通

* 接入 Video VAE；
* latent cache；
* masked diffusion；
* 16 个 clip 过拟合；
* 单 scene 过拟合。

第 3 周：完整训练

* 完整 nuScenes 训练；
* 验证集固定采样；
* EMA；
* PSNR、SSIM、LPIPS、时序指标；
* 生成视频对比。

第 4 周：Ego 可控性

* 未来 Ego trajectory encoder；
* 直行、左转、右转、停车实验；
* Ego condition ablation；
* 控制指标。

第 5 周：几何条件

* HD Map；
* 3D Boxes；
* Track ID；
* 几何一致性评估。

第 6 周：文本和长时预测

* 自动生成 Caption；
* 行为事件标签；
* Text Encoder；
* 8→16 滚动到 48 帧；
* Demo 和 README。

⸻

十三、第一阶段最终 Demo

单前视角阶段结束时，最好交付一个四宫格 Demo：

┌────────────────┬────────────────┐
│ 历史输入视频     │ Ground Truth   │
├────────────────┼────────────────┤
│ 无 Ego 预测      │ Ego 可控预测    │
└────────────────┴────────────────┘

再提供一个反事实控制 Demo：

同一段历史：
直行轨迹 → 生成视频 A
左转轨迹 → 生成视频 B
右转轨迹 → 生成视频 C
停车轨迹 → 生成视频 D

这个结果比单纯展示一段“看起来不错”的驾驶视频更有项目含金量，因为它能证明模型不是普通视频生成器，而是在学习：

给定历史世界状态和未来动作，预测动作执行后的未来观测。
