# RoboTwin 数据说明文档

本文档介绍 RoboTwin 数据从采集到训练所涉及的 **三种数据格式**，以及它们之间的转换关系。

## 三种数据格式及转换流程

在整个训练流程中，数据会经历三次格式变化：

```
┌─────────────────────┐     preprocess_aloha.py     ┌─────────────────────┐     robotwin_builder.py     ┌─────────────────────┐
│  ① RoboTwin 原始数据 │  ────────────────────────>  │  ② ALOHA 中间格式    │  ────────────────────────>  │  ③ RLDS (TFRecord)  │
│     (HDF5)          │                              │     (HDF5)          │                              │   训练最终格式       │
└─────────────────────┘                              └─────────────────────┘                              └─────────────────────┘
  仿真环境直接采集                                      统一图像尺寸                                         TensorFlow 标准格式
  包含完整传感器数据                                    划分 train/val、写入语言指令                           OpenVLA-OFT 直接读取
  4 相机 JPEG + 内外参                                 保留关节角度绝对值 (action)
  关节绝对角度 + 末端绝对位姿                           丢弃末端位姿 (endpose)
```

**动作（action）在三个阶段的流转：**

| 阶段 | 字段 | 物理含义 | 绝对/相对 | 空间 | 备注 |
|------|------|---------|-----------|------|------|
| ① 原始 HDF5 | `endpose/*` | 末端执行器 XYZ + 四元数 | **绝对** | 笛卡尔空间 | ⚠️ 后续流程**未使用**，仅供分析 |
| ① 原始 HDF5 | `joint_action/vector` | 14维双臂关节角度 | **绝对** | 关节空间 | ✅ 这是贯穿全流程的核心 action |
| ② ALOHA HDF5 | `action` | 14维双臂关节角度 | **绝对** | 关节空间 | 原样拷贝自 `joint_action/vector` |
| ② ALOHA HDF5 | `relative_action` | 14维关节角度差分 | **相对** | 关节空间 | ⚠️ 当前流程**未使用**（冗余计算） |
| ③ RLDS | `action` | 14维双臂关节角度 | **绝对** | 关节空间 | 取自 ② 的 `action[t+1]` |
| ③ RLDS | `state` | 14维双臂关节角度 | **绝对** | 关节空间 | 取自 ② 的 `action[t]`（当前帧状态） |

> **关键结论**：整条流水线中，实际参与训练的 action 始终是**关节角度绝对值**，而非末端位姿。末端位姿（endpose）在 ① 采集后就被丢弃了。训练出的模型（OpenVLA-OFT）输出的也是 14 维绝对关节角度，直接发给机器人关节控制器执行。

**每种格式的定位：**

| 格式 | 文件类型 | 存放位置 | 用途 | 本文对应章节 |
|------|---------|---------|------|------------|
| ① RoboTwin 原始数据 | HDF5 (.hdf5) | `data/{task}/{config}/data/` | 仿真采集的原始数据，保留所有信息 | [RoboTwin 原始 HDF5 格式](#robotwin-原始-hdf5-格式) |
| ② ALOHA 中间格式 | HDF5 (.hdf5) | `data/{task}/processed_openvla/` | 预处理后的中间格式，统一了图像尺寸和字段命名 | [ALOHA 中间 HDF5 格式](#aloha-中间-hdf5-格式) |
| ③ RLDS (TFRecord) | TFRecord | `~/tensorflow_datasets/aloha_{task}/` | 训练用的最终格式，OpenVLA-OFT 直接读取 | [RLDS (TFRecord) 格式](#rlds-tfrecord-格式) |


---

# ① RoboTwin 原始 HDF5 格式

## 目录结构概览

```
data/
├── README_data.md                  # 本说明文档
├── visualize_hdf5.py               # HDF5 数据可视化脚本
├── process_stuck.py                # 数据采集卡住时的处理脚本
└── {task_name}/
    └── {task_config}/              # 如 demo_clean / demo_randomized
        ├── data/                   # 轨迹数据（HDF5 格式）
        │   ├── episode0.hdf5
        │   ├── episode1.hdf5
        │   └── ...
        ├── instructions/           # 语言指令（JSON 格式）
        │   ├── episode0.json
        │   └── ...
        ├── video/                  # 头部相机录制视频（MP4 格式）
        │   ├── episode0.mp4
        │   └── ...
        ├── _traj_data/             # 辅助文件：细粒度动作轨迹（采集过程中间产物）
        ├── scene_info.json         # 辅助文件：场景配置信息
        └── seed.txt                # 辅助文件：随机种子记录
```

---

## RoboTwin 原始 HDF5 格式

### 什么是 HDF5？

HDF5（Hierarchical Data Format version 5）是一种用于存储和管理大规模科学数据的**层级式二进制文件格式**。它的核心概念类似于文件系统：

- **Group（组）**：类似于文件夹，可以嵌套，用于组织数据层级
- **Dataset（数据集）**：类似于文件，存储实际的多维数组数据
- **Attribute（属性）**：附加在 Group 或 Dataset 上的元数据

HDF5 的优势：
- 支持存储任意维度的大规模数组（如图像、轨迹、点云）
- 支持数据压缩，存储效率高
- 支持随机访问，无需读取整个文件即可获取特定数据
- 跨平台、跨语言通用（Python / C++ / MATLAB 等均有支持库）

### RoboTwin 原始 HDF5 的层级结构

以 `episode0.hdf5`（约 6MB，包含 116 个时间步）为例：

```
episode0.hdf5
│
├── endpose/                              # 末端执行器位姿（Task Space）
│   ├── left_endpose    (T, 7)  float64   # 左臂: XYZ位置(3) + 四元数姿态(4)
│   ├── left_gripper    (T,)    float64   # 左夹爪: 1=张开, 0=闭合
│   ├── right_endpose   (T, 7)  float64   # 右臂: XYZ位置(3) + 四元数姿态(4)
│   └── right_gripper   (T,)    float64   # 右夹爪: 1=张开, 0=闭合
│
├── joint_action/                         # 关节空间动作（Joint Space）
│   ├── left_arm        (T, 6)  float64   # 左臂 6 个关节角度
│   ├── left_gripper    (T,)    float64   # 左夹爪
│   ├── right_arm       (T, 6)  float64   # 右臂 6 个关节角度
│   ├── right_gripper   (T,)    float64   # 右夹爪
│   └── vector          (T, 14) float64   # 拼接向量: [左臂6维, 左夹爪, 右臂6维, 右夹爪]
│
├── observation/                          # 多相机观测
│   ├── head_camera/                      # 头部相机
│   │   ├── rgb          (T,)   bytes     # JPEG 编码的图像比特流
│   │   ├── cam2world_gl (T,4,4) float32  # 相机→世界 变换矩阵 (OpenGL)
│   │   ├── extrinsic_cv (T,3,4) float32  # 相机外参矩阵 (OpenCV)
│   │   └── intrinsic_cv (T,3,3) float32  # 相机内参矩阵
│   ├── front_camera/                     # 前方相机（结构同上）
│   ├── left_camera/                      # 左侧相机（结构同上）
│   └── right_camera/                     # 右侧相机（结构同上）
│
└── pointcloud           (T, 0) float64   # 点云数据（部分任务可能为空）
```

> 其中 `T` 表示该 episode 的总时间步数，不同 episode 的 T 可能不同。

### 数据字段详细说明

#### 1. endpose — 末端执行器位姿（笛卡尔空间，绝对值）

> ⚠️ **注意**：endpose 仅在原始 HDF5 中存在。后续的 ALOHA 预处理和 RLDS 转换**均不使用**此字段——它会被直接丢弃。训练用的是下面的 joint_action。

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `left_endpose` | (T, 7) | float64 | 左臂末端位姿（**绝对值**）：前3维为 XYZ 坐标（米），后4维为四元数姿态 (w, x, y, z) |
| `left_gripper` | (T,) | float64 | 左夹爪开合状态：1.0 = 完全张开，0.0 = 完全闭合 |
| `right_endpose` | (T, 7) | float64 | 右臂末端位姿（**绝对值**），格式同 left_endpose |
| `right_gripper` | (T,) | float64 | 右夹爪开合状态 |

#### 2. joint_action — 关节空间动作（关节空间，绝对值）⭐ 训练核心字段

> ✅ **这是贯穿整个训练流水线的核心 action 字段。** `vector` 字段会被原样传递到 ALOHA 中间格式和 RLDS 最终格式，也是模型学习预测的目标。

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `left_arm` | (T, 6) | float64 | 左臂 6 个关节的**绝对**角度值（弧度） |
| `right_arm` | (T, 6) | float64 | 右臂 6 个关节的**绝对**角度值（弧度） |
| `left_gripper` | (T,) | float64 | 左夹爪 |
| `right_gripper` | (T,) | float64 | 右夹爪 |
| `vector` | (T, 14) | float64 | 14维拼接向量（**绝对值**）：`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]` |

#### 3. observation — 多相机观测

每个相机（head / front / left / right）都包含以下字段：

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `rgb` | (T,) | bytes | **JPEG 编码的图像比特流**（非原始像素数组），每帧是一个变长字节串 |
| `cam2world_gl` | (T, 4, 4) | float32 | 相机坐标系到世界坐标系的变换矩阵（OpenGL 约定） |
| `extrinsic_cv` | (T, 3, 4) | float32 | 相机外参矩阵（OpenCV 约定），包含旋转和平移 |
| `intrinsic_cv` | (T, 3, 3) | float32 | 相机内参矩阵，包含焦距和主点偏移 |

#### 4. pointcloud — 点云

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `pointcloud` | (T, N) | float64 | 点云数据。N=0 表示该任务未采集点云 |

### 图像解码方法

HDF5 中的图像以 **JPEG bit stream** 形式存储（文件头标识 `0xFFD8FFE0`），而非原始像素数组。这样做可以大幅减小文件体积。

恢复为 numpy 图像数组的方法：

```python
import cv2
import numpy as np

# 从 HDF5 读取图像比特流
image_bytes = f['observation/head_camera/rgb'][0]

# 解码为 BGR 格式的 numpy 数组 (H, W, 3)
image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

# 如需 RGB 格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

## instructions 目录 — 语言指令

每个 episode 对应一个 JSON 文件，包含两类自然语言指令：

```json
{
  "seen": ["指令1", "指令2", ...],    // 100 条「已见」指令（训练时可用）
  "unseen": ["指令1", "指令2", ...]   // 100 条「未见」指令（测试泛化时使用）
}
```

这些指令是对该 episode 中机器人执行任务的多样化自然语言描述，涵盖不同的措辞和表达方式。

### seen / unseen 指令的生成机制

指令的生成是一个**两层模板填充**的过程，由脚本 `description/utils/generate_episode_instructions.py` 完成。

#### 第一层：任务级指令模板

每个任务在 `description/task_instruction/{task_name}.json` 中预定义了两组**指令模板**。以 `beat_block_hammer` 为例：

```json
{
  "seen": [
    "Pick {A} and strike the block.",
    "Lift {A} using {a} to hit the block.",
    ...
  ],
  "unseen": [
    "Grab {A} and beat the block.",
    "Use {a} to grab {A}, beat the block.",
    ...
  ]
}
```

其中 `{A}` 是物体占位符，`{a}` 是机械臂占位符。

#### 第二层：物体描述词

每个物体在 `description/objects_description/` 中也有 seen/unseen 两组**描述词**。以锤子 `020_hammer/base0.json` 为例：

```json
{
  "seen": [
    "silver hammer",
    "nail-driving hammer",
    "grippy handle hammer",
    ...
  ],
  "unseen": [
    "hammer with black handle",
    "silver hammer head and claw",
    ...
  ]
}
```

#### 组合生成过程

数据采集完成后，脚本根据 `scene_info.json` 中记录的每个 episode 的场景参数（如使用哪个物体变体、用哪只机械臂），进行模板填充：

| 指令类型 | 指令模板来源 | 物体描述词来源 | 生成示例 |
|----------|-------------|---------------|---------|
| **seen** | seen 模板池 | seen 描述词池 | "Lift **the nail-driving hammer** using **the right arm** to hit the block." |
| **unseen** | unseen 模板池 | unseen 描述词池 | "Grab **the hammer with claw and smooth head** and hit the block." |

两条路径**完全隔离**，不存在交叉泄漏。

#### 不同 episode 间的异同

| 维度 | 同任务不同 episode 之间 |
|------|------------------------|
| 指令模板池（seen/unseen） | **相同** — 任务级预定义，固定不变 |
| 物体描述词池（seen/unseen） | **相同** — 物体级预定义，固定不变 |
| 场景参数（用哪只臂、哪个物体变体等） | **可能不同** — 取决于 `scene_info.json` 中该 episode 的配置 |
| 最终生成的 100 条具体指令 | **不同** — 因为描述词是随机抽取的 + 场景参数可能不同 |

因此，**同一个任务反复采集数据，生成的具体指令可能不同，但 seen/unseen 的边界始终保持隔离**：
- seen 指令只会使用 seen 模板 + seen 物体描述词
- unseen 指令只会使用 unseen 模板 + unseen 物体描述词
- 唯一的特殊情况：如果某物体没有 unseen 描述词，unseen 指令会回退使用 seen 描述词（不影响评测公平性，因为方向是"unseen 借用 seen"，训练集不会看到 unseen 内容）

---

## video 目录 — 头部相机视频

每个 episode 对应一个 MP4 视频文件，录制的是该 episode 执行过程中头部相机的画面。可直接用视频播放器查看。

---

## 辅助文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `_traj_data/` | 数据采集过程中记录的细粒度动作轨迹（中间产物） |
| `scene_info.json` | 场景配置信息，包含物体位置、机器人初始状态等 |
| `seed.txt` | 数据采集使用的随机种子列表 |

---

## 可视化 HDF5 文件

本目录已提供 `visualize_hdf5.py` 脚本，可生成可视化结果：

```bash
# 依赖安装（如尚未安装）
pip install h5py opencv-python-headless matplotlib numpy

# 基础可视化：生成末端执行器轨迹图、关节动作图、多相机帧拼图
python data/visualize_hdf5.py data/beat_block_hammer/demo_clean/data/episode0.hdf5

# 同时导出相机视频
python data/visualize_hdf5.py data/beat_block_hammer/demo_clean/data/episode0.hdf5 --video

# 指定输出目录和相机
python data/visualize_hdf5.py data/beat_block_hammer/demo_clean/data/episode0.hdf5 \
    --video --camera front_camera --output_dir ./my_vis
```

生成的可视化包括：
- `endpose_trajectory.png` — 左右臂 XYZ 轨迹 + 夹爪状态 + 3D 轨迹图
- `joint_action.png` — 左右臂关节角度曲线 + 14维动作向量热力图
- `camera_frames.png` — 4 个相机视角的均匀采样帧拼图
- `{camera}_replay.mp4` — 指定相机的回放视频（需加 `--video` 参数）

---

# ② ALOHA 中间 HDF5 格式

## 为什么需要这一步？

RoboTwin 原始 HDF5 包含了非常丰富的信息（4 相机内外参、末端位姿、关节角度、JPEG 压缩图像等），但训练模型并不需要这么多。`preprocess_aloha.py` 会将原始数据精简并统一为 OpenVLA-OFT 能理解的格式：

- 图像从 JPEG bit stream **解码并 resize 为 256×256** 的 RGB 数组
- 相机名称**重映射**为 ALOHA 命名规范
- **保留 `joint_action/vector`** 作为 `action`（14维关节角度绝对值）
- **丢弃 `endpose/`**（末端位姿不参与训练）
- 计算 `relative_action`（相邻帧的关节角度差分）— ⚠️ 此字段在当前流程中**未被下游使用**，属于冗余计算
- 将语言指令从 JSON 文件**写入 HDF5**
- 将数据随机**划分为 train/val**

## 转换命令

```bash
python policy/openvla-oft/preprocess_aloha.py \
    --dataset_path data/beat_block_hammer/demo_clean/data \
    --out_base_dir data/beat_block_hammer/processed_openvla \
    --instruction_dir data/beat_block_hammer/demo_clean/instructions \
    --percent_val 0.05
```

## 输出目录结构

```
data/beat_block_hammer/processed_openvla/
├── train/
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   └── ...                (约 95% 的 episodes)
└── val/
    ├── episode_0.hdf5
    └── ...                (约 5% 的 episodes)
```

## ALOHA 中间 HDF5 的字段

| 字段 | Shape | 类型 | 物理含义 | 绝对/相对 | 说明 |
|------|-------|------|---------|-----------|------|
| `head_camera_image` | (T, 256, 256, 3) | uint8 | — | — | 头部相机 RGB 图像（来自 head_camera） |
| `left_wrist_image` | (T, 256, 256, 3) | uint8 | — | — | 左手腕相机（来自 left_camera） |
| `right_wrist_image` | (T, 256, 256, 3) | uint8 | — | — | 右手腕相机（来自 right_camera） |
| `low_cam_image` | (T, 256, 256, 3) | uint8 | — | — | 低位相机（来自 front_camera） |
| `action` | (T, 14) | float64 | **关节角度** | **绝对** | ✅ 原样拷贝自 `joint_action/vector`，下一步 RLDS 使用此字段 |
| `relative_action` | (T, 14) | float64 | **关节角度差分** | **相对** | ⚠️ `action[t+1] - action[t]`，当前流程**未被下游使用** |
| `seen` | (N,) | string | — | — | seen 语言指令列表 |
| `unseen` | (N,) | string | — | — | unseen 语言指令列表 |

> **关于 `relative_action` 的冗余**：`preprocess_aloha.py` 会计算并存储 `relative_action`（关节角度的帧间差分），但在后续 `robotwin_builder.py` 转换为 RLDS 时，**只读取了 `action`（绝对值），完全没有使用 `relative_action`**。训练时 `configs.py` 也将所有 14 维标记为绝对值（`absolute_action_mask = [True]*14`）。因此在当前 RoboTwin → OpenVLA-OFT 流程中，`relative_action` 的计算是冗余的。

### 相机名称映射关系

| RoboTwin 原始名称 | ALOHA 中间格式名称 | OpenVLA-OFT 中的角色 |
|---|---|---|
| head_camera | `head_camera_image` | primary（主相机） |
| left_camera | `left_wrist_image` | left_wrist（左手腕） |
| right_camera | `right_wrist_image` | right_wrist（右手腕） |
| front_camera | `low_cam_image` | secondary（辅助相机） |

### 与原始格式的主要区别

| 对比项 | ① 原始 HDF5 | ② ALOHA 中间 HDF5 |
|--------|------------|-------------------|
| 图像格式 | JPEG bit stream（变长字节） | 解码后的 RGB 数组 (256×256×3) |
| 图像尺寸 | 原始分辨率（如 320×240） | 统一为 256×256 |
| 相机内外参 | 有（cam2world_gl, extrinsic_cv, intrinsic_cv） | **丢弃**（训练不需要） |
| 末端位姿 | 有（endpose/，笛卡尔空间绝对值） | **丢弃**（训练不需要） |
| 关节动作 | 绝对关节角度（joint_action/vector） | 绝对关节角度（action）+ 关节角度差分（relative_action，⚠️ 未被使用） |
| 语言指令 | 单独的 JSON 文件 | 直接嵌入 HDF5 |
| 数据划分 | 无 | 已划分 train/val |

---

# ③ RLDS (TFRecord) 格式

## 什么是 RLDS？

RLDS（Reinforcement Learning Datasets）是 Google 提出的**强化学习数据集标准**，基于 TensorFlow Datasets (tfds) 框架。它将数据存储为 TFRecord 格式，这是 TensorFlow 的高效二进制数据格式。

**OpenVLA-OFT 训练脚本直接读取 RLDS 格式的数据**，所以这是训练前的最后一步转换。

## 转换命令

```bash
python policy/openvla-oft/datasets/robotwin_builder.py \
    --task_name beat_block_hammer \
    --data_dir data/beat_block_hammer/processed_openvla
```

## 输出位置

```
~/tensorflow_datasets/aloha_beat_block_hammer/1.0.0/
├── aloha_beat_block_hammer-train.tfrecord-00000-of-00001   # 训练集（所有 episode 打包在一个分片中）
├── aloha_beat_block_hammer-val.tfrecord-00000-of-00001     # 验证集
├── dataset_info.json                                        # 数据集元信息（条目数、大小等）
└── features.json                                            # 字段结构定义
```

> **关于分片命名：** `00000-of-00001` 表示"第 0 个分片，共 1 个分片"，不是 episode 编号。数据量大时 tfds 会自动拆成多个分片文件。

## RLDS 的数据结构

RLDS 规定了一个**标准骨架**：每个数据集由若干 episode 组成，每个 episode 由若干 step 组成。

### 每个 step 的标准字段（RLDS 通用）

以下 7 个字段是 **所有 RLDS 数据集都必须包含的**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `observation` | Dict | 观测信息（具体内容由各数据集自定义） |
| `action` | Tensor | 动作（具体维度由各数据集自定义） |
| `reward` | Scalar float32 | 奖励信号 |
| `discount` | Scalar float32 | 折扣因子 |
| `is_first` | Scalar bool | 是否为 episode 的第一步 |
| `is_last` | Scalar bool | 是否为 episode 的最后一步 |
| `is_terminal` | Scalar bool | 是否为终止状态 |

### RoboTwin 自定义的字段

在上述骨架之上，RoboTwin 在 `observation` 中放入了自己的数据：

| 字段 | Shape | 类型 | 物理含义 | 绝对/相对 | 说明 |
|------|-------|------|---------|-----------|------|
| `observation/image` | (256, 256, 3) | uint8 | — | — | 主相机图像（head_camera） |
| `observation/left_wrist_image` | (256, 256, 3) | uint8 | — | — | 左手腕相机 |
| `observation/right_wrist_image` | (256, 256, 3) | uint8 | — | — | 右手腕相机 |
| `observation/low_cam_image` | (256, 256, 3) | uint8 | — | — | 低位相机（front_camera） |
| `observation/state` | (14,) | float32 | **关节角度** | **绝对** | = ② 中 `action[t]`，即**当前帧的关节位置** |
| `action` | (14,) | float32 | **关节角度** | **绝对** | = ② 中 `action[t+1]`，即**下一帧的目标关节位置** |
| `language_instruction` | Sequence[Text] | — | — | — | 语言指令列表（来自 seen） |

> **state 与 action 的关系**：在 RLDS 中，`state[t]` 存的是 t 时刻机器人各关节的**实际位置**（绝对角度），`action[t]` 存的是 t 时刻模型应该输出的**目标位置**（也是绝对角度，对应原始数据中 t+1 帧的关节角度）。两者都是关节角度绝对值，不是末端位姿，也不是增量/差分。

另外每个 episode 还有元数据：
| 字段 | 类型 | 说明 |
|------|------|------|
| `episode_metadata/file_path` | Text | 来源 HDF5 文件路径 |

### 与 ALOHA 中间格式的主要区别

| 对比项 | ② ALOHA 中间 HDF5 | ③ RLDS TFRecord |
|--------|-------------------|-----------------|
| 文件格式 | HDF5（每个 episode 一个文件） | TFRecord（所有 episode 打包） |
| 时间步处理 | 保留完整 T 步 | T-1 步（第一帧作为初始观测，动作从第二帧开始） |
| action 含义 | 关节角度**绝对值** `action[0..T-1]` | 关节角度**绝对值** `action[1..T-1]`（即下一帧的目标关节位置） |
| relative_action | 有（关节角度差分）— ⚠️ 未被使用 | **无**（转换时直接忽略） |
| state | 无单独 state 字段 | 关节角度**绝对值**，`state[t] = action[t]`（当前帧的关节位置） |
| 语言指令 | seen[] + unseen[] | 仅 seen[]（训练只用 seen） |
| 额外字段 | 无 | reward, discount, is_first, is_last, is_terminal |

## 注册数据集

生成 TFRecord 后，还需要在 OpenVLA-OFT 的三个配置文件中**注册**你的数据集，训练脚本才能找到并正确读取它：

| 配置文件 | 位置 | 告诉训练脚本什么 |
|---------|------|----------------|
| `configs.py` | `prismatic/vla/datasets/rlds/oxe/` | 数据集有哪些相机、状态和动作的字段名与维度 |
| `transforms.py` | 同上 | 对数据做什么预处理变换 |
| `mixtures.py` | 同上 | 训练时使用哪些数据集、各自的采样权重 |

这三个文件不需要运行，**训练脚本启动时会自动读取**。注册的方法就是在已有 dict 末尾追加几行，照搬其他任务的格式即可。

---

## OpenVLA-OFT 的动作编码方式

OpenVLA-OFT 是一个通用框架，支持多种动作编码。注册数据集时在 `configs.py` 中选择 `action_encoding`，决定了模型的输出含义：

| 编码类型 | 维度 | 含义 | 绝对/相对 | 适用场景 |
|---------|------|------|-----------|---------|
| `EEF_POS` | 7 | 末端 XYZ 增量(3) + RPY 增量(3) + 夹爪绝对值(1) | 前 6 维**相对**，夹爪**绝对** | 单臂机器人（Bridge 等） |
| `EEF_R6` | 10 | 末端 XYZ 增量(3) + 6D旋转增量(6) + 夹爪绝对值(1) | 前 9 维**相对**，夹爪**绝对** | 单臂机器人（需要更精确旋转） |
| `JOINT_POS_BIMANUAL` | **14** | **双臂关节角度(2×6) + 双夹爪(2×1)** | **全部绝对** | **RoboTwin 双臂** ← 当前使用 |

**RoboTwin 使用 `JOINT_POS_BIMANUAL`**，这意味着：

1. 训练出的模型预测 **14 维绝对关节角度**，而非末端位姿增量
2. 推理时，模型输出经反归一化后，直接作为目标关节角度发送给机器人控制器
3. `materialize.py` 中对应的配置为 `absolute_action_mask = [True] * 14`（全部 14 维都是绝对值），`action_normalization_mask = [True] * 14`（全部 14 维都参与归一化）

```
推理时的执行流程：

观测图像 + 语言指令 → OpenVLA-OFT 模型 → 14维归一化输出
                                          ↓ 反归一化
                                  14维绝对关节角度
                                          ↓
                            直接发送给机器人的 14 个关节控制器
                          [左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
```
