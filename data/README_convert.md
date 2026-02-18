# RoboTwin 数据转换说明

本文档介绍 RoboTwin 数据从采集到训练所涉及的 **三种数据格式**，以及它们之间的转换关系。

> 关于 RoboTwin 原始数据的格式说明和可视化工具，请参阅 [README_rawdata.md](README_rawdata.md)。

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

| 格式 | 文件类型 | 存放位置 | 用途 |
|------|---------|---------|------|
| ① RoboTwin 原始数据 | HDF5 (.hdf5) | `data/{task}/{config}/data/` | 仿真采集的原始数据，保留所有信息（详见 [README_rawdata.md](README_rawdata.md)） |
| ② ALOHA 中间格式 | HDF5 (.hdf5) | `data/{task}/processed_openvla/` | 预处理后的中间格式，统一了图像尺寸和字段命名 |
| ③ RLDS (TFRecord) | TFRecord | `~/tensorflow_datasets/aloha_{task}/` | 训练用的最终格式，OpenVLA-OFT 直接读取 |


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
