# RoboTwin 数据目录说明

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

## HDF5 文件格式说明

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

### RoboTwin 中 HDF5 的层级结构

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

#### 1. endpose — 末端执行器位姿

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `left_endpose` | (T, 7) | float64 | 左臂末端位姿：前3维为 XYZ 坐标（米），后4维为四元数姿态 (w, x, y, z) |
| `left_gripper` | (T,) | float64 | 左夹爪开合状态：1.0 = 完全张开，0.0 = 完全闭合 |
| `right_endpose` | (T, 7) | float64 | 右臂末端位姿，格式同 left_endpose |
| `right_gripper` | (T,) | float64 | 右夹爪开合状态 |

#### 2. joint_action — 关节空间动作

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `left_arm` | (T, 6) | float64 | 左臂 6 个关节的角度值（弧度） |
| `right_arm` | (T, 6) | float64 | 右臂 6 个关节的角度值（弧度） |
| `left_gripper` | (T,) | float64 | 左夹爪 |
| `right_gripper` | (T,) | float64 | 右夹爪 |
| `vector` | (T, 14) | float64 | 14维拼接向量：`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]` |

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
