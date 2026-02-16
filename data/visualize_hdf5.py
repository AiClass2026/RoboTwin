"""
RoboTwin HDF5 数据可视化脚本
用法: python visualize_hdf5.py <hdf5_file_path> [--output_dir OUTPUT_DIR]

功能:
1. 打印 HDF5 文件的完整层级结构
2. 绘制机器人末端执行器轨迹 (endpose)
3. 绘制关节动作序列 (joint_action)
4. 提取并拼接多相机 RGB 图像帧
5. 生成多相机视频 (可选)
"""

import argparse
import os
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import numpy as np


def print_hdf5_structure(f, indent=0):
    """递归打印 HDF5 文件的完整层级结构"""
    for key in f.keys():
        item = f[key]
        prefix = "  " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}[Group] {key}/")
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}[Dataset] {key}  shape={item.shape}  dtype={item.dtype}")


def decode_image(image_bytes):
    """将 HDF5 中存储的 JPEG bit stream 解码为 numpy 图像"""
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


def visualize_endpose(f, output_dir):
    """可视化末端执行器位姿轨迹 (3D 位置 + gripper 状态)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("End-Effector Pose Trajectory", fontsize=14)

    # 左臂末端位姿
    left_ep = f['endpose/left_endpose'][:]  # (T, 7): xyz + quat
    right_ep = f['endpose/right_endpose'][:]
    left_grip = f['endpose/left_gripper'][:]
    right_grip = f['endpose/right_gripper'][:]
    T = left_ep.shape[0]
    timesteps = np.arange(T)

    # 左臂 XYZ
    ax = axes[0, 0]
    ax.plot(timesteps, left_ep[:, 0], label='X')
    ax.plot(timesteps, left_ep[:, 1], label='Y')
    ax.plot(timesteps, left_ep[:, 2], label='Z')
    ax.set_title("Left Arm Position (XYZ)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右臂 XYZ
    ax = axes[0, 1]
    ax.plot(timesteps, right_ep[:, 0], label='X')
    ax.plot(timesteps, right_ep[:, 1], label='Y')
    ax.plot(timesteps, right_ep[:, 2], label='Z')
    ax.set_title("Right Arm Position (XYZ)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gripper 状态
    ax = axes[1, 0]
    ax.plot(timesteps, left_grip, label='Left Gripper', color='blue')
    ax.plot(timesteps, right_grip, label='Right Gripper', color='red')
    ax.set_title("Gripper State")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gripper Open/Close")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3D 轨迹
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot(left_ep[:, 0], left_ep[:, 1], left_ep[:, 2], 'b-', label='Left Arm', alpha=0.7)
    ax.plot(right_ep[:, 0], right_ep[:, 1], right_ep[:, 2], 'r-', label='Right Arm', alpha=0.7)
    ax.scatter(*left_ep[0, :3], c='blue', marker='o', s=50, label='Left Start')
    ax.scatter(*left_ep[-1, :3], c='blue', marker='x', s=50, label='Left End')
    ax.scatter(*right_ep[0, :3], c='red', marker='o', s=50, label='Right Start')
    ax.scatter(*right_ep[-1, :3], c='red', marker='x', s=50, label='Right End')
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(fontsize=7)
    # 删除原来axes[1,1]的占位
    axes[1, 1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "endpose_trajectory.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> 保存末端执行器轨迹图: {path}")


def visualize_joint_action(f, output_dir):
    """可视化关节动作序列"""
    vec = f['joint_action/vector'][:]  # (T, 14)
    left_arm = f['joint_action/left_arm'][:]  # (T, 6)
    right_arm = f['joint_action/right_arm'][:]  # (T, 6)
    left_grip = f['joint_action/left_gripper'][:]
    right_grip = f['joint_action/right_gripper'][:]
    T = vec.shape[0]
    timesteps = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Joint Action Sequence", fontsize=14)

    # 左臂 6 个关节
    ax = axes[0, 0]
    for j in range(6):
        ax.plot(timesteps, left_arm[:, j], label=f'J{j}')
    ax.set_title("Left Arm Joint Angles")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Joint Value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 右臂 6 个关节
    ax = axes[0, 1]
    for j in range(6):
        ax.plot(timesteps, right_arm[:, j], label=f'J{j}')
    ax.set_title("Right Arm Joint Angles")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Joint Value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Gripper
    ax = axes[1, 0]
    ax.plot(timesteps, left_grip, label='Left Gripper', color='blue')
    ax.plot(timesteps, right_grip, label='Right Gripper', color='red')
    ax.set_title("Gripper Actions")
    ax.set_xlabel("Timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # joint_action/vector 热力图
    ax = axes[1, 1]
    im = ax.imshow(vec.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title("Joint Action Vector Heatmap (14-dim)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Dimension")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(output_dir, "joint_action.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> 保存关节动作图: {path}")


def visualize_camera_frames(f, output_dir, sample_indices=None):
    """提取并可视化多相机图像帧"""
    cameras = []
    for cam_name in ['head_camera', 'front_camera', 'left_camera', 'right_camera']:
        path = f'observation/{cam_name}/rgb'
        if path in f:
            cameras.append(cam_name)

    if not cameras:
        print("  [警告] 未找到相机 RGB 数据")
        return

    T = f[f'observation/{cameras[0]}/rgb'].shape[0]
    if sample_indices is None:
        # 均匀采样 8 帧
        sample_indices = np.linspace(0, T - 1, min(8, T), dtype=int)

    n_frames = len(sample_indices)
    n_cams = len(cameras)

    fig, axes = plt.subplots(n_cams, n_frames, figsize=(n_frames * 2.5, n_cams * 2.5))
    if n_cams == 1:
        axes = axes[np.newaxis, :]
    if n_frames == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Multi-Camera RGB Frames (sampled)", fontsize=14)

    for row, cam_name in enumerate(cameras):
        rgb_ds = f[f'observation/{cam_name}/rgb']
        for col, idx in enumerate(sample_indices):
            img = decode_image(rgb_ds[idx])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f't={idx}', fontsize=8)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(cam_name, fontsize=9, rotation=90, labelpad=40)

    plt.tight_layout()
    path = os.path.join(output_dir, "camera_frames.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> 保存相机帧图: {path}")


def export_camera_video(f, output_dir, camera='head_camera', fps=15):
    """将某个相机的所有帧导出为视频"""
    rgb_path = f'observation/{camera}/rgb'
    if rgb_path not in f:
        print(f"  [警告] 未找到 {rgb_path}")
        return

    rgb_ds = f[rgb_path]
    T = rgb_ds.shape[0]

    # 先解码第一帧获取尺寸
    first_frame = decode_image(rgb_ds[0])
    h, w = first_frame.shape[:2]

    video_path = os.path.join(output_dir, f"{camera}_replay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for i in range(T):
        frame = decode_image(rgb_ds[i])
        writer.write(frame)

    writer.release()
    print(f"  -> 保存相机视频: {video_path} ({T} frames, {w}x{h}, {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="RoboTwin HDF5 数据可视化")
    parser.add_argument("hdf5_path", help="HDF5 文件路径")
    parser.add_argument("--output_dir", default=None, help="输出目录 (默认: 与 HDF5 文件同目录下的 vis/)")
    parser.add_argument("--video", action="store_true", help="是否同时导出相机视频")
    parser.add_argument("--camera", default="head_camera", help="导出视频时使用的相机名称")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_path):
        print(f"错误: 文件不存在 {args.hdf5_path}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.hdf5_path), "..", "vis")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"打开 HDF5 文件: {args.hdf5_path}")
    f = h5py.File(args.hdf5_path, 'r')

    print("\n" + "=" * 60)
    print("HDF5 文件层级结构:")
    print("=" * 60)
    print_hdf5_structure(f)

    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    visualize_endpose(f, args.output_dir)
    visualize_joint_action(f, args.output_dir)
    visualize_camera_frames(f, args.output_dir)

    if args.video:
        print("\n导出相机视频...")
        export_camera_video(f, args.output_dir, camera=args.camera)

    f.close()
    print(f"\n全部可视化结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
