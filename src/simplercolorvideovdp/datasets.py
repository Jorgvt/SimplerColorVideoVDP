"""PyTorch Dataset and DataLoader streaming utilities for GAIM-240 and video quality evaluation.

Provides streaming video decoding via FFmpeg raw pipe to eliminate memory bottlenecks
when loading high-frame-rate (240Hz) and high-resolution video pairs on demand.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Default scenes in GAIM240
ALL_SCENES: List[str] = [
    "attic",
    "bistro_exterior",
    "bistro_interior",
    "classroom",
    "landscape",
    "marbles",
    "pink_room",
    "subway",
    "zeroday",
]

# Benchmark validation scenes matching metric_calibration convention
DEFAULT_VAL_SCENES: List[str] = ["subway", "zeroday"]

DEFAULT_CSV_PATH = "/media/disk/vista/BBDD_video_image/GAIM240/jod.csv"
DEFAULT_DATA_DIR = "/media/disk/vista/BBDD_video_image/GAIM240"


def _get_ffmpeg_cmd() -> str:
    """Finds or resolves the ffmpeg binary executable."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_video_ffmpeg(
    video_path: str,
    num_frames: Optional[int] = None,
    start_frame: int = 0,
    fps: float = 240.0,
    scale: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Reads RGB video frames using high-performance ffmpeg raw pipe.

    Args:
        video_path: Path to MP4 video file.
        num_frames: Number of frames to extract (None or <= 0 extracts all frames).
        start_frame: Frame offset to start extraction.
        fps: Native frame rate (default: 240.0).
        scale: (height, width) to rescale frames, or None for native resolution.

    Returns:
        np.ndarray of shape (num_frames, H, W, 3) normalized to float32 in [0, 1].
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg_bin = _get_ffmpeg_cmd()
    cmd = [ffmpeg_bin, "-v", "error"]

    if start_frame > 0:
        cmd.extend(["-ss", f"{start_frame / fps:.4f}"])
    cmd.extend(["-i", video_path])
    if scale is not None:
        cmd.extend(["-vf", f"scale={scale[1]}:{scale[0]}"])
    if num_frames is not None and num_frames > 0:
        cmd.extend(["-vframes", str(num_frames)])
    cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])

    pipe = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
    )
    raw_bytes, stderr = pipe.communicate()

    if pipe.returncode != 0 or len(raw_bytes) == 0:
        raise RuntimeError(
            f"FFmpeg failed for {video_path}: {stderr.decode('utf-8', errors='ignore')}"
        )

    h = scale[0] if scale is not None else 720
    w = scale[1] if scale is not None else 1280
    frames = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((-1, h, w, 3))

    if num_frames is not None and num_frames > 0:
        if len(frames) < num_frames:
            pad_count = num_frames - len(frames)
            last_frame = (
                frames[-1:]
                if len(frames) > 0
                else np.zeros((1, h, w, 3), dtype=np.uint8)
            )
            frames = np.concatenate(
                [frames, np.repeat(last_frame, pad_count, axis=0)], axis=0
            )
        frames = frames[:num_frames]

    return (frames / 255.0).astype(np.float32)


def generate_synthetic_video(
    video_name: str,
    num_frames: Optional[int],
    height: int,
    width: int,
    is_ref: bool = False,
) -> np.ndarray:
    """Generates deterministic synthetic video stimulus for testing and validation."""
    n_frames = num_frames if (num_frames is not None and num_frames > 0) else 240
    h = height if (height is not None and height > 0) else 720
    w = width if (width is not None and width > 0) else 1280

    rng = np.random.RandomState(abs(hash(video_name)) % (2**31))
    t_grid = np.linspace(0, 1, n_frames)[:, None, None, None]
    y_grid = np.linspace(0, 4 * np.pi, h)[None, :, None, None]
    x_grid = np.linspace(0, 4 * np.pi, w)[None, None, :, None]

    base = 0.5 + 0.3 * np.sin(x_grid - 10.0 * t_grid + y_grid)
    base = np.repeat(base, 3, axis=-1)

    if not is_ref:
        noise = rng.normal(0, 0.05, size=base.shape)
        frames = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
    else:
        frames = np.clip(base, 0.0, 1.0).astype(np.float32)

    return frames


class GAIM240TorchDataset(Dataset):
    """PyTorch Dataset for GAIM-240 Video Quality Assessment with on-demand streaming decoding.

    Each item is decoded dynamically on demand, drastically reducing RAM footprint
    and allowing multi-worker prefetching.
    """

    def __init__(
        self,
        csv_path: str = DEFAULT_CSV_PATH,
        data_dir: str = DEFAULT_DATA_DIR,
        mode: str = "all",  # 'all', 'train', 'val'
        val_scenes: Optional[List[str]] = None,
        num_frames: Optional[int] = 60,
        start_frame: int = 0,
        scale: Optional[Tuple[int, int]] = (360, 640),
        fps: float = 240.0,
        as_torch_tensor: bool = True,
        tensor_layout: str = "TCHW",  # 'TCHW' or 'THWC'
        allow_mock: bool = False,
    ):
        super().__init__()
        self.csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
        self.data_dir = os.path.expanduser(data_dir) if data_dir else data_dir
        self.mode = mode
        self.val_scenes = (
            val_scenes if val_scenes is not None else DEFAULT_VAL_SCENES
        )
        self.num_frames = (
            num_frames if (num_frames is not None and num_frames > 0) else None
        )
        self.start_frame = start_frame
        self.scale = (
            scale
            if (scale is not None and scale[0] > 0 and scale[1] > 0)
            else None
        )
        self.fps = fps
        self.as_torch_tensor = as_torch_tensor
        self.tensor_layout = tensor_layout.upper()
        self.allow_mock = allow_mock

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"GAIM-240 JOD metadata not found at {self.csv_path}"
            )

        df = pd.read_csv(
            self.csv_path,
            skipinitialspace=True,
            dtype={"distortion_level": str},
        )

        # Exclude reference identical comparisons if present
        if "distortion_level" in df.columns:
            df = df[df["distortion_level"] != "Ref"]
        if "dist_vid_path" in df.columns and "ref_vid_path" in df.columns:
            df = df[df["dist_vid_path"] != df["ref_vid_path"]]

        # Train / Val scene split
        if self.mode == "train":
            df = df[~df["scene_name"].isin(self.val_scenes)]
        elif self.mode == "val":
            df = df[df["scene_name"].isin(self.val_scenes)]

        self.table = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        row = self.table.iloc[idx]
        return {
            "index": idx,
            "dist_vid_path": str(row["dist_vid_path"]),
            "ref_vid_path": str(row["ref_vid_path"]),
            "scene_name": str(row["scene_name"]),
            "distortion_type": str(row["distortion_type"]),
            "distortion_level": str(row["distortion_level"]),
            "jod": float(row["jod"]),
        }

    def _load_video(self, video_name: str, is_ref: bool = False) -> np.ndarray:
        full_path = os.path.join(self.data_dir, video_name)

        if os.path.exists(full_path):
            return read_video_ffmpeg(
                full_path,
                num_frames=self.num_frames,
                start_frame=self.start_frame,
                fps=self.fps,
                scale=self.scale,
            )
        elif self.allow_mock:
            h = self.scale[0] if self.scale is not None else 360
            w = self.scale[1] if self.scale is not None else 640
            return generate_synthetic_video(
                video_name=video_name,
                num_frames=self.num_frames,
                height=h,
                width=w,
                is_ref=is_ref,
            )
        else:
            raise FileNotFoundError(
                f"Video file not found at {full_path}. Ensure GAIM240 videos are placed in '{self.data_dir}' "
                "or pass allow_mock=True for synthetic benchmarking."
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a pair of (ref_video, dist_video) on demand."""
        meta = self.get_sample_metadata(idx)
        ref_np = self._load_video(meta["ref_vid_path"], is_ref=True)  # (T, H, W, C)
        dist_np = self._load_video(meta["dist_vid_path"], is_ref=False)  # (T, H, W, C)

        if self.as_torch_tensor:
            ref_tensor = torch.from_numpy(ref_np)
            dist_tensor = torch.from_numpy(dist_np)
            if self.tensor_layout == "TCHW":
                ref_tensor = ref_tensor.permute(0, 3, 1, 2).contiguous()
                dist_tensor = dist_tensor.permute(0, 3, 1, 2).contiguous()
            jod_tensor = torch.tensor(meta["jod"], dtype=torch.float32)
            return {
                "ref": ref_tensor,
                "dist": dist_tensor,
                "jod": jod_tensor,
                "meta": meta,
            }
        else:
            if self.tensor_layout == "TCHW":
                ref_np = np.transpose(ref_np, (0, 3, 1, 2))
                dist_np = np.transpose(dist_np, (0, 3, 1, 2))
            return {
                "ref": ref_np,
                "dist": dist_np,
                "jod": np.float32(meta["jod"]),
                "meta": meta,
            }


def create_gaim240_dataloader(
    csv_path: str = DEFAULT_CSV_PATH,
    data_dir: str = DEFAULT_DATA_DIR,
    mode: str = "train",
    val_scenes: Optional[List[str]] = None,
    num_frames: Optional[int] = 60,
    start_frame: int = 0,
    scale: Optional[Tuple[int, int]] = (360, 640),
    fps: float = 240.0,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = False,
    tensor_layout: str = "TCHW",
    allow_mock: bool = False,
) -> DataLoader:
    """Factory function creating a configured PyTorch DataLoader for GAIM-240 video streaming."""
    dataset = GAIM240TorchDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        mode=mode,
        val_scenes=val_scenes,
        num_frames=num_frames,
        start_frame=start_frame,
        scale=scale,
        fps=fps,
        as_torch_tensor=True,
        tensor_layout=tensor_layout,
        allow_mock=allow_mock,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return loader
