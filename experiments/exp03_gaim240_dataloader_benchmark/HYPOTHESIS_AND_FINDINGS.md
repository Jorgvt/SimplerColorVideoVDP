# Experiment 03: GAIM-240 Streaming DataLoader Performance Benchmark

## 1. Hypothesis & Objective

### Objective
Accurately profile and benchmark the on-demand video data loading pipeline on the GAIM-240 Video Quality Assessment dataset ($240\text{ fps}$ native frame rate, $1280\times 720$ resolution). We evaluate:
1. **Per-epoch processing time** across different numbers of DataLoader worker processes.
2. **Streaming Throughput** (video pairs/second, frames/second, MB/second).
3. **Batch latency distributions** (mean, median, 95th percentile).
4. **Memory footprint** (Peak RSS vs. monolithic in-memory preloading).
5. **Scaling with resolution** (downsampled $360\times 640$ vs. native full-size $720\times 1280$).

### Hypothesis
1. **Memory Efficiency**: Streaming decoded frames via an FFmpeg pipe into PyTorch Tensors on-the-fly will maintain a constant, minimal RAM footprint (< $2\text{ GB}$) regardless of dataset size, avoiding the massive ~212.6 GB RAM requirement of monolithic preloading at full resolution.
2. **Multi-worker Parallel Prefetching**: Single-worker decoding (`num_workers=0`) creates an I/O and decoding bottleneck. Scaling to `num_workers=8` or `16` will enable multi-process concurrency, hiding decoding latency behind background prefetching queues and drastically accelerating full epochs.

---

## 2. Experimental Setup

- **Hardware**: Server `deep` (Debian Linux, 40 CPU cores, 251 GB RAM).
- **Dataset**: GAIM-240 (`/media/disk/vista/BBDD_video_image/GAIM240/`)
  - **Training Split**: 7 scenes (`attic`, `bistro_exterior`, `bistro_interior`, `classroom`, `landscape`, `marbles`, `pink_room`) $\to$ $168$ video pairs.
  - **Validation Split**: 2 scenes (`subway`, `zeroday`) $\to$ $48$ video pairs.
  - **Total Dataset**: $216$ comparisons across 8 distortion types and 3 severity levels.
- **DataLoader**: PyTorch `DataLoader` wrapping `GAIM240TorchDataset` with on-demand raw FFmpeg pipe decoding (`prefetch_factor=2`, `persistent_workers=True`).

---

## 3. Benchmark Results & Performance Metrics

### Part A: Downsampled Videos ($360\times 640$, 60 frames / 0.25s)

| Workers | Batch Size | Total Epoch Time (168 pairs) | Throughput (pairs/s) | Frames/sec | Streaming Bandwidth | Mean Latency | Median Latency | Peak RAM (RSS) | Memory Savings |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1 | **29 min 26 s** (1765.8 s) | 0.10 pairs/s | 11.4 fps | 30.1 MB/s | 13.57 ms | 0.03 ms | 1.93 GB | **27.6×** |
| **2** | 1 | **18 min 00 s** (1080.2 s) | 0.16 pairs/s | 18.7 fps | 49.2 MB/s | 23.30 ms | 0.08 ms | 1.93 GB | **27.6×** |
| **4** | 1 | **9 min 16 s** (555.6 s) | 0.30 pairs/s | 36.3 fps | 95.7 MB/s | 24.62 ms | 0.03 ms | 1.93 GB | **27.6×** |
| **8** | 1 | **5 min 36 s** (335.7 s) | **0.50 pairs/s** | **60.1 fps** | **158.4 MB/s** | 31.00 ms | 0.03 ms | 1.93 GB | **27.6×** |

---

### Part B: Full-Size Native Videos ($1280\times 720$, 60 frames / 0.25s)

| Workers | Batch Size | Total Epoch Time (168 pairs) | Throughput (pairs/s) | Frames/sec | Streaming Bandwidth | Mean Latency | Median Latency | Peak RAM (RSS) | Memory Savings |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 1 | **32 min 27 s** (1947.3 s) | 0.09 pairs/s | 10.4 fps | 109.2 MB/s | 101.4 ms | 0.02 ms | 0.61 GB | **347.6×** |
| **8** | 1 | **20 min 29 s** (1229.2 s) | 0.14 pairs/s | 16.4 fps | 173.0 MB/s | 117.0 ms | 0.02 ms | 0.61 GB | **347.6×** |
| **16** | 1 | **14 min 57 s** (897.1 s) | **0.19 pairs/s** | **22.5 fps** | **237.0 MB/s** | 146.1 ms | 0.04 ms | 0.61 GB | **347.6×** |

---

## 4. Key Findings & Analysis

1. **Massive Memory Savings**:
   - Monolithic preloading of full-resolution 720p clips for the 168 training pairs would require **212.6 GB of RAM**, which would exhaust system memory and cause swap thrashing.
   - On-demand streaming constrained peak RAM usage to **~611 MB** (a **$347.6\times$ reduction**).

2. **Scaling with Worker Concurrency**:
   - **Downsampled ($360\times 640$)**: Epoch time scales from **29.4 min (0 workers)** down to **5.6 min (8 workers)** ($5.3\times$ speedup).
   - **Full-Size ($1280\times 720$)**: Scaling from 4 to 16 workers reduces epoch time from **32.5 min** down to **14.9 min** (streaming bandwidth reaches **237.0 MB/s**).

3. **Background Prefetching Efficiency**:
   - Because `prefetch_factor=2` and `persistent_workers=True` keep the worker pool active, the median wait latency for the training loop once a batch is requested is **< 0.1 ms**. 
   - When paired with GPU computation during metric calibration, data decoding happens completely asynchronously in the background.

---

## 5. Recommendations for Model Calibration Workflow

- **Phase 1: Rapid Parameter Calibration & Optimization (50–100 epochs)**
  - Use $360\times 640$ spatial resolution with $60\text{ frames}$ ($0.25\text{s}$) and `num_workers = 8`.
  - Full epoch runtime: **~5.5 minutes**, enabling 50 epochs of end-to-end optimization in ~4.5 hours.
- **Phase 2: Fine-Tuning & Final Model Evaluation**
  - Use native $1280\times 720$ resolution with `num_workers = 16`.
  - Full validation epoch (48 pairs): **~4.2 minutes**.
