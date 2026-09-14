# Experiments & Quantitative Verification

This document summarizes the quantitative validation experiments designed to verify the physical correctness, mathematical consistency, and computational performance of `evsim`. All experiments are automated and reproducible via `python scripts/run_experiments.py`.

---

## 1. Threshold Sweep ($N \propto 1/C$)

### Motivation & Theory
The neuromorphic contrast threshold $C$ controls event sensitivity. In a scene undergoing continuous motion without threshold saturation, the total number of events $N$ generated across an interval is inversely proportional to $C$:

$$N \approx \frac{\int \int |\Delta L(x, y, t)|\,dx\,dy}{C} \implies N \cdot C \approx \text{const}$$

### Experimental Results
Varying $C$ uniformly from $0.10$ to $0.40$ on the synthetic 960 FPS benchmark yields:

| Threshold $C$ | Total Events $N$ | Event Rate over Input (Hz) | Normalization Metric ($N \cdot C$) |
|:---:|---:|---:|---:|
| **0.10** | 256,211 | 1,546,934.3 | 25,621.1 |
| **0.15** | 156,546 | 945,183.4 | 23,481.9 |
| **0.20** | 114,563 | 691,701.1 | 22,912.6 |
| **0.30** | 72,298 | 436,516.2 | 21,689.4 |
| **0.40** | 51,748 | 312,440.8 | 20,699.2 |

The total count decreases strictly monotonically as $C$ increases, while $N \cdot C$ remains within the same order of magnitude ($\approx 2.1 \times 10^4$ to $2.5 \times 10^4$), directly corroborating the theoretical scaling model.

Artifacts: `output/experiments/threshold_sweep.csv`, `output/experiments/threshold_sweep.png`.

---

## 2. Input Frame Rate Convergence (FPS Sweep)

### Motivation & Methodology
Because real event cameras operate with continuous-time analog circuits, a discrete-frame simulator approximates continuous irradiance changes using piecewise-linear interpolation between frames. As the input frame rate increases, the discrete temporal error must asymptotically converge toward continuous-time ground truth.

Events generated at lower frame rates (120, 240, 480, 960, 1920 FPS) were compared against a high-rate baseline (3840 FPS) using an order-preserving dynamic programming sequence alignment with an insertion/deletion penalty:

| Input FPS | Emitted Events | Relative Count Diff | Matched Events | Unmatched Events | Timestamp RMSE vs 3840 FPS |
|---:|---:|---:|---:|---:|---:|
| **120** | 35 | 0.0789 | 35 | 3 | 2,221.1 $\mu\text{s}$ |
| **240** | 37 | 0.0263 | 37 | 1 | 1,363.4 $\mu\text{s}$ |
| **480** | 38 | 0.0000 | 38 | 0 | 790.5 $\mu\text{s}$ |
| **960** | 38 | 0.0000 | 38 | 0 | 326.5 $\mu\text{s}$ |
| **1920** | 38 | 0.0000 | 38 | 0 | 136.2 $\mu\text{s}$ |
| **3840** | 38 | 0.0000 | 38 | 0 | 0.0 $\mu\text{s}$ (baseline) |

### Observation
- At $\ge 480\text{ FPS}$, the event count converges with $100\%$ zero-defect sequence recovery (0 unmatched events).
- Timestamp RMSE decreases monotonically toward zero as frame rate doubles, validating temporal convergence under piecewise-linear interpolation.

Artifacts: `output/experiments/fps_sweep.csv`, `output/experiments/fps_sweep.png`.

---

## 3. Sensor Non-Ideality Ablation

### Progression of Models
To isolate the effect of each physical non-ideality, four progressive sensor models were evaluated on the same 960 FPS sequence:
1. **Model A (Ideal)**: Fixed uniform thresholds $C^+ = C^- = 0.20$, no noise, zero refractory period.
2. **Model B (+ Threshold Mismatch)**: Adds static per-pixel Gaussian threshold variation ($\sigma = 0.02$).
3. **Model C (+ Background Activity)**: Adds Poisson dark current / leakage noise ($0.02\text{ Hz/pixel}$).
4. **Model D (+ Refractory Period)**: Adds a $100\ \mu\text{s}$ refractory dead-time filter.

| Model Stage | Total Events | ON Events | OFF Events | ON/OFF Ratio | Rate over Input (Hz) |
|---|---:|---:|---:|---:|---:|
| **A: Ideal** | 69,767 | 34,221 | 35,546 | 0.9627 | 421,234.7 |
| **B: + Mismatch** | 71,051 | 34,823 | 36,228 | 0.9612 | 428,987.2 |
| **C: + Background** | 71,150 | 34,865 | 36,285 | 0.9609 | 429,584.9 |
| **D: + Refractory** | 66,744 | 32,879 | 33,865 | 0.9709 | 402,982.6 |

### Physical Interpretation
- Threshold mismatch (Model B) slightly increases overall event count as pixels with sampled thresholds lower than $C$ fire more easily.
- Background noise (Model C) introduces sporadic, uncorrelated spontaneous events.
- Refractory dead-time (Model D) acts as an upper-rate limiter, eliminating burst events that occur faster than the circuit reset time, reducing the total count by $\approx 6.2\%$.

Artifacts: `output/experiments/noise_ablation.json`, `output/experiments/noise_ablation.png`.

---

## 4. Accumulation Window Comparison

### Visual Temporal Trade-Off
Event camera visualization requires accumulating discrete events into a 2D image over a temporal window $\Delta T_{\text{acc}}$. Accumulating over varying windows:

| Window ($\Delta T_{\text{acc}}$) | Events in Window | Active Pixels | Visual Characteristics |
|---:|---:|---:|---|
| **1,000 $\mu\text{s}$ (1 ms)** | 812 | 157 | Highly temporal, sparse edges, motion direction clearly separated. |
| **5,000 $\mu\text{s}$ (5 ms)** | 4,804 | 1,343 | Balanced representation; sharp object contours with minimal motion blur. |
| **10,000 $\mu\text{s}$ (10 ms)** | 9,088 | 2,125 | Solid outlines, suitable for slow-moving objects; slight temporal trail. |
| **20,000 $\mu\text{s}$ (20 ms)** | 17,676 | 3,180 | Dense spatial coverage; visible motion blur trail behind fast-moving disks. |

> [!NOTE]
> The accumulation window affects **visualization only**. The underlying event stream preserves exact microsecond timestamps regardless of the accumulation duration.

Artifacts: `output/experiments/accumulation_window.json`, `output/experiments/accumulation_window_montage.png`.

---

## 5. Computational Efficiency Benchmark

### Vectorized vs. Pixel-Loop Speedup
`evsim` implements two independent execution engines:
- **`vectorized`**: NumPy array broadcasting across all pixels in parallel.
- **`loop`**: Explicit nested pixel loops serving as an analytical reference.

Both backends were validated to produce mathematically identical event streams (`np.array_equal` passes across all frames and channels).

| Backend | Runtime (s) | Processing Throughput (FPS) | Megapixel-Frames / s | Emitted Events |
|---|---:|---:|---:|---:|
| **Vectorized** | 0.427 | 67.96 | 0.470 | 889,455 |
| **Loop (Reference)** | 5.318 | 5.45 | 0.038 | 889,455 |

Measured speedup: **$12.46\times$** for the vectorized backend over the pixel loop.
Artifacts: `output/benchmark.json`.

