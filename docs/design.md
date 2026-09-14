# Architecture & Mathematical Design

This document details the software architecture, physical sensor abstractions, and mathematical formulations implemented in `evsim`.

---

## 1. System Objective & Scope

The objective of `evsim` is to phenomenologically simulate the response of a neuromorphic event camera (such as DVS, ATIS, or DAVIS) given a sequence of high-frame-rate intensity frames.

```text
High-FPS Video / Image Sequence + Sensor Parameters
                     │
                     ▼
           Event Camera Simulator
                     │
                     ▼
  Asynchronous (x, y, t, p) Event Stream + Diagnostics
```

> [!IMPORTANT]
> The simulator operates purely at the sensor-physics and pixel-transduction level. It does not perform computer vision downstream tasks such as edge detection, optical flow estimation, tracking, or SLAM.

---

## 2. Decoupled Architecture

The codebase adheres to clean separation of concerns, decoupling the computational backend from frontends:

```text
┌─────────────────────────────────────────────────────────────┐
│                    Frontends & Interfaces                   │
│                                                             │
│   CLI (Tyro)          Desktop GUI (PySide6)   Web (Streamlit)│
│   `src/evsim/cli.py`  `src/evsim/gui.py`      `app.py`      │
└───────────────┬──────────────────────┬───────────────┬──────┘
                │                      │               │
                ▼                      ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                 High-Level Pipeline & IO                    │
│   `src/evsim/pipeline.py`      `src/evsim/sources.py`       │
│   `src/evsim/events.py`        `src/evsim/visualization.py` │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Core                         │
│   `src/evsim/simulator.py`     `src/evsim/preprocessing.py` │
│   `src/evsim/config.py`        `src/evsim/validation.py`    │
└─────────────────────────────────────────────────────────────┘
```

1. **Backend Core (`src/evsim/simulator.py`)**:
   - Maintains stateful per-pixel reference levels $L_{\text{ref}}$ and last event timestamps $t_{\text{last}}$.
   - Implements both a high-throughput **vectorized** engine and an analytical **loop** reference backend.
2. **IO & Data Sources (`src/evsim/sources.py`, `src/evsim/events.py`)**:
   - Supports OpenCV video streams and naturally sorted image sequences with microsecond timestamp files (`ts_frame.txt`).
   - Compact memory layout: structured NumPy array `EVENT_DTYPE` and streaming serialization (NPZ, CSV).
3. **Visualization & Rendering (`src/evsim/visualization.py`)**:
   - Frame-aligned accumulation window ($\Delta T_{\text{acc}}$).
   - Robust fallback video encoding pipeline (`mp4v` $\to$ `avc1` $\to$ `MJPG`).
4. **Three Decoupled Interfaces**:
   - **CLI (`src/evsim/cli.py`)**: Structured, type-safe command-line interface powered by Tyro.
   - **Desktop GUI (`src/evsim/gui.py`)**: Responsive PySide6 desktop application with multi-threaded simulation and real-time interactive preview.
   - **Web UI (`app.py`)**: Browser-accessible Streamlit interface suitable for cloud deployment.

---

## 3. Mathematical Formulation

### 3.1 Coordinate System & Event Tuple
An event $e_i$ is an asynchronous 4-tuple:
$$e_i = (x_i, y_i, t_i, p_i)$$
- **Pixel Coordinates**: $x_i \in [0, W - 1]$, $y_i \in [0, H - 1]$ (origin at top-left corner).
- **Timestamp**: $t_i \in \mathbb{N}$ in microseconds ($\mu\text{s}$), non-decreasing ($t_{i+1} \ge t_i$).
- **Polarity**: $p_i \in \{-1, +1\}$ (+1 for brightness increase, -1 for brightness decrease).

### 3.2 Photoreceptor Model & Log Transformation
Neuromorphic pixels integrate photocurrent using a logarithmic transimpedance amplifier:
$$L(x, y, t) = \ln(I(x, y, t) + \epsilon)$$
where:
- $I(x, y, t) \in [0.0, 1.0]$ is the normalized optical intensity.
- $\epsilon > 0$ is a small stabilization offset (default: $10^{-3}$) that models dark current and avoids $\ln(0)$ singularity.

If `input.linearize` is enabled, input digital numbers ($\text{DN} \in [0, 2^B - 1]$) are linearized prior to logarithm:
$$I = \left(\frac{\text{DN}}{2^B - 1}\right)^\gamma$$
where $\gamma = 2.2$ by default.

### 3.3 Temporal Contrast & Threshold Crossings
Between consecutive frames at times $t_0$ and $t_1$, the change in log intensity relative to the pixel's stateful reference level $L_{\text{ref}}$ is:
$$\Delta L(t) = L(x, y, t) - L_{\text{ref}}(x, y)$$

An event triggers whenever $|\Delta L(t)|$ reaches the contrast threshold:
- **ON Event** ($p = +1$): $\Delta L \ge C^+$
- **OFF Event** ($p = -1$): $\Delta L \le -C^-$

### 3.4 Multiple Crossings in a Single Interval
For high-velocity motion or low frame rates, the brightness change across $[t_0, t_1]$ may span multiple threshold multiples:
$$N^+ = \left\lfloor \frac{\max(L(t_1) - L_{\text{ref}}, 0)}{C^+} \right\rfloor$$
$$N^- = \left\lfloor \frac{\max(L_{\text{ref}} - L(t_1), 0)}{C^-} \right\rfloor$$

Each $k$-th crossing occurs at the discrete log level:
$$L_k = \begin{cases} L_{\text{ref}} + k \cdot C^+, & \text{if } N^+ > 0 \\ L_{\text{ref}} - k \cdot C^-, & \text{if } N^- > 0 \end{cases}$$

### 3.5 Sub-Frame Piecewise-Linear Interpolation
Assuming linear intensity change between $t_0$ and $t_1$:
$$L(t) = L(t_0) + \alpha \cdot (L(t_1) - L(t_0)), \quad \alpha \in [0.0, 1.0]$$

The continuous crossing timestamp $t_k^*$ is calculated by solving for $\alpha_k$:
$$\alpha_k = \frac{L_k - L(t_0)}{L(t_1) - L(t_0)}$$
$$t_k^* = t_0 + \alpha_k \cdot (t_1 - t_0)$$

If $L(t_1) \approx L(t_0)$ while a residual crossing remains, the simulator uses the explicit convention $\alpha_k = 1$ and assigns the residual event to $t_1$.

### 3.6 Timestamp Floor Quantization
Analog event camera readout systems operate with finite clock resolution $\Delta t_{\text{res}}$ (default: $1\ \mu\text{s}$):
$$t_k = \left\lfloor \frac{t_k^*}{\Delta t_{\text{res}}} \right\rfloor \cdot \Delta t_{\text{res}}$$
The continuous crossing time is floor-quantized once to the sensor clock grid. Multiple events within the same clock bin share the same quantized timestamp.

### 3.7 Refractory Period Suppression
Physical photoreceptors require a finite circuit reset and refractory dead-time $\Delta t_{\text{refr}}$ after generating an event:
$$\text{If } t_k - t_{\text{last}}(x, y) < \Delta t_{\text{refr}} \implies \text{Event is dropped.}$$

For accepted events:
$$t_{\text{last}}(x, y) \leftarrow t_k$$
$$L_{\text{ref}}(x, y) \leftarrow L_k$$

> [!NOTE]
> The reference level $L_{\text{ref}}$ updates to the crossed threshold level $L_k$, **not** to the end-of-frame intensity $L(t_1)$. Any remaining residual difference $(L(t_1) - L_k)$ is preserved across subsequent frame intervals. In the simplified model, refractory-suppressed contrast events do not update $L_{\text{ref}}$; this state convention is deliberate and not a transistor-level model.

---

## 4. Sensor Non-Ideality Models

### 4.1 Transistor Threshold Mismatch
Due to fabrication tolerances in sub-threshold MOS transistors, contrast thresholds exhibit static spatial variation:
$$C^+(x, y) \sim \mathcal{N}\left(C^+, \sigma_C^2\right), \quad C^-(x, y) \sim \mathcal{N}\left(C^-, \sigma_C^2\right)$$
These threshold maps are sampled once at simulator initialization and remain fixed across the entire sequence.

### 4.2 Background Leakage Activity
Thermal noise and parasitic junction leakage generate spontaneous background events modelled as a spatial Poisson process with rate $\lambda_{\text{bg}}$ (Hz/pixel):
$$P(k \text{ background events in } \Delta t) = \frac{(\lambda_{\text{bg}} \Delta t)^k e^{-\lambda_{\text{bg}} \Delta t}}{k!}$$
Background events are assigned uniformly distributed timestamps in $[t_0, t_1]$ and random polarities $p \in \{-1, +1\}$. Accepted background events undergo refractory checking but do not modify $L_{\text{ref}}$.

---

## 5. Explicit Modeling Assumptions ($A1$–$A7$)

1. **A1 (Piecewise Linear Intensity)**: Log intensity varies linearly between consecutive input frames.
2. **A2 (Pixel Independence)**: Each pixel operates as an independent state machine without spatial cross-talk.
3. **A3 (Contrast Thresholding)**: Events are triggered strictly by relative changes exceeding $C^+$ and $C^-$.
4. **A4 (Fixed Mismatch)**: Threshold variations are fixed per-pixel offsets for the duration of a capture.
5. **A5 (Poisson Noise)**: Background activity is governed by an uncorrelated homogeneous Poisson point process.
6. **A6 (No Readout Bus Contention)**: Readout bus arbitration, AER handshake bottlenecks, and FIFO buffer overflow are not simulated.
7. **A7 (Information Bound)**: The simulator cannot recover scene information already destroyed by source camera motion blur, low shutter speed, or severe temporal aliasing.

---

## 6. Visualization & Color Encoding Conventions

Event streams are accumulated over a frame-aligned window $\Delta T_{\text{acc}}$ (default: $10,000\ \mu\text{s} = 10\text{ ms}$) to generate illustrative video frames in OpenCV BGR format:

| Event State | BGR Value | Displayed Color |
|---|---|---|
| No events | `(25, 25, 25)` | Dark Gray |
| ON event ($p = +1$) | `(20, 80, 255)` | Red |
| OFF event ($p = -1$) | `(255, 80, 20)` | Blue |
| Both ON & OFF | `(255, 0, 255)` | Magenta |

The composite video frame arranges three panels side by side:
`[ Input Grayscale Frame | Accumulated Events | Alpha-Blended Overlay ]`
