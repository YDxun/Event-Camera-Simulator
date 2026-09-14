# EE5110/EE6110 Segment B — Continuous Assessment (CA) Specification

> **Purpose of this file**
>
> This specification extracts the CA requirements and CA-relevant references stated in the provided **Week 1 Event Based Vision** and **Week 2 Event Based Vision** lecture PDFs. It is intended to serve as a programming agent's authoritative task specification together with the original PDFs.
>
> This document is **not an implementation plan**. It does not prescribe software architecture, libraries, programming language, optimization strategy, experimental protocol, or optional extensions beyond what the lecture materials state. Items phrased in the slides as questions, considerations, examples, or references remain identified as such rather than being converted into mandatory requirements.

---

## 1. Assignment Identity

### 1.1 Assignment

**Continuous Assessment (CA): Design an Event Camera Simulator / Event Data Simulator**

The Week 1 slides use both titles:

- “CA: Design an Event Camera Simulator”
- “Discussion on CA: Design an Event Data Simulator”

Both refer to the same CA task.

### 1.2 Aim

Design a **simulation framework to generate events, given high frame-rate images as inputs**.

The reference input/output structure shown in the slides is:

```text
Inputs                              Output
------                              ------
High-FPS frames  ─┐
(e.g. 960 FPS)    ├─> Event simulator ─> Simulated events
Parameters        ─┘                     + illustrative video
```

The event-camera principle and event-data simulation mechanism taught in Week 1 form the technical background for the assignment.

---

## 2. Group and Submission Requirements

### 2.1 Group Work

The CA is **group work**.

- Form groups of **1–5 students**.
- Each group must have **one group leader**.
- Students are instructed to self-organize into groups in **Canvas**.

### 2.2 Required Submission Components

The CA submission consists of **four parts**:

1. **Presentation video**, **less than 15 minutes**
2. **Presentation slides**
3. **Source code**
4. **AI Use and Review Report**, **if any AI tool was used**, in **PDF format**

### 2.3 Submission Package

Submit all required materials in **one ZIP file**.

Required filename:

```text
<group_leader_NUS_ID>.zip
```

Example from the slides:

```text
A0123456X.zip
```

### 2.4 Deadline

All CA results must be submitted to **Canvas** by:

**11:59 pm, Friday, 18 September 2026 (Week 6).**

---

## 3. Group Contribution Weight

A contribution table must be attached **at the end of the presentation slides**.

The table contains:

| No. | Name | Student number | Contribution weight |
| --- | --- | --- | --- |
| 1 (team leader) |  |  | \(w_1\) |
| 2 |  |  | \(w_2\) |
| … |  |  | … |

The slides state that the final mark of each student is based on the group's answer mark and weighted according to contribution weights:

\[
FM_i = \frac{w_i}{\sum_i w_i}\sum_i M
\]

where \(i\) denotes the student index.

For equal contribution:

- enter **“1” for all contribution weights**;
- answers submitted **without the table** are deemed to have **equal contribution**.

---

## 4. Required Presentation Content

The course states that the written report / presentation for CA and RP should contain **at least** the following:

1. **Abstract**
   - Briefly summarize what was done and what the results are.
2. **Introduction**
   - Brief introduction to the project and the problem being solved.
3. **Derivation of the model (“problem formulation”)**
4. **Description of the framework and algorithm(s) used**
5. **Results and discussion**
6. **Conclusions and/or summary**
7. **References**, if applicable

For the CA, the explicitly required submitted presentation artifact is the **presentation slides**; the CA submission list does not separately specify a written report.

---

## 5. Event-Camera Model Taught as the Basis of the CA

### 5.1 Event Output

Unlike a traditional camera that outputs frames at fixed time intervals, an event camera outputs **asynchronous events**.

An event is generated when a single pixel detects an intensity change larger than a threshold.

The slides represent an event as:

\[
(t,x,y,\mathrm{sign})
\]

with:

- \(t\): timestamp (microsecond-scale in the event-camera description)
- \(x,y\): pixel coordinates
- sign / polarity: \(-1\) or \(+1\), indicating decrease or increase of brightness

The CA pixel-array discussion uses the event-list ordering:

\[
(x,y,p,t)
\]

where \(p\) is event polarity.

### 5.2 Generative Event Model

For a single pixel, the Week 1 generative model is:

\[
\pm C = \log I(\mathbf{x},t)-\log I(\mathbf{x},t-\Delta t)
\]

where:

- \(I\) is intensity;
- \(C\) is the **contrast threshold**;
- positive and negative threshold crossings produce ON and OFF events;
- events are triggered asynchronously.

The slides also define:

\[
L(x,y,t)=\log(I(x,y,t))
\]

and derive the first-order approximation:

\[
-\nabla L\cdot \mathbf{u}=C
\]

under brightness constancy, where \(\mathbf{u}=(u,v)\) is motion / optical flow over the interval.

The Week 2 material further describes EVS triggering thresholds \(c(k)\) as **known triggering thresholds that are configurable**.

---

## 6. CA Design Discussion: Bottom-Up Pixel Design

The Week 1 CA discussion presents **top-down and bottom-up design philosophies** and asks which one to use. It does not mandate one.

For the **bottom-up method**, the slides explicitly discuss the following pixel design.

### 6.1 Single-Pixel Inputs

Input:

- **Digital Numbers (DN) on a single pixel with respect to time**

### 6.2 Single-Pixel Parameters

Parameters shown include:

- **Triggering threshold**
- **Timestamp resolution**
- `…` (the slide explicitly leaves room for additional parameters)

### 6.3 Single-Pixel Output

Output:

```text
Single pixel events (p, t)
```

### 6.4 Pixel Processing Items Shown in the CA Discussion

The slide identifies the following processing/design items:

1. **DN → log illuminance**
2. **Time**
3. **Discrete-to-continuous interpolation**
   - The slide explicitly asks: **“What is the interpolation method?”**
4. **Noise simulation**
   - The slide explicitly asks: **“What is your assumption on noises?”**
5. At intervals determined by the **timestamp resolution**, check the log-illuminance change against the triggering threshold:
   - shown on the slide as **“ΔI > c?”**

The slides do **not** prescribe a particular interpolation method or a particular noise model in the CA discussion.

---

## 7. CA Design Discussion: Pixel Array

For the bottom-up method, the single-pixel design is extended to a pixel array.

The conceptual process shown is:

```text
Event sensing pixel
        |
Single pixel events (p,t)
        |
Repeat for each pixel
        |
Event list (x,y,p,t)
```

### 7.1 Scanning / Arbitration

The slides state:

- In practice, a sensor has a **scanning mechanism / arbitration logic**.
- The CA discussion asks: **“Could you propose any scanning mechanism?”**
- If not, a **“naïve but effective alternative”** is to treat **each pixel independently**, which is described as easy to implement within a loop.

Therefore, proposing a scanning mechanism is presented as a design question; the independent-pixel approach is explicitly given as an acceptable alternative in the lecture discussion.

---

## 8. CA Design Discussion: I/O and Visualization

### 8.1 Input

The CA I/O discussion states:

- **Input: a video clip**
- Design question: **“Supporting multiple FPS?”**

### 8.2 Output

The CA I/O discussion states:

- **Output: an event list**

The pixel-array discussion represents this event list as:

\[
(x,y,p,t)
\]

### 8.3 Visualization

The CA I/O discussion states:

- **Visualization: Event frames on top of the input video**
- Design question: **“Supporting customized accumulation time?”**

The overall CA diagram additionally specifies an **illustrative video** as part of the simulator output.

---

## 9. Other Explicit CA Design Considerations

The Week 1 CA discussion lists the following under **“Other considerations”**:

- **Running efficiency?**
- **User-friendly interface / visualization?**
- **Output data compression?**
- **And more…**

These are explicitly presented as **considerations/questions**, not as separately stated mandatory submission requirements.

---

## 10. Event Simulation References Provided in Week 1

The Week 1 lecture introduces event simulation to explain the assignment context.

### 10.1 Why Event Data Simulation

Reasons given:

- Not everyone has an event camera.
- Simulation allows configuration of event-generation parameters, including:
  - noise level
  - resolution
  - triggering threshold
  - …
- Simulation helps build the event-sensor model mathematically.

### 10.2 ESIM

**ESIM** is presented as an event-data simulator.

The lecture diagram shows a rendering/simulation setup involving scene, camera trajectory, camera parameters, intensity, optical flow, camera pose, translational/rotational velocities, and simulated events.

### 10.3 v2e

**v2e** is presented as another event-data simulator.

The Week 1 slide describes its processing conceptually as including:

- RGB → luma video
- optional interpolation
- lin-log conversion
- intensity-dependent low-pass filtering
- DVS event generation at each pixel
- temporal noise
- DVS frame accumulation

### 10.4 Noise Examples from v2e

The Week 1 lecture gives these examples of event-generation noise/nonidealities:

**Threshold mismatch**
- Typical threshold: \(\theta=0.3\)
- Threshold variation modeled with a Gaussian distribution:
  \[
  \sigma_\theta \approx 0.03
  \]

**Hot pixels**
- Some EVS pixels continuously fire events at a high rate even without input.

**Leak noise events**
- Spontaneous ON events.
- Typical rate given:
  \[
  \approx 0.1\ \mathrm{Hz}
  \]

These are examples/reference material from the lecture; the CA discussion itself asks students to choose and state their noise assumptions rather than mandating all of these effects.

### 10.5 ESIM vs. v2e Comparison

The lecture describes:

**ESIM**
- Complex front-end
- Can simulate camera motion
- Programmed in C++
- Basic pixel model
- Contrast thresholds with Gaussian noise for the whole sensor

**v2e**
- Simpler front-end
- Processes movies without modeling 3D environments
- Programmed in Python
- More realistic pixel model, including:
  - pixel-to-pixel Gaussian temporal contrast-threshold variation
  - finite, intensity-dependent photoreceptor bandwidth
  - leak events / intensity-dependent background activity noise
  - intensity-dependent temporal noise

These are reference comparisons, not a statement that the CA must reproduce either simulator in full.

---

## 11. Event Representations Relevant to the Required Output/Visualization

Week 1 introduces several event representations.

### 11.1 Raw Events / Event Point Cloud

Raw event data retain all information but require algorithms that can accept point-cloud-like input.

### 11.2 Event Frames

Events can be accumulated over an **accumulation interval** into different event frames, including:

- **Event binary frame**
  - indicates whether a pixel has been triggered
- **Event count frame**
  - records the number of triggering times
- **Event flow frame**
  - represents pixel triggering time

The lecture states that event frames:

- are easy to process using off-the-shelf machine-vision algorithms;
- lose information;
- contain only events in the accumulation interval.

This material provides context for the CA visualization requirement of **event frames on top of the input video** and the question about customizable accumulation time.

### 11.3 Event Tensors

Event tensors add another dimension to event frames to represent events over a longer interval. The lecture notes that their interval number/length normally requires experimentation depending on the application.

Event tensors are lecture material; they are not separately required by the CA discussion.

---

## 12. Data Compression Context

Week 1 explains that event data rate depends on the scene and is not necessarily always lower than conventional-camera data rate.

Two broad event-data compression streams are introduced:

- **Lossy compression**
  - event frames and event tensors are identified as lossy representations
- **Lossless compression**
  - presented as the main research direction for event-data compression

EVS-specific compression can exploit:

- temporal redundancy
- spatial redundancy

Examples discussed include:

- **Address-Prior Mode**
  - temporal redundancy
  - delta coding on time
- **Time-Prior Mode**
  - spatial redundancy
  - delta coding on \(x,y\)

For the CA, **output data compression** appears only under “Other considerations”; no particular compression scheme is mandated.

---

## 13. Week 2 Material Relevant as Technical Reference

Week 2 primarily covers downstream event-based vision algorithms and applications rather than issuing a new CA specification. The following material is directly relevant as reference to simulator behavior/modeling.

### 13.1 EVS Nonidealities

The supplementary material discusses EVS nonidealities including:

- **pixel latency**
- **readout latency**
- slower EVS response under low-light conditions
- **pixel refractory period**

It states that these nonidealities can significantly affect downstream image reconstruction/interpolation quality.

The accompanying sensor diagram illustrates:

- event firing
- updating the reference level
- readout latency
- refractory period

These are technical references; Week 2 does not state that the CA simulator must implement all of them.

### 13.2 Configurable Triggering Thresholds

In its EVS measurement formulation, Week 2 states that \(c(k)\) represents **known triggering thresholds that are configurable**.

### 13.3 Example Event Image Representation

In the EV-FlowNet discussion, Week 2 gives an example image-form event representation with four channels at camera resolution:

- positive event-count image
- negative event-count image
- positive event-timestamp image normalized to \([0,1]\)
- negative event-timestamp image normalized to \([0,1]\)

This is an example used by a downstream optical-flow method, not a required CA output format.

### 13.4 Surface of Active Events

For event-based corner detection, Week 2 introduces the **Surface of Active Events (SAE)**:

- a map storing the timestamp of the latest event at each pixel.

This illustrates that meaningful event streams contain exploitable spatiotemporal timestamp structure. SAE is not specified as a required CA output.

---

## 14. AI Policy

AI tools are **optional**.

If **any AI tool is used**, the group must submit a concise **AI Use and Review Report** documenting its use and the group's review of AI-generated content.

The report must include:

1. **AI tool used**
2. **Purpose of using AI**
3. **2–3 representative prompts**
   - full chat history is not required
4. **Summary of useful AI feedback**
5. **Changes made after AI feedback**
6. **One AI suggestion not accepted, with reason**
7. **Brief reflection on the reliability / limitations of AI**

The course further states:

- Students are fully responsible for the correctness of submitted code and outputs.
- Students must understand and be able to explain submitted work.
- Any accessible AI tool may be used, subject to university and course guidelines.

If no AI tool is used, the submission is regarded as entirely the students' own work without AI assistance.

---

## 15. Requirement Status Clarification

To preserve the distinction made in the lecture material:

### Explicit assignment/submission requirements include

- group work (1–5 students) and a group leader;
- design an event simulator that generates events from high-frame-rate image/video input;
- simulator inputs include high-FPS frames and parameters;
- simulator output includes simulated events and an illustrative video;
- presentation video <15 minutes;
- presentation slides;
- source code;
- AI Use and Review Report in PDF if AI is used;
- ZIP naming by group leader's NUS ID;
- contribution table at the end of slides;
- required presentation sections;
- Canvas deadline;
- CA I/O discussion explicitly identifies video input, event-list output, and event-frame visualization over the input video.

### Items explicitly posed by the CA slides as design questions or considerations include

- top-down vs. bottom-up design;
- interpolation method;
- noise assumptions;
- scanning mechanism / arbitration logic;
- support for multiple FPS;
- customized accumulation time;
- running efficiency;
- user-friendly interface / visualization;
- output data compression;
- other possible considerations.

### Technical references/examples, not separately stated CA requirements, include

- ESIM architecture/features;
- v2e architecture/features and noise models;
- specific v2e numerical noise values;
- event tensors;
- particular compression schemes;
- Week 2 EVS latency/refractory-period models;
- EV-FlowNet four-channel representation;
- SAE;
- downstream detection, tracking, optical-flow, SLAM, reconstruction, recognition, or motion-segmentation algorithms.

---

## 16. Source Scope

This specification is extracted only from:

- **Week 1 Event Based Vision.pdf**
- **Week 2 Event Based Vision.pdf**

Where the slides leave a choice open (for example interpolation method or noise assumption), this specification intentionally leaves it open. Where a feature is presented only as a question, consideration, example, simulator reference, downstream application, or supplementary material, this specification does not promote it to a mandatory CA requirement.
