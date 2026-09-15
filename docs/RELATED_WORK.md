# Related Work

## Event-camera simulators

### Event-camera dataset and simulator

Mueggler et al. introduced a widely used event-camera dataset and simulator
that converts high-frame-rate intensity images to events using temporal
interpolation and contrast thresholds. The approach establishes the baseline
`frame -> threshold crossing -> event` pipeline adopted here.

### ESIM

ESIM, by Rebecq et al., focuses on efficient and high-quality synthetic event
generation, including adaptive rendering and contrast-threshold simulation.
Our project borrows the event-generation model and interpolation idea but does
not reproduce the full 3D rendering system, because the CA input is already a
high-FPS frame sequence.

### v2e

v2e, by Hu et al., converts video frames to realistic DVS events and models
several sensor non-idealities such as threshold mismatch, noise, leak events
and refractory behavior. Our implementation adopts only a deliberately small
subset: fixed per-pixel Gaussian threshold mismatch, simplified Poisson
background activity and a refractory period. We do not claim full physical
sensor fidelity.

## Positioning of this project

| Capability | Event-camera dataset simulator | ESIM | v2e | This CA |
|---|---:|---:|---:|---:|
| High-FPS frame sequence input | yes | rendered scene | video | yes |
| Stateful log-intensity threshold model | yes | yes | yes | yes |
| Multiple events per frame interval | yes | yes | yes | yes |
| Temporal interpolation | yes | adaptive | yes | linear |
| Asymmetric thresholds | limited | yes | yes | yes |
| Threshold mismatch | limited | yes | yes | simplified |
| Background activity | no | limited | yes | simplified |
| Refractory period | no | limited | yes | yes |
| Physical pixel/readout model | no | partial | extensive | no |
| Reproducible analytical validation | limited | research tool | research tool | explicit |
| Course-scale understandable codebase | no | no | no | yes |

## Project statement

This project is a configurable and verifiable implementation
focusing on the log-intensity contrast-threshold model, temporal interpolation,
simplified sensor non-idealities, reproducible validation and computational
efficiency.

It intentionally does not implement object detection, tracking, optical flow,
SLAM, reconstruction or deep-learning downstream tasks.

## References

1. G. Gallego et al., "Event-based Vision: A Survey," IEEE TPAMI, 2020.
2. E. Mueggler, H. Rebecq, G. Gallego, T. Delbruck, D. Scaramuzza,
   "The Event-Camera Dataset and Simulator," IJRR, 2017.
3. H. Rebecq, D. Gehrig, D. Scaramuzza, "ESIM: an Open Event Camera
   Simulator," CoRL, PMLR 87, 2018.
4. Y. Hu, S.-C. Liu, T. Delbruck, "v2e: From Video Frames to Realistic DVS
   Events," CVPR Workshops, 2021.
5. Sensory Systems and Robotics event-based vision resources:
   https://github.com/uzh-rpg/event-based_vision_resources
6. Event-based datasets collection:
   https://github.com/lisiqi19971013/event-based-datasets
