# Abandoned / Removed Object, Scene-Change and Camera-Tamper Monitor

A computer-vision system that watches a video or camera stream and raises an alarm when a bag is left unattended. It also tells the difference between an object that was **left behind** and one that was **taken away**, ignores shadows and lighting changes, and warns when the camera itself is blocked, blurred, frozen or moved.

Everything runs in software on a normal laptop CPU. No special hardware is needed.

<!-- Add a screenshot of the GUI here, for example: -->
<!-- ![GUI](report/figures/screenshot.png) -->

## Features

- **Dual background model** (a fast and a slow model). A stopped object is found as "long-term foreground but not short-term foreground". I wrote my own running-average version and also support MOG2 and KNN for comparison.
- **Abandoned vs removed:** an edge test on the blob boundary, plus an optional 6-channel MobileNetV2 that sees the current crop and the background crop together.
- **Shadow and lighting handling:** HSV and texture test for shadows; edge-map correlation to tell a lighting change from a real scene change.
- **Owner logic:** people are detected and tracked (YOLOv8n + ByteTrack, HOG+SVM as a baseline). The owner is the person nearest the object when it stopped. Distance is measured in metres on the floor using a homography, and the owner is re-identified by colour after a tracker ID switch.
- **Alarm rules:** warning when the owner is more than 2 m away, alarm after 3 m for 30 s (the PETS2006 definition). Both are configurable.
- **Presence check:** the stored template is matched against the scene (normalised cross-correlation + SIFT) so an object survives occlusion, and "taken" is detected.
- **Camera-health gate:** blocked, blurred, frozen, shifted or disconnected camera. Small shake is stabilised.
- **Zones:** watch zones and ignore zones drawn on the video.
- **Alerts:** on screen, CSV log, snapshot, short evidence clip, and optional Telegram message.
- **Web GUI** (Flask, runs locally in the browser): live video, event log, tracked objects, settings, zone and floor calibration, run history, and an evaluation page.
- **Evaluation tools:** event-level precision / recall / F1, alarm delay, false alarms per hour, FPS, and 14 ablation variants.

All timers use seconds taken from frame timestamps, never frame counts.

## How it works

1. Camera-health gate (`tamper.py`)
2. Preprocess (`preprocess.py`)
3. Dual background model (`background.py`)
4. Zones (`pipeline.py`)
5. Candidate blobs with shadow removal (`candidates.py`)
6. Global-change check: lighting vs scene change (`tamper.py`)
7. Verification: edge test, optional CNN (`verify.py`, `classifier.py`)
8. Owner detection, distance and re-identification (`owner.py`)
9. State machine: CANDIDATE → STATIC → UNATTENDED → ALARM (`state_machine.py`)
10. Alerts, snapshots, evidence clips (`alerts.py`, `recorder.py`)

## Project layout

```
cv-monitor/
├── app.py              # starts the web GUI
├── run.py              # command-line runner
├── config.yaml         # all thresholds, timers and switches
├── src/                # the detection pipeline
├── gui/                # Flask server + web front end
├── eval/               # evaluation and ablation scripts
├── train/              # synthetic data generator + CNN training
├── tools/              # demo clip generator, floor calibration
├── tests/              # unit and integration tests
└── data/clips/         # sample video (demo.mp4)
```

## Installation

Use **Python 3.10 to 3.12** (3.11 recommended). Very new Python versions do not have torch / OpenCV builds yet.

```bash
git clone https://github.com/YOUR-USERNAME/cv-monitor.git
cd cv-monitor
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If the video window or the HOG detector stops working after installing, a second OpenCV build has overwritten the first. Fix it with:

```bash
pip uninstall -y opencv-python-headless
pip install --force-reinstall "opencv-python<5"
```

## Usage

**Web GUI**

```bash
python app.py
```

Open http://127.0.0.1:8000, choose `demo.mp4`, and press Start. You can also upload your own video from the GUI.

**Command line**

```bash
python run.py --video data/clips/demo.mp4 --show
python run.py --video your_video.mp4 --show --set pipeline.people=none
```

`--set` overrides any value in `config.yaml`, for example `owner.alarm_s=15`.

**Tests**

```bash
python tests/run_all.py
```

## Sample video

`data/clips/demo.mp4` is a synthetic clip made by `tools/make_demo_clip.py` (run it to regenerate the file). It contains an abandoned bag, an object taken away, a lighting change, a camera shift and a blocked lens. Real test videos are not included; use your own clips or download public datasets such as PETS2006 or ABODA from their official sources.

## Evaluation

Ground truth is stored in `data/labels.csv` with the columns `video,type,t_start,t_end`. The type is `abandoned`, `removed`, `tamper` or `none` (a clip with no events).

```bash
python eval/run_eval.py          # event-level metrics
python eval/ablations.py         # all ablation variants
```

Both can also be run from the Results page of the GUI.

## Training the CNN (optional)

Without trained weights the system uses the rule-based edge test only.

```bash
python train/synth_data.py --out data/synth --floors 80 --per_floor 25
python train/train_cnn.py --data data/synth --epochs 12
```

This writes `models/cnn6.pt`. A GPU (for example Google Colab) makes training much faster.

## Limitations

- Accuracy on real footage has not been measured on a full dataset yet; the numbers from the synthetic demo only show that the code works.
- Re-identification uses colour only, so two people in similar clothes can be confused.
- Repeating floor patterns can confuse camera-shift estimation.
- Without a person detector, a slowly moving person can leak into the static-object mask.
- The GUI is a local single-user tool with no login. Keep it on `127.0.0.1`.
- Webcam input works only on the machine running the app; a hosted copy can use uploaded videos or stream URLs.

## Tech stack

Python, OpenCV, NumPy, PyTorch / torchvision (MobileNetV2), Ultralytics YOLOv8 + ByteTrack, Albumentations, scikit-learn, pandas, matplotlib, Flask.

## Licences and credits

- YOLOv8 / Ultralytics is licensed under **AGPL-3.0**, so this repository is public and the source is available.
- MobileNetV2 weights come from torchvision (BSD).
- Dual-foreground method: Porikli, Ivanov, Haga, *Robust Abandoned Object Detection Using Dual Foregrounds*, EURASIP Journal on Advances in Signal Processing, 2008.
- The abandoned/removed edge test follows the classic rule from the abandoned-object literature (for example Tian et al.).
- The Macnica `abandoned_object_detection` repository was read for ideas only; no code was copied.
- Datasets such as PETS2006 and ABODA are not included. Download them from their official sources and check their terms of use.

## Author

Nannuri Harshitha