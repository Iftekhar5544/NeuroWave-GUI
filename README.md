# NeuroWave-EEG

Desktop live-monitor and EEG machine-learning workflow app for `OpenBCI Cyton + Daisy`, built with `BrainFlow`, `PyQt5`, and `pyqtgraph`.

NeuroWave-EEG helps collect, visualize, label, train on, and run real-time predictions from 16-channel EEG streams. It also includes a BrainFlow synthetic-stream simulator for development without the physical board.

## What v1 does

- Lists available COM ports in a dropdown and lets you refresh them on demand
- Starts and stops the BrainFlow stream
- Displays 16 live EEG traces as a stacked scrolling graph
- Includes real-time display filter controls (preset + custom high-pass, low-pass, and notch)
- Includes an EEG ML pipeline window for labeled collection, model training, and real-time prediction
- Supports optional class-label image cues during guided data-recording sessions
- Keeps a fixed rolling window in memory for responsive plotting
- Logs lifecycle and error events to the terminal and `neurowave.log`

## Prerequisites

- Windows machine with the OpenBCI dongle and board already working
- Python `3.10` or `3.11` installed and available in `PATH`
- The OpenBCI GUI already successfully tested on the same hardware, which confirms the radio and driver path

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` is not available in `PATH`, install Python first or use the full path to your interpreter.

## Run

Open [code/main.py](c:\Users\User\Desktop\ML_ROOT\04_NeuroWave-EEG\code\main.py) in VS Code and run it directly with the Run button.

You can also run it from the terminal:

```powershell
python code/main.py
```

Run EEG simulator (for development without Cyton):

```powershell
python code/simulator_main.py
```

## How to use

1. Launch the app.
2. Click `Refresh` to load the currently available COM ports.
3. Pick your OpenBCI dongle COM port from the dropdown, or type one manually if needed.
4. Click `Connect`.
5. Click `Start` to begin the live EEG stream.
6. Click `Stop` to halt streaming.
7. Click the red `Disconnect` toggle button (same Connect button) when finished.
8. Use `Data Collection`, `Train Model`, `Load Model`, and `Realtime Prediction` to open the EEG ML workflow windows.
9. Without hardware, run `code/simulator_main.py`, then select `sim://225.1.1.1:6677` from COM dropdown and connect.

Filter notes:
- Filters are display-only and can be changed while streaming.
- Right sidebar `Display Filters` includes preset and custom controls with `Apply`.
- Presets: `Raw`, `OpenBCI Default`, `EEG Lab`, `Delta`, `Theta`, `Alpha`, `Beta`, `Gamma`, `Custom`, plus user-saved presets.

EEG ML notes:
- Training/inference is EEG-specific (not EMG feature logic).
- Collection writes labeled **raw EEG** rows with `label` and `trial_id` into a dataset CSV.
- Collection uses a guided auto protocol (`prep -> hold(record) -> rest`, repeated).
- Class labels can optionally include image cues. Images are copied into `code/images/class_label_image/` and saved with collection configs.
- After a completed collection session, the recording window stays on the finished screen until it is closed; then the live graph resumes buffering.
- Trained artifacts are Random-Forest models (with scaler) saved as `.joblib`.
- Live prediction uses the loaded model window/stride settings.
- Robustness metrics shown after training include holdout accuracy/F1 and cross-validation accuracy.

## Notes

- The app uses BrainFlow's `CYTON_DAISY_BOARD` profile.
- The actual sample rate is read from BrainFlow after connection.
- The graph is display-filtered only for visualization; ML collection data remains raw.
- If the board disconnects or the stream fails, the app keeps the UI responsive and surfaces the error in the status area.

## Project layout

```text
code/
  main.py
  simulator_main.py
  config.py
  images/
    class_label_image/
  core/
  ui/
dataset/
trained_model/
PROJECT_DESCRIPTION.md
README.md
requirements.txt
```
