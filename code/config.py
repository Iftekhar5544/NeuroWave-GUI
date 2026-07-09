from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

APP_TITLE = "NeuroWave-EEG"
DEFAULT_CHANNEL_COUNT = 16
DEFAULT_WINDOW_SECONDS = 8
DEFAULT_SAMPLE_RATE = 125
DEFAULT_POLL_INTERVAL_MS = 50
DEFAULT_REDRAW_INTERVAL_MS = 40
DEFAULT_STREAM_BUFFER_SIZE = 45000
DISPLAY_FIXED_SCALE_UV = 200.0
DISPLAY_HIGHPASS_HZ = 1.0
DISPLAY_LOWPASS_HZ = 45.0
DISPLAY_NOTCH_FREQ_HZ = 50
DISPLAY_FILTER_PAD_SECONDS = 1.0
DISPLAY_AUTOSCALE_MIN_UV = 20.0
DISPLAY_AUTOSCALE_MARGIN = 1.35
DISPLAY_AUTOSCALE_SMOOTHING = 0.82
DISPLAY_AUTOSCALE_DECAY = 0.985
# Recompute per-channel auto-scale ranges every N redraw frames instead of every
# frame. At the default 40ms redraw interval this yields ~5 Hz autoscale updates,
# which is visually smooth while cutting the per-frame percentile cost by ~5x.
DISPLAY_AUTOSCALE_REFRESH_FRAMES = 5
# Only re-apply a channel's Y-range when its target scale moves more than this
# fraction, to avoid redundant viewbox range updates every frame.
DISPLAY_AUTOSCALE_APPLY_THRESHOLD = 0.02
DISPLAY_FILTER_HISTORY_SECONDS = 2.0
DISPLAY_FILTER_ORDER = 2
DISPLAY_FILTER_PRESET = "Raw"
DISPLAY_USER_PRESETS_FILE = "filter_user_presets.json"
STATUS_READY = "Ready"
STATUS_CONNECTED = "Connected"
STATUS_STREAMING = "Streaming"
DEFAULT_SIM_STREAM_URL = "sim://225.1.1.1:6677"
DEFAULT_SIM_MULTICAST_IP = "225.1.1.1"
DEFAULT_SIM_MULTICAST_PORT = 6677
