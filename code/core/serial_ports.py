from __future__ import annotations

from typing import List

try:
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover - import guard for environments without dependencies
    list_ports = None
    PYSERIAL_IMPORT_ERROR = exc
else:
    PYSERIAL_IMPORT_ERROR = None

from config import DEFAULT_SIM_STREAM_URL
from config import PROJECT_ROOT


def list_serial_ports() -> List[str]:
    simulator_endpoints = [DEFAULT_SIM_STREAM_URL]
    streamer_data_dir = PROJECT_ROOT / "code" / "streamer_data"
    if streamer_data_dir.is_dir():
        for csv_path in sorted(streamer_data_dir.glob("*.csv"), key=lambda p: p.name.lower()):
            try:
                rel_path = csv_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                rel_path = str(csv_path.resolve())
            simulator_endpoints.append(f"csv://{rel_path}")
    if PYSERIAL_IMPORT_ERROR is not None:
        return simulator_endpoints

    ports = sorted(list_ports.comports(), key=lambda port: port.device)
    serial_ports = [port.device for port in ports]
    return simulator_endpoints + serial_ports
