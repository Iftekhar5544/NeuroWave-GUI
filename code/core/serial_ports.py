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


def list_serial_ports() -> List[str]:
    simulator_endpoints = [DEFAULT_SIM_STREAM_URL]
    if PYSERIAL_IMPORT_ERROR is not None:
        return simulator_endpoints

    ports = sorted(list_ports.comports(), key=lambda port: port.device)
    serial_ports = [port.device for port in ports]
    return simulator_endpoints + serial_ports
