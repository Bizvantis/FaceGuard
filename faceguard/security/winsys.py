"""Thin ctypes/winreg wrappers around the Windows features FaceGuard uses.

Everything here is user-level: locking the workstation, reading the session
lock state and keyboard/mouse idle time, and the per-user "Run at sign-in"
registry value.  Nothing requires administrator rights and nothing changes how
Windows authenticates you - unlocking always uses your normal PIN/password.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "FaceGuard"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _wtsapi32 = ctypes.WinDLL("wtsapi32", use_last_error=True)

    class _LastInputInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    class _WtsInfoExLevel1(ctypes.Structure):
        _fields_ = [
            ("SessionId", wintypes.ULONG),
            ("SessionState", ctypes.c_int),
            ("SessionFlags", wintypes.LONG),
            ("WinStationName", wintypes.WCHAR * 33),
            ("UserName", wintypes.WCHAR * 21),
            ("DomainName", wintypes.WCHAR * 18),
            ("LogonTime", ctypes.c_longlong),
            ("ConnectTime", ctypes.c_longlong),
            ("DisconnectTime", ctypes.c_longlong),
            ("LastInputTime", ctypes.c_longlong),
            ("CurrentTime", ctypes.c_longlong),
            ("IncomingBytes", wintypes.DWORD),
            ("OutgoingBytes", wintypes.DWORD),
            ("IncomingFrames", wintypes.DWORD),
            ("OutgoingFrames", wintypes.DWORD),
            ("IncomingCompressedBytes", wintypes.DWORD),
            ("OutgoingCompressedBytes", wintypes.DWORD),
        ]

    class _WtsInfoEx(ctypes.Structure):
        _fields_ = [("Level", wintypes.DWORD), ("Data", _WtsInfoExLevel1)]

    _user32.LockWorkStation.restype = wintypes.BOOL
    _user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LastInputInfo)]
    _user32.GetLastInputInfo.restype = wintypes.BOOL
    _kernel32.GetTickCount.restype = wintypes.DWORD
    _wtsapi32.WTSQuerySessionInformationW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)
    ]
    _wtsapi32.WTSQuerySessionInformationW.restype = wintypes.BOOL
    _wtsapi32.WTSFreeMemory.argtypes = [ctypes.c_void_p]
    _wtsapi32.WTSFreeMemory.restype = None

    _WTS_CURRENT_SESSION = 0xFFFFFFFF
    _WTS_SESSION_INFO_EX = 25
    _WTS_SESSIONSTATE_LOCK = 0
    _WTS_SESSIONSTATE_UNLOCK = 1


def lock_workstation() -> bool:
    """Lock the PC exactly like Win+L. Returns True if the request was accepted."""
    if not IS_WINDOWS:
        log.info("lock_workstation() ignored: not on Windows")
        return False
    ok = bool(_user32.LockWorkStation())
    if not ok:
        log.warning("LockWorkStation failed: %s", ctypes.get_last_error())
    return ok


def is_session_locked() -> bool | None:
    """True if the Windows session is locked, False if unlocked, None if unknown."""
    if not IS_WINDOWS:
        return None
    buf = ctypes.c_void_p()
    size = wintypes.DWORD()
    if not _wtsapi32.WTSQuerySessionInformationW(
        None, _WTS_CURRENT_SESSION, _WTS_SESSION_INFO_EX, ctypes.byref(buf), ctypes.byref(size)
    ):
        return None
    try:
        if size.value < ctypes.sizeof(_WtsInfoEx):
            return None
        info = ctypes.cast(buf, ctypes.POINTER(_WtsInfoEx)).contents
        if info.Level != 1:
            return None
        flags = info.Data.SessionFlags
        if flags == _WTS_SESSIONSTATE_LOCK:
            return True
        if flags == _WTS_SESSIONSTATE_UNLOCK:
            return False
        return None
    finally:
        _wtsapi32.WTSFreeMemory(buf)


def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input in this session."""
    if not IS_WINDOWS:
        return 0.0
    info = _LastInputInfo(ctypes.sizeof(_LastInputInfo), 0)
    if not _user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    elapsed_ms = (_kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return elapsed_ms / 1000.0


# ---------------------------------------------------------------- start with Windows
def startup_command() -> str:
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    runner = pythonw if pythonw.exists() else exe
    return f'"{runner}" -m faceguard tray'


def get_run_at_startup() -> str | None:
    if not IS_WINDOWS:
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, RUN_VALUE)
            return str(value)
    except OSError:
        return None


def set_run_at_startup(enabled: bool, command: str | None = None) -> None:
    """Add/remove FaceGuard from the current user's sign-in programs (HKCU Run key)."""
    if not IS_WINDOWS:
        raise OSError("Start-with-Windows is only supported on Windows")
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, command or startup_command())
        else:
            try:
                winreg.DeleteValue(key, RUN_VALUE)
            except FileNotFoundError:
                pass


def open_path(path: str | Path) -> None:
    """Open a file or folder with the default app (Explorer for folders)."""
    path = str(path)
    if IS_WINDOWS:
        os.startfile(path)  # noqa: S606 - opening our own data folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
