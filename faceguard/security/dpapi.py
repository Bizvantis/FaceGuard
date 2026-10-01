"""Windows Data Protection API (DPAPI) via ctypes.

Face templates are biometric data.  FaceGuard encrypts the database with
``CryptProtectData``, which derives the key from the logged-in Windows account:
the file cannot be decrypted by other users or on another machine, and no
password needs to be stored anywhere.
"""

from __future__ import annotations

import ctypes
import sys

_ENTROPY = b"FaceGuard/face-db/v1"
CRYPTPROTECT_UI_FORBIDDEN = 0x01


def available() -> bool:
    return sys.platform == "win32"


if sys.platform == "win32":
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.POINTER(_DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    def _to_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
        buf = (ctypes.c_ubyte * max(1, len(data))).from_buffer_copy(data or b"\0")
        return _DataBlob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))), buf

    def _take(blob: _DataBlob) -> bytes:
        try:
            return ctypes.string_at(blob.pbData, blob.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))

    def protect(data: bytes, entropy: bytes = _ENTROPY) -> bytes:
        src, _keep1 = _to_blob(data)
        ent, _keep2 = _to_blob(entropy)
        out = _DataBlob()
        if not _crypt32.CryptProtectData(
            ctypes.byref(src), "FaceGuard", ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out)
        ):
            raise OSError(ctypes.get_last_error(), "CryptProtectData failed")
        return _take(out)

    def unprotect(blob: bytes, entropy: bytes = _ENTROPY) -> bytes:
        src, _keep1 = _to_blob(blob)
        ent, _keep2 = _to_blob(entropy)
        out = _DataBlob()
        if not _crypt32.CryptUnprotectData(
            ctypes.byref(src), None, ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out)
        ):
            raise OSError(ctypes.get_last_error(), "CryptUnprotectData failed (different Windows user or corrupted data?)")
        return _take(out)

else:  # pragma: no cover - non-Windows fallback keeps the rest of the app usable

    def protect(data: bytes, entropy: bytes = _ENTROPY) -> bytes:
        raise OSError("DPAPI is only available on Windows")

    def unprotect(blob: bytes, entropy: bytes = _ENTROPY) -> bytes:
        raise OSError("DPAPI is only available on Windows")
