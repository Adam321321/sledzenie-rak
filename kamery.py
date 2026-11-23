"""
Lista kamer w Windowsie z ich nazwami (DirectShow).

OpenCV potrafi otworzyć kamerę tylko po numerze, ale nie poda jej nazwy. Tu pytamy o listę
ten sam systemowy mechanizm (DirectShow), z którego korzysta OpenCV z CAP_DSHOW,
więc pozycja na liście = numer kamery w OpenCV. Widać tu też kamery wirtualne,
np. „OBS Virtual Camera” (iPhone z aplikacji Kamerka).
"""
from __future__ import annotations

from ctypes import POINTER, c_int, c_ulonglong, c_void_p
from ctypes.wintypes import DWORD, ULONG

import comtypes
import comtypes.client
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
from comtypes.automation import VARIANT
from comtypes.persist import IPropertyBag

CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")


class IMoniker(IUnknown):
    _iid_ = GUID("{0000000f-0000-0000-C000-000000000046}")


class IEnumMoniker(IUnknown):
    _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")


class ICreateDevEnum(IUnknown):
    _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")


# Kolejność metod musi zgadzać się z tablicą wirtualną COM (IUnknown → IPersist → IPersistStream → IMoniker).
IMoniker._methods_ = [
    COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID), "pClassID")),
    COMMETHOD([], HRESULT, "IsDirty"),
    COMMETHOD([], HRESULT, "Load", (["in"], c_void_p, "pStm")),
    COMMETHOD([], HRESULT, "Save", (["in"], c_void_p, "pStm"), (["in"], c_int, "fClearDirty")),
    COMMETHOD([], HRESULT, "GetSizeMax", (["out"], POINTER(c_ulonglong), "pcbSize")),
    COMMETHOD([], HRESULT, "BindToObject", (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
              (["in"], POINTER(GUID), "riidResult"), (["out"], POINTER(c_void_p), "ppvResult")),
    COMMETHOD([], HRESULT, "BindToStorage", (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
              (["in"], POINTER(GUID), "riid"), (["out"], POINTER(POINTER(IUnknown)), "ppvObj")),
]

IEnumMoniker._methods_ = [
    COMMETHOD([], HRESULT, "Next", (["in"], ULONG, "celt"), (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
              (["out"], POINTER(ULONG), "pceltFetched")),
    COMMETHOD([], HRESULT, "Skip", (["in"], ULONG, "celt")),
    COMMETHOD([], HRESULT, "Reset"),
    COMMETHOD([], HRESULT, "Clone", (["out"], POINTER(POINTER(IEnumMoniker)), "ppenum")),
]

ICreateDevEnum._methods_ = [
    COMMETHOD([], HRESULT, "CreateClassEnumerator", (["in"], POINTER(GUID), "clsidDeviceClass"),
              (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"), (["in"], DWORD, "dwFlags")),
]


def lista_kamer() -> list[str]:
    """Nazwy kamer w kolejności, w jakiej OpenCV (CAP_DSHOW) nadaje im numery 0, 1, 2…"""
    try:
        comtypes.CoInitialize()
    except OSError:
        pass  # wątek ma już zainicjowany COM (np. przez Qt)
    wyliczanie = comtypes.client.CreateObject(CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    lista = wyliczanie.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, 0)
    nazwy: list[str] = []
    if not lista:  # brak jakiejkolwiek kamery
        return nazwy
    while True:
        moniker, pobrano = lista.Next(1)
        if not pobrano or not moniker:
            break
        try:
            torba = moniker.BindToStorage(None, None, IPropertyBag._iid_).QueryInterface(IPropertyBag)
            nazwy.append(str(torba.Read("FriendlyName", VARIANT(), None)))  # None zamiast VARIANT() = błąd
        except comtypes.COMError:
            nazwy.append(f"Kamera {len(nazwy) + 1}")
    return nazwy


if __name__ == "__main__":
    for i, n in enumerate(lista_kamer()):
        print(i, n)
