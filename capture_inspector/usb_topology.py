"""Walk the Windows device tree via cfgmgr32 to work out how a capture device is attached.

The interesting question for a USB capture card is not "is there a driver" but
"did it negotiate SuperSpeed".  A UVC capture card plugged into a USB 2.0 path
silently advertises a tiny format list instead of failing, so the only way to
explain a missing 1080p60 option is to follow the device up to its host
controller and look at what kind of controller it is.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import Enum

_cfgmgr = ctypes.WinDLL("cfgmgr32")

CR_SUCCESS = 0
MAX_DEVICE_ID_LEN = 200

# CM_DRP_* registry property ids (1-based, unlike SPDRP_*).
CM_DRP_DEVICEDESC = 0x01
CM_DRP_SERVICE = 0x05
CM_DRP_FRIENDLYNAME = 0x0D
CM_DRP_LOCATION_INFORMATION = 0x0E

_cfgmgr.CM_Locate_DevNodeW.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.LPCWSTR, wintypes.ULONG]
_cfgmgr.CM_Locate_DevNodeW.restype = wintypes.DWORD

_cfgmgr.CM_Get_Parent.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, wintypes.ULONG]
_cfgmgr.CM_Get_Parent.restype = wintypes.DWORD

_cfgmgr.CM_Get_Device_IDW.argtypes = [wintypes.DWORD, wintypes.LPWSTR, wintypes.ULONG, wintypes.ULONG]
_cfgmgr.CM_Get_Device_IDW.restype = wintypes.DWORD

_cfgmgr.CM_Get_DevNode_Registry_PropertyW.argtypes = [
    wintypes.DWORD,
    wintypes.ULONG,
    ctypes.POINTER(wintypes.ULONG),
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.ULONG),
    wintypes.ULONG,
]
_cfgmgr.CM_Get_DevNode_Registry_PropertyW.restype = wintypes.DWORD


class LinkSpeed(Enum):
    """How fast the path from the device to the host controller can possibly run."""

    SUPERSPEED = "USB 3.x (SuperSpeed)"
    HIGHSPEED = "USB 2.0 (High-Speed)"
    UNKNOWN = "unknown"


@dataclass
class DeviceNode:
    instance_id: str
    description: str
    friendly_name: str = ""
    location: str = ""

    @property
    def label(self) -> str:
        return self.friendly_name or self.description or self.instance_id


@dataclass
class Topology:
    """The chain from a device up to its PCI host controller."""

    chain: list[DeviceNode] = field(default_factory=list)

    @property
    def host_controller(self) -> DeviceNode | None:
        for node in reversed(self.chain):
            if node.instance_id.upper().startswith("PCI\\"):
                return node
        return None

    @property
    def hub_count(self) -> int:
        """Physical hubs between the device and the port. Root hubs are part of
        the controller, not something the user plugged in, so they don't count."""
        count = 0
        for node in self.chain[1:]:
            if "ROOT_HUB" in node.instance_id.upper():
                continue
            if "hub" in node.label.lower() or "ハブ" in node.label:
                count += 1
        return count

    @property
    def usb_chain(self) -> list[DeviceNode]:
        """Just the USB portion, ending at the host controller."""
        out: list[DeviceNode] = []
        for node in self.chain:
            out.append(node)
            if node.instance_id.upper().startswith("PCI\\"):
                break
        return out

    @property
    def link_speed(self) -> LinkSpeed:
        """Classify the path by the controller and root hub it terminates at.

        xHCI ("eXtensible") controllers carry both USB 2 and USB 3 devices, so a
        device sitting under one is only *capable* of SuperSpeed.  An EHCI
        ("Enhanced") controller or a ROOT_HUB20 node is decisive the other way:
        nothing below it can exceed 480 Mbps.
        """
        for node in self.chain:
            upper = node.instance_id.upper()
            if "ROOT_HUB30" in upper:
                return LinkSpeed.SUPERSPEED
            if "ROOT_HUB20" in upper:
                return LinkSpeed.HIGHSPEED

        controller = self.host_controller
        if controller is None:
            return LinkSpeed.UNKNOWN

        name = controller.label.lower()
        if "extensible" in name or "xhci" in name:
            return LinkSpeed.SUPERSPEED
        if "enhanced" in name or "ehci" in name or "usb2" in name:
            return LinkSpeed.HIGHSPEED
        return LinkSpeed.UNKNOWN


def _get_property(devinst: int, prop_id: int) -> str:
    buf_len = wintypes.ULONG(0)
    prop_type = wintypes.ULONG(0)
    _cfgmgr.CM_Get_DevNode_Registry_PropertyW(devinst, prop_id, ctypes.byref(prop_type), None, ctypes.byref(buf_len), 0)
    if buf_len.value == 0:
        return ""
    buf = ctypes.create_unicode_buffer(buf_len.value // 2 + 1)
    rc = _cfgmgr.CM_Get_DevNode_Registry_PropertyW(
        devinst, prop_id, ctypes.byref(prop_type), buf, ctypes.byref(buf_len), 0
    )
    return buf.value if rc == CR_SUCCESS else ""


def _node_from_devinst(devinst: int) -> DeviceNode:
    buf = ctypes.create_unicode_buffer(MAX_DEVICE_ID_LEN + 1)
    _cfgmgr.CM_Get_Device_IDW(devinst, buf, MAX_DEVICE_ID_LEN, 0)
    return DeviceNode(
        instance_id=buf.value,
        description=_get_property(devinst, CM_DRP_DEVICEDESC),
        friendly_name=_get_property(devinst, CM_DRP_FRIENDLYNAME),
        location=_get_property(devinst, CM_DRP_LOCATION_INFORMATION),
    )


_cfgmgr.CM_Get_Device_ID_List_SizeW.argtypes = [
    ctypes.POINTER(wintypes.ULONG),
    wintypes.LPCWSTR,
    wintypes.ULONG,
]
_cfgmgr.CM_Get_Device_ID_List_SizeW.restype = wintypes.DWORD

_cfgmgr.CM_Get_Device_ID_ListW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.ULONG,
    wintypes.ULONG,
]
_cfgmgr.CM_Get_Device_ID_ListW.restype = wintypes.DWORD

CM_GETIDLIST_FILTER_ENUMERATOR = 0x00000001


def _enumerate(enumerator: str = "USB") -> list[str]:
    size = wintypes.ULONG(0)
    if (
        _cfgmgr.CM_Get_Device_ID_List_SizeW(ctypes.byref(size), enumerator, CM_GETIDLIST_FILTER_ENUMERATOR)
        != CR_SUCCESS
        or size.value == 0
    ):
        return []
    buf = ctypes.create_unicode_buffer(size.value)
    if _cfgmgr.CM_Get_Device_ID_ListW(enumerator, buf, size.value, CM_GETIDLIST_FILTER_ENUMERATOR) != CR_SUCCESS:
        return []
    return [chunk for chunk in buf[: size.value].split("\0") if chunk]


def find_instances_by_name(name: str, enumerator: str = "USB") -> list[DeviceNode]:
    """Find USB device nodes whose friendly name or description matches `name`.

    DirectShow only hands us a display name, so this is how we cross over from
    the capture API's view of the device to the device tree's view of it.
    """
    wanted = name.strip().lower()
    matches: list[DeviceNode] = []
    for instance_id in _enumerate(enumerator):
        devinst = wintypes.DWORD()
        if _cfgmgr.CM_Locate_DevNodeW(ctypes.byref(devinst), instance_id, 0) != CR_SUCCESS:
            continue
        node = _node_from_devinst(devinst.value)
        haystack = f"{node.friendly_name} {node.description}".lower()
        if wanted and wanted in haystack:
            matches.append(node)
    return matches


def list_usb3_controllers() -> list[DeviceNode]:
    """xHCI controllers present on this machine.

    If the card is stuck on USB 2.0 but the machine has an xHCI controller,
    the fix is just a different port - worth saying so explicitly.
    """
    found: list[DeviceNode] = []
    for instance_id in _enumerate("PCI"):
        devinst = wintypes.DWORD()
        if _cfgmgr.CM_Locate_DevNodeW(ctypes.byref(devinst), instance_id, 0) != CR_SUCCESS:
            continue
        node = _node_from_devinst(devinst.value)
        name = node.label.lower()
        if ("extensible" in name or "xhci" in name) and "host controller" in name:
            found.append(node)
    return found


def get_service(instance_id: str) -> str:
    """Return the kernel driver servicing a device (e.g. 'usbvideo' for stock UVC)."""
    devinst = wintypes.DWORD()
    if _cfgmgr.CM_Locate_DevNodeW(ctypes.byref(devinst), instance_id, 0) != CR_SUCCESS:
        return ""
    return _get_property(devinst.value, CM_DRP_SERVICE)


def trace(instance_id: str, max_depth: int = 16) -> Topology:
    """Return the chain of devices from `instance_id` up to the host controller."""
    devinst = wintypes.DWORD()
    if _cfgmgr.CM_Locate_DevNodeW(ctypes.byref(devinst), instance_id, 0) != CR_SUCCESS:
        return Topology()

    chain: list[DeviceNode] = []
    current = devinst.value
    for _ in range(max_depth):
        chain.append(_node_from_devinst(current))
        parent = wintypes.DWORD()
        if _cfgmgr.CM_Get_Parent(ctypes.byref(parent), current, 0) != CR_SUCCESS:
            break
        current = parent.value
        # The ACPI/HAL root is above the host controller and tells us nothing.
        probe = _node_from_devinst(current)
        if probe.instance_id.upper().startswith(("ACPI_HAL", "HTREE")):
            break

    return Topology(chain=chain)
