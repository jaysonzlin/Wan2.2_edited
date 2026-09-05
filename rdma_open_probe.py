#!/usr/bin/env python3
"""Report the errno from opening every RDMA device through libibverbs."""

import ctypes
import os


def main() -> int:
    try:
        verbs = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    except OSError as error:
        print(f"RDMA_OPEN_LOAD_FAILED error={error}")
        return 1

    verbs.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    verbs.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    verbs.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    verbs.ibv_free_device_list.restype = None
    verbs.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    verbs.ibv_get_device_name.restype = ctypes.c_char_p
    verbs.ibv_open_device.argtypes = [ctypes.c_void_p]
    verbs.ibv_open_device.restype = ctypes.c_void_p
    verbs.ibv_close_device.argtypes = [ctypes.c_void_p]
    verbs.ibv_close_device.restype = ctypes.c_int

    device_count = ctypes.c_int()
    devices = verbs.ibv_get_device_list(ctypes.byref(device_count))
    if not devices:
        error_number = ctypes.get_errno()
        print(
            "RDMA_OPEN_LIST_FAILED "
            f"errno={error_number} error={os.strerror(error_number)}"
        )
        return 1

    failures = 0
    try:
        for index in range(device_count.value):
            device = devices[index]
            device_name = verbs.ibv_get_device_name(device).decode()
            ctypes.set_errno(0)
            context = verbs.ibv_open_device(device)
            if context:
                print(f"RDMA_OPEN_OK device={device_name}")
                verbs.ibv_close_device(context)
                continue

            error_number = ctypes.get_errno()
            print(
                "RDMA_OPEN_FAILED "
                f"device={device_name} errno={error_number} "
                f"error={os.strerror(error_number)}"
            )
            failures += 1
    finally:
        verbs.ibv_free_device_list(devices)

    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
