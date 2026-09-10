# Request for help: NCCL over InfiniBand from SingularityCE in a Slurm GPU job

**Subject:** SingularityCE container cannot open RDMA devices in Slurm GPU job; NCCL falls back to sockets

Hello RC Help,

Could you confirm whether SingularityCE container processes in Slurm GPU jobs are expected to have `rw` access to `/dev/infiniband/uverbs*` and `/dev/infiniband/rdma_cm`? The host task can open them, but the same user's containerized process gets `errno=13 (Permission denied)` even though the devices and RDMA provider libraries are present. Is there a supported FASRC Singularity configuration, site-provided image, or device-cgroup setting required for NCCL `NET/IB`?

## Summary

I ran a two-node NCCL smoke test in Slurm job `44653138`. It allocates one GPU and one rank on each node, then runs the same RDMA verbs open probe before and inside a SingularityCE container. The job's resumed attempt used `holygpu8a16301` and `holygpu8a16602`:

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:199-204
Job ID: 44653138
Nodes: holygpu8a[16301,16602]
Rendezvous: 10.31.183.43:33138
NCCL logs: /n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/nccl-smoke-44653138
Node rank: 0; host: holygpu8a16301.rc.fas.harvard.edu; CUDA_VISIBLE_DEVICES: 0
Node rank: 1; host: holygpu8a16602.rc.fas.harvard.edu; CUDA_VISIBLE_DEVICES: 0
```

The host process can open every detected Mellanox HCA, while the same probe inside the container receives `Permission denied` for every HCA. NCCL therefore falls back from `NET/IB` to `NET/Socket`. The two-rank all-reduce completes, but it is using the socket fallback rather than RDMA.

## Job setup

The submission script starts one Slurm task per node and intentionally allows NCCL to select its own network interface/HCA; it does not force `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, or another NCCL networking override.

```bash
# submit_nccl_smoke_2gpu_2node.sh:43-55
srun \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    ...
    echo ---Host-RDMA-open-probe---
    python3 rdma_open_probe.py || true
```

It then runs the same checks in the container. The launch includes `--nv` and explicitly binds the InfiniBand device directory:

```bash
# submit_nccl_smoke_2gpu_2node.sh:57-79
exec singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    -B /tmp:/dev/shm \
    -B /dev/infiniband \
    "${PROJECT_DIR}/cur.sif" \
    bash -lc "
    ...
    echo ---Container-RDMA-open-probe---
    python3 rdma_open_probe.py || true
    ...
    ibv_devinfo -d \${RDMA_DEVICE} || true
```

The probe calls `ibv_open_device()` through `libibverbs` and prints the resulting errno:

```python
# rdma_open_probe.py:40-53
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
```

## Observations from the smoke-test logs

### 1. The host Slurm tasks can open every HCA

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:246-251
RDMA_OPEN_OK device=mlx5_0
RDMA_OPEN_OK device=mlx5_1
RDMA_OPEN_OK device=mlx5_2
RDMA_OPEN_OK device=mlx5_3
RDMA_OPEN_OK device=mlx5_4
RDMA_OPEN_OK device=mlx5_5
```

This establishes that the Slurm allocation, user identity, host RDMA devices, and host verbs stack are functional.

### 2. The container can see the RDMA character devices and HCAs

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:331-357
---Container-RDMA-diagnostics---
...
crw-rw-rw-. 1 root root  10, 125 ... rdma_cm
...
crw-rw-rw-. 1 root root 231, 192 ... uverbs0
crw-rw-rw-. 1 root root 231, 193 ... uverbs1
crw-rw-rw-. 1 root root 231, 194 ... uverbs2
crw-rw-rw-. 1 root root 231, 195 ... uverbs3
crw-rw-rw-. 1 root root 231, 196 ... uverbs4
crw-rw-rw-. 1 root root 231, 197 ... uverbs5
    device               node GUID
    ------              ----------------
    mlx5_0              8c3b4a03003a155c
    mlx5_1              8c3b4a03003a155d
    mlx5_2              b8e9240300ba8364
    mlx5_3              b8e9240300ba8368
    mlx5_4              b8e9240300ba835c
    mlx5_5              b8e9240300ba8360
```

The container has the same user UID and group memberships as the host task:

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:358-361
---Container-RDMA-security---
uid=67899(jaysonzlin) gid=11399(ydu_lab) groups=11399(ydu_lab),10242(starfish_users),10738(kempner_users),11403(kempner_ydu_lab),11432(training),34540(cluster_users),34739(seas)
         0          0 4294967295
crw-rw-rw-. 1 root root system_u:object_r:infiniband_device_t:s0 231, 192 ... /dev/infiniband/uverbs0
```

### 3. The same verbs open probe fails inside the container

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:362-368
---Container-RDMA-open-probe---
RDMA_OPEN_FAILED device=mlx5_0 errno=13 error=Permission denied
RDMA_OPEN_FAILED device=mlx5_1 errno=13 error=Permission denied
RDMA_OPEN_FAILED device=mlx5_2 errno=13 error=Permission denied
RDMA_OPEN_FAILED device=mlx5_3 errno=13 error=Permission denied
RDMA_OPEN_FAILED device=mlx5_4 errno=13 error=Permission denied
RDMA_OPEN_FAILED device=mlx5_5 errno=13 error=Permission denied
```

`ibv_devinfo` also fails in the job stderr:

```text
# logs/nccl_smoke_2gpu_2node_44653138.err:13-24
Failed to open device
Failed to open device
...
```

### 4. NCCL reports the corresponding `NET/IB` failure and socket fallback

The rank-0 NCCL debug log reports an `ibv_open_device` failure for each `mlx5` device, then explicitly selects sockets:

```text
# logs/nccl-smoke-44653138/nccl.holygpu8a16301.1110173.log:6-32
NCCL WARN Call to ibv_open_device failed
NCCL WARN NET/IB : Unable to open device mlx5_0
...
NCCL WARN NET/IB : Unable to open device mlx5_5
NCCL INFO NET/IB : No device found.
NCCL INFO NET/Socket : Using [0]ib0:10.31.183.43<0>
NCCL INFO Using network Socket
```

The channel setup confirms that data communication used sockets:

```text
# logs/nccl-smoke-44653138/nccl.holygpu8a16301.1110173.log:61-70
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0
NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Socket/0
```

### 5. The collective completes, but over the fallback transport

```text
# logs/nccl_smoke_2gpu_2node_44653138.out:405-407
NCCL version 2.20.5+cuda12.4
rank=1 host=holygpu8a16602.rc.fas.harvard.edu allreduce_256MiB=0.1958s logical_bw=1.31 GB/s
rank=0 host=holygpu8a16301.rc.fas.harvard.edu allreduce_256MiB=0.1958s logical_bw=1.31 GB/s
```

Therefore, the two nodes can form an NCCL process group and run a collective, but this does not validate RDMA: the NCCL debug logs establish that it used `NET/Socket`.

## Additional in-container checks

These checks were run interactively in the same container image after inspecting the smoke logs.

The mlx5 verbs provider is present, so this is not explained by a missing provider library:

```text
/etc/libibverbs.d/mlx5.driver
/usr/lib/x86_64-linux-gnu/libmlx5.so.1.22.39.0
/usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav34.so
/usr/lib/x86_64-linux-gnu/libmlx5.so.1
```

The image definition also installs the expected RDMA userspace packages:

```text
# current.def:15-20
libibverbs1
ibverbs-providers
librdmacm1
libnl-3-200
libnl-route-3-200
ibverbs-utils
```

Finally, a direct Python `os.open(path, O_RDWR)` check confirms the denial occurs at the RDMA character-device boundary, before NCCL can establish a verbs context:

```text
OPEN_FAILED /dev/infiniband/uverbs2: errno=13 Permission denied
OPEN_FAILED /dev/infiniband/uverbs1: errno=13 Permission denied
OPEN_FAILED /dev/infiniband/uverbs0: errno=13 Permission denied
OPEN_FAILED /dev/infiniband/rdma_cm: errno=13 Permission denied
```

## Requested guidance

Could you please advise on the supported way to run NCCL over InfiniBand/RDMA from SingularityCE on FASRC? In particular:

1. Are container processes in Slurm GPU jobs expected to have `rw` access to `/dev/infiniband/uverbs*` and `/dev/infiniband/rdma_cm`?
2. Is a particular SingularityCE invocation, site-provided image, or additional bind configuration required?
3. Does FASRC apply a device-cgroup or other policy that requires enabling the RDMA character devices for container processes? The visible device majors/minors include `uverbs*` at major `231` and `rdma_cm` at major `10`, minor `125`.

The desired validation is that `ibv_devinfo -d mlx5_0` and the verbs open probe succeed inside the container. NCCL should then report `NET/IB` and channels such as `via NET/IB/...`, rather than `NET/Socket/0`.

Thank you,

Jayson
