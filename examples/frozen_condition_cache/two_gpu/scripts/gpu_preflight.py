"""Verify the two physical GPUs selected by the task-owned Ray cluster."""

from __future__ import annotations

import csv
import socket
import subprocess
import uuid
from typing import Any


def canonical_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        if len(value) == 16:
            return "GPU-" + str(uuid.UUID(bytes=value))
        value = value.decode("ascii")
    text = str(value).removeprefix("GPU-")
    return "GPU-" + str(uuid.UUID(text))


def check_idle(client: Any) -> None:
    """Call before ray.init, so the probe's own driver is not an active job."""
    active = [
        job.submission_id for job in client.list_jobs() if str(job.status).split(".")[-1] in ("RUNNING", "PENDING")
    ]
    if active:
        raise ValueError(f"Cluster has active jobs; inspect them without global cleanup: {active}")


def verify(ray: Any, authorized_ids: str) -> dict[str, Any]:
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if len(nodes) != 1 or nodes[0]["Resources"]["GPU"] != 2:
        raise ValueError("Use a task-owned single-node Ray cluster restricted to exactly two authorized GPUs")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            authorized_ids,
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    physical = [
        dict(
            uuid=canonical_uuid(row[0].strip()),
            name=row[1].strip(),
            memory_total_mib=int(row[2].strip()),
        )
        for row in csv.reader(output.splitlines())
        if row
    ]
    expected = {row["uuid"] for row in physical}
    if len(physical) != 2 or len(expected) != 2:
        raise ValueError("nvidia-smi must identify two distinct authorized physical GPU UUIDs")

    class GPUIdentityProbe:
        def read(self):
            import os
            import socket

            import ray
            import torch

            properties = torch.cuda.get_device_properties(0)
            identity = properties.uuid
            return dict(
                uuid=identity if isinstance(identity, (str, bytes)) else str(identity),
                name=properties.name,
                memory_total_bytes=properties.total_memory,
                hostname=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ray_gpu_ids=ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
                logical_cuda_device=0,
                torch_version=str(torch.__version__),
            )

    probe = ray.remote(num_gpus=1)(GPUIdentityProbe)
    actors = []
    try:
        # Keep both actors alive until both results arrive: they must occupy
        # different GPUs, rather than sequentially probing one idle GPU twice.
        for _ in range(2):
            actors.append(probe.remote())
        observed = ray.get([actor.read.remote() for actor in actors], timeout=120)
        for row in observed:
            row["uuid"] = canonical_uuid(row["uuid"])
        if len(observed) != 2 or {row["uuid"] for row in observed} != expected:
            raise ValueError(
                f"Ray GPU UUIDs do not match the authorized pair: expected={sorted(expected)}, observed={observed}"
            )
        return dict(
            passed=True,
            driver_hostname=socket.gethostname(),
            authorized_ids=authorized_ids,
            authorized_physical_gpus=physical,
            ray_actors=observed,
        )
    finally:
        errors = []
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError(f"Owned GPU probe cleanup failed: {errors}")
