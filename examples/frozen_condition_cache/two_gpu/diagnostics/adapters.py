"""Use these explicit adapters only for the single-update diagnostic jobs."""

import os

import torch

from relax.models.qwen_image.sft_adapter import QwenImageEditSFTAdapter
from relax.models.sensenova.adapter import SenseNovaSFTAdapter

from .ray_hook import install


def activate() -> None:
    if not os.environ.get("DIAG_ROOT") or not os.environ.get("DIAG_PHASE"):
        raise ValueError("Diagnostic adapters require DIAG_ROOT and DIAG_PHASE")
    deterministic = os.environ.get("DIAG_DETERMINISTIC", "1")
    if deterministic not in {"0", "1"}:
        raise ValueError("DIAG_DETERMINISTIC must be 0 or 1")
    if deterministic == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    install()


class SenseNovaDiagnosticAdapter(SenseNovaSFTAdapter):
    _relax_single_update_diagnostic = True

    def __init__(self):
        activate()


class QwenImageEditDiagnosticAdapter(QwenImageEditSFTAdapter):
    _relax_single_update_diagnostic = True

    def __init__(self):
        activate()
