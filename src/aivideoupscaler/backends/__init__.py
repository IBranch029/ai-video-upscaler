"""Super-resolution backends.

Each backend implements :class:`~aivideoupscaler.backends.base.UpscalerBackend`
so the pipeline can swap between them transparently.
"""

from .base import BackendStatus, UpscalerBackend
from .classical import ClassicalBackend
from .ncnn import NcnnBackend, find_ncnn_executable, find_ncnn_models_dir, install_ncnn_bundle
from .torch_esrgan import TorchBackend, detect_device, torch_available

__all__ = [
    "BackendStatus",
    "ClassicalBackend",
    "NcnnBackend",
    "TorchBackend",
    "UpscalerBackend",
    "detect_device",
    "find_ncnn_executable",
    "find_ncnn_models_dir",
    "install_ncnn_bundle",
    "torch_available",
]
