from cupyram.cupyram import CuPyRAM
from cupyram.pade import compute_pade_coefficients
from cupyram.batching import AsyncBatcher

try:
    from ._version import __version__
except ImportError:
    # Fallback in case package is not installed/built yet (e.g. raw git clone)
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("cupyram")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"

__all__ = [
    'CuPyRAM', 
    'compute_pade_coefficients',
    'AsyncBatcher',
]
