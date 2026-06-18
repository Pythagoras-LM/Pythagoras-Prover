"""Pipeline namespace."""

from importlib import import_module

from . import a2d, bert, dream, editflow, fastdllm, llada, llada2, llada21

__all__ = [
    "a2d",
    "bert",
    "dream",
    "editflow",
    "fastdllm",
    "llada",
    "llada2",
    "llada21",
    "rl",
]


def __getattr__(name):
    if name == "rl":
        module = import_module(f"{__name__}.rl")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
