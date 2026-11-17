"""Optional Rust sampling for Ray360."""

from typing import TYPE_CHECKING

from ._sampler import sample as sample

if TYPE_CHECKING:
    from ._prepared import PreparedView as PreparedView

__all__ = ["PreparedView", "sample"]


def __getattr__(name):
    if name == "PreparedView":
        try:
            from ._prepared import PreparedView
        except ModuleNotFoundError as error:
            if error.name != "ray360":
                raise
            raise ImportError("Install ray360 to use PreparedView.") from error

        globals()[name] = PreparedView
        return PreparedView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
