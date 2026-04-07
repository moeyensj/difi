"""difi: Did I Find It?

Evaluate linkage completeness and purity for astronomical surveys.

v3 is a Rust rewrite with Python bindings via PyO3/maturin.
"""

from difi._core import version

__version__ = version()

__all__ = [
    "version",
]
