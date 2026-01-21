"""
ElectrolyserLib - Python library for hydrogen production calculations
based on PEM electrolyser models.
"""

from .pem_electrolyser import (
    Electrolyser,
    DynamicElectrolyser,
    DEFAULT_EFFICIENCY_CURVE
)

__version__ = "0.1.0"
__author__ = "moend95"

__all__ = [
    "Electrolyser",
    "DynamicElectrolyser",
    "DEFAULT_EFFICIENCY_CURVE",
]
