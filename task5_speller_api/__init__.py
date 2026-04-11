"""task5_speller_api — BCI predictive speller backend.

Public entry point is `predict_words`. See `speller.py` for the full contract
and `INTEGRATION.md` for the UI-side integration guide.
"""

from .speller import API

__all__ = ["predict_words"]
__version__ = "0.1.0"
