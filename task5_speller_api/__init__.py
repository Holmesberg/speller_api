"""task5_speller_api — BCI predictive speller backend.

Public entry points:
- `predict_words(prefix, context, sentence, mental_state)` — stable function
  form used by the Unity UI (see INTEGRATION.md).
- `API` — class form for callers that want a long-lived instance or a custom
  `prediction_count`.
"""

from .speller import API, predict_words

__all__ = ["API", "predict_words"]
__version__ = "0.1.0"
