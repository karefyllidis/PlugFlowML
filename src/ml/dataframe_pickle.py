"""
Portable pickle I/O for training DataFrames.

``StringDtype`` and some Arrow-backed columns pickle in a way that breaks when
loading with another pandas minor version. ``save_dataframe_pickle`` coerces
those columns to plain ``object`` dtype before saving so the resulting ``.pkl``
loads cleanly across pandas/Python versions.

On load, if ``pd.read_pickle`` raises a dtype-compat error the fallback chain is:
  1. Sibling ``.csv`` (legacy runs that set ``save_complete_csv=True``).
  2. Clear ``RuntimeError`` — re-run Main_2 to regenerate a portable ``.pkl``.

**Root cause of the old file:** the ``training_data_complete_*.pkl`` written
before this module existed was saved with plain ``pd.DataFrame.to_pickle`` /
``pickle.dump``, preserving the raw ``StringDtype`` backing.  pandas ≥2.3
(issue #61763) cannot unpickle that format.  The fix is to re-run Main_2 so
``save_dataframe_pickle`` writes a new, coerced file.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def _is_pickle_compat_error(exc: BaseException) -> bool:
    """True for dtype / version mismatch during unpickle (not generic bugs)."""
    if isinstance(exc, NotImplementedError):
        # pandas NDArrayBacked.__setstate__ + old StringDtype (pandas ≥2.3 / py3.14)
        text = str(exc)
        if "StringDtype" in text or "stringdtype" in text.lower():
            return True
    msg = str(exc).lower()
    return any(
        x in msg
        for x in (
            "stringdtype",
            "issubclass",
            "unpickl",
            "positional arguments",
            "__nat_unpickle",
            "can't get attribute",
        )
    )


def coerce_dataframe_for_pickle(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce extension dtypes that often break unpickling across pandas versions."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        d = s.dtype
        if isinstance(d, pd.StringDtype):
            out[col] = s.astype(object)
        elif hasattr(pd, "ArrowDtype") and isinstance(d, pd.ArrowDtype):
            try:
                import pyarrow as pa

                dt = d.pyarrow_dtype
                if pa.types.is_string(dt) or pa.types.is_large_string(dt):
                    out[col] = s.astype(object)
            except Exception:
                out[col] = s.astype(object)
        elif getattr(d, "name", None) == "string":
            out[col] = s.astype(object)
    return out


def save_dataframe_pickle(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    protocol: Optional[int] = None,
) -> None:
    path = Path(path)
    payload = coerce_dataframe_for_pickle(df)
    proto = protocol if protocol is not None else pickle.HIGHEST_PROTOCOL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=proto)


def coerce_pickle_payload(obj: object) -> object:
    """Recursively coerce DataFrames inside dicts / sequences for stable pickling."""
    if isinstance(obj, pd.DataFrame):
        return coerce_dataframe_for_pickle(obj)
    if isinstance(obj, dict):
        return {k: coerce_pickle_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_pickle_payload(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(coerce_pickle_payload(x) for x in obj)
    return obj


def save_pickle_portable(
    obj: object,
    path: Union[str, Path],
    *,
    protocol: Optional[int] = None,
) -> None:
    """Pickle arbitrary object, coercing nested DataFrames first."""
    path = Path(path)
    payload = coerce_pickle_payload(obj)
    proto = protocol if protocol is not None else pickle.HIGHEST_PROTOCOL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=proto)


def load_portable_pickle(path: Union[str, Path]) -> object:
    """``pd.read_pickle`` with clear errors for pandas version / dtype mismatches."""
    path = Path(path)
    try:
        return pd.read_pickle(path)
    except (
        TypeError,
        ModuleNotFoundError,
        AttributeError,
        pickle.UnpicklingError,
        NotImplementedError,
    ) as e:
        if _is_pickle_compat_error(e):
            raise RuntimeError(
                f"Could not read {path}: pickle saved with an older pandas/StringDtype "
                f"format (pandas issue #61763). Re-run Main_2 to regenerate a portable .pkl."
            ) from e
        raise


def load_dataframe_pickle(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single-DataFrame pickle.

    If ``pd.read_pickle`` raises a pandas dtype/version compat error, the
    fallback chain is:

    1. Sibling ``.csv`` — if a legacy Main_2 run wrote one (``save_complete_csv=True``).
    2. ``RuntimeError`` with clear instructions to re-run Main_2.

    The permanent fix for old pickles is to re-run Main_2, which calls
    ``save_dataframe_pickle`` — this coerces ``StringDtype`` → ``object``
    before pickling, making the file loadable on any pandas/Python version.
    """
    path = Path(path)
    try:
        obj = pd.read_pickle(path)
    except (
        TypeError,
        ModuleNotFoundError,
        AttributeError,
        pickle.UnpicklingError,
        NotImplementedError,
    ) as e:
        if not _is_pickle_compat_error(e):
            raise

        # Sibling CSV fallback (legacy runs with save_complete_csv=True)
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            warnings.warn(
                f"Pickle unreadable ({path.name}); loaded sibling {csv_path.name} instead. "
                f"Re-run Main_2 to regenerate a portable .pkl (faster + exact dtypes).",
                UserWarning,
                stacklevel=2,
            )
            return pd.read_csv(csv_path, low_memory=False)

        raise RuntimeError(
            f"\n{'='*70}\n"
            f"Cannot load: {path.name}\n"
            f"{'='*70}\n"
            f"This pickle was saved with an older pandas version that stored string\n"
            f"columns as StringDtype — pandas ≥2.3 cannot unpickle that format\n"
            f"(pandas issue #61763).\n\n"
            f"FIX → Re-run Main_2 to regenerate the training data file.\n"
            f"The new .pkl will be saved via save_dataframe_pickle() which coerces\n"
            f"StringDtype → object dtype before pickling and loads on any version.\n"
            f"{'='*70}"
        ) from e

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj).__name__}")
    return obj
