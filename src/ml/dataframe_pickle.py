"""
Portable pickle I/O for training DataFrames.

``StringDtype`` and some Arrow-backed columns pickle in a way that breaks when
loading with another pandas minor version. We coerce those columns before save.

On load, if unpickling fails and a legacy sibling ``training_data_complete_*.csv``
exists (from an older run when CSV was enabled), that file is used automatically.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _is_pickle_compat_error(exc: BaseException) -> bool:
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
    """``pd.read_pickle`` with clearer errors for pandas version / dtype mismatches."""
    path = Path(path)
    try:
        return pd.read_pickle(path)
    except (TypeError, ModuleNotFoundError, AttributeError, pickle.UnpicklingError) as e:
        if _is_pickle_compat_error(e):
            raise RuntimeError(
                f"Could not read {path}: the pickle was likely saved with a different "
                f"pandas version (extension dtypes like StringDtype are fragile across "
                f"minor releases). Fix: (1) `pip install -U pandas` to match the writer; "
                f"(2) if this is training data from Main_2, load via `load_dataframe_pickle` "
                f"which falls back to a sibling `.csv` when present; or (3) re-run Main_2 "
                f"so a new `.pkl` is written with portable dtypes."
            ) from e
        raise


def load_dataframe_pickle(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single-DataFrame pickle.

    If unpickling fails (pandas version / StringDtype), and Main_2 wrote a CSV next to
    the pickle (``training_data_complete_*.csv``), that file is loaded instead.
    """
    path = Path(path)
    try:
        obj = pd.read_pickle(path)
    except (TypeError, ModuleNotFoundError, AttributeError, pickle.UnpicklingError) as e:
        if not _is_pickle_compat_error(e):
            raise
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            warnings.warn(
                f"Pickle unreadable ({path.name}); loaded {csv_path.name} instead. "
                f"For faster loads and exact dtypes, run: pip install -U pandas "
                f"to match the environment that created the pickle, or re-run Main_2.",
                UserWarning,
                stacklevel=2,
            )
            return pd.read_csv(csv_path, low_memory=False)
        raise RuntimeError(
            f"Could not read {path} and no sibling CSV was found at {csv_path}. "
            f"Try `pip install -U pandas`, or re-run Main_2 to regenerate both `.pkl` and `.csv`."
        ) from e

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj).__name__}")
    return obj
