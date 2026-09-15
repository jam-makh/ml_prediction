"""Sealing a fitted model: the artefact, its metadata, and reading it back.

A pickle on its own is not a saved model. Six months from now the file will
load into a different library version, against a feature table that has gained
a column, and nothing in the file itself will say whether the numbers it
produces are still the numbers that were signed off. So every save writes two
files side by side:

``<name>.joblib``
    The fitted model object. Because each model owns its own preprocessing, this
    single object is the entire transformation from raw feature row to
    prediction. There is no second step to reconstruct and no chance of the
    prediction-time transform drifting away from the one used in training.

``<name>.json``
    Everything needed to judge the artefact without unpickling it: what it was
    trained on, through which month, which columns it expects, what it scored,
    and the exact library versions present when it was fitted. Readable in a
    text editor, diffable, and safe to open.

The JSON carries a SHA-256 of the model file. Not for security, for identity: it
answers "is the file next to this metadata the file this metadata describes",
which is the question that actually comes up when a directory has been copied
around a few times.

Loading checks two things and raises on neither by default. A checksum mismatch
is an error, because the pair is then meaningless. A library version mismatch is
a warning, because it is often fine and occasionally catastrophic, and the
caller is better placed than this module to decide which.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from src.models_code.base_class import Model

# Versions recorded on every save. A model unpickled against a different
# scikit-learn can load without complaint and then behave differently, which is
# the failure this list exists to make visible.
def library_versions() -> dict[str, str]:
    """Return the versions of everything the artefact depends on.

    Returns
    -------
    dict of str to str
        Library name to version string, including the Python interpreter.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, as hex.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    # Read in chunks rather than whole: a tree model over a few thousand rows
    # is small, but nothing here should assume that stays true.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelArtifact(BaseModel):
    """The metadata written beside a saved model.

    Everything here answers a question that comes up when an artefact is found
    later and its provenance is not obvious.

    Attributes
    ----------
    name : str
        The model's name, matching the filenames.
    kind : str
        Which constructor built it, a key of ``train.MODEL_REGISTRY``.
    params : dict
        Constructor arguments, so the model can be rebuilt from config alone.
    created_at : str
        UTC timestamp of the save, ISO 8601.
    trained_through : str or None
        Last training month, ``YYYY-MM``. The single most important field: it
        says which months this model is entitled to be scored on.
    n_training_months : int
        How many months it saw.
    n_training_rows : int
        How many rows it saw.
    target_column : str
        What it predicts.
    feature_columns : list of str
        The columns present at fit time, in order.
    required_columns : list of str
        The subset it actually reads at prediction time.
    data_summary : str
        One-line description of the panel it came from.
    split_summary : str
        One-line description of the split that produced its training rows.
    scores : dict
        Whatever it scored when it was sealed, normally a ``Scores`` dump.
    library_versions : dict of str to str
        Versions present at fit time.
    model_file : str
        Filename of the joblib file, relative to the metadata file.
    model_sha256 : str
        Digest of that file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str
    kind: str = "unknown"
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    trained_through: str | None = None
    n_training_months: int = 0
    n_training_rows: int = 0
    target_column: str = ""
    feature_columns: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    data_summary: str = ""
    split_summary: str = ""
    scores: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    model_file: str = ""
    model_sha256: str = ""

    def describe(self) -> str:
        """Return a one-line description for a log or a listing.

        Returns
        -------
        str
            Name, training span and creation time.
        """
        through = self.trained_through or "unfitted"
        return (
            f"{self.name} ({self.kind}): trained through {through} on "
            f"{self.n_training_rows:,} rows, saved {self.created_at}"
        )


def save_model(
    model: Model,
    output_dir: Path,
    kind: str = "unknown",
    params: dict[str, Any] | None = None,
    scores: dict[str, Any] | None = None,
    data_summary: str = "",
    split_summary: str = "",
    n_training_rows: int = 0,
) -> tuple[Path, ModelArtifact]:
    """Write a fitted model and its metadata to ``output_dir``.

    Parameters
    ----------
    model : Model
        A fitted model. Saving an unfitted one is refused, since the file would
        look exactly like a real artefact and predict nothing.
    output_dir : pathlib.Path
        Directory to write into. Created if missing.
    kind : str, optional
        Registry key that built the model, recorded for rebuilding.
    params : dict, optional
        Constructor arguments, recorded for rebuilding.
    scores : dict, optional
        What the model scored, normally ``Scores.model_dump()``.
    data_summary : str, optional
        One-line description of the panel, from ``Dataset.summary()``.
    split_summary : str, optional
        One-line description of the split, from ``Split.summary()``.
    n_training_rows : int, optional
        Rows the model was fitted on.

    Returns
    -------
    tuple of (pathlib.Path, ModelArtifact)
        Path of the joblib file, and the metadata written beside it.

    Raises
    ------
    ValueError
        If the model has not been fitted.
    """
    if not model.is_fitted:
        raise ValueError(
            f"{model.name} has not been fitted; there is nothing to save"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model.name}.joblib"
    metadata_path = output_dir / f"{model.name}.json"

    # The model goes down first, because the metadata has to describe the file
    # that actually exists, digest included.
    joblib.dump(model, model_path)

    through = model.trained_through
    artifact = ModelArtifact(
        name=model.name,
        kind=kind,
        params=params or {},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        trained_through=f"{through:%Y-%m}" if through is not None else None,
        n_training_months=len(model.training_months),
        n_training_rows=n_training_rows,
        target_column=model.target_column,
        feature_columns=list(model.feature_columns),
        required_columns=list(model.required_columns),
        data_summary=data_summary,
        split_summary=split_summary,
        scores=scores or {},
        library_versions=library_versions(),
        model_file=model_path.name,
        model_sha256=file_digest(model_path),
    )
    metadata_path.write_text(
        json.dumps(artifact.model_dump(), indent=2, default=str), encoding="utf-8"
    )
    return model_path, artifact


def load_metadata(metadata_path: Path) -> ModelArtifact:
    """Read an artefact's metadata without touching the model file.

    The cheap, safe half of loading. Unpickling executes code; reading JSON does
    not, so anything that only needs to know what an artefact is should come
    through here.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Path to the ``.json`` sidecar.

    Returns
    -------
    ModelArtifact
        The parsed metadata.

    Raises
    ------
    FileNotFoundError
        If the file is not there.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata at {metadata_path}")
    return ModelArtifact.model_validate(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )


def load_model(
    metadata_path: Path, check_versions: bool = True
) -> tuple[Model, ModelArtifact]:
    """Load a saved model and verify it matches its metadata.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Path to the ``.json`` sidecar. The model file is found beside it, from
        the name recorded in the metadata.
    check_versions : bool, optional
        Whether to warn when the current libraries differ from the ones present
        at fit time. Default True.

    Returns
    -------
    tuple of (Model, ModelArtifact)
        The loaded model and its metadata.

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    ValueError
        If the model file's digest does not match the metadata.
    """
    artifact = load_metadata(metadata_path)
    model_path = metadata_path.parent / artifact.model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Metadata at {metadata_path} names {artifact.model_file}, "
            f"which is not in {metadata_path.parent}"
        )

    # Checked before loading, not after. A mismatched pair means the metadata
    # describes something else, so its claims about training months cannot be
    # trusted either, and those are the claims that keep scoring honest.
    actual = file_digest(model_path)
    if actual != artifact.model_sha256:
        raise ValueError(
            f"{model_path.name} does not match its metadata: expected digest "
            f"{artifact.model_sha256[:12]}, found {actual[:12]}. The pair has "
            f"been mixed up, or the model file was replaced without resaving."
        )

    if check_versions:
        current = library_versions()
        drifted = {
            library: (recorded, current.get(library, "absent"))
            for library, recorded in artifact.library_versions.items()
            if current.get(library) != recorded
        }
        if drifted:
            # A warning, not an error. Often harmless, occasionally the reason
            # a model silently predicts something different than it used to.
            warnings.warn(
                f"{artifact.name} was fitted under different libraries: "
                + ", ".join(
                    f"{library} {was} -> {now}" for library, (was, now) in drifted.items()
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    model: Model = joblib.load(model_path)
    return model, artifact


def list_artifacts(output_dir: Path) -> list[ModelArtifact]:
    """Return the metadata of every artefact in a directory.

    Reads only the JSON sidecars, so listing a directory never unpickles
    anything.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to scan.

    Returns
    -------
    list of ModelArtifact
        One per readable sidecar, newest first. Run summaries and any other
        JSON that is not an artefact sidecar are skipped.
    """
    artifacts: list[ModelArtifact] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            artifacts.append(load_metadata(path))
        except Exception:
            # Not every JSON in this directory is an artefact sidecar: the run
            # summaries live here too. Skip anything that does not parse as one.
            continue
    return sorted(artifacts, key=lambda artifact: artifact.created_at, reverse=True)
