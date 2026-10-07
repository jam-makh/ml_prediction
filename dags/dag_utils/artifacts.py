"""Champion and challenger artifacts: the model file, its metadata.json and atomic promotion."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

from dag_utils.routed_model import RoutedModel
from src.config.config import PROJECT_ROOT
from src.models_code.base_class import Model

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class SplitScores(BaseModel):
    """Error metrics of one model on one set of rows.

    Parameters
    ----------
    mae, rmse : float
        Mean absolute and root mean squared error, in USD.
    wape : float
        Weighted absolute percentage error, in percent.
    n_rows : int
        Rows scored.
    """

    mae: float
    rmse: float
    wape: float
    n_rows: int


class ArtifactMetadata(BaseModel):
    """What produced a saved model: its version, data, code and scores.

    Parameters
    ----------
    model_version : str
        Version tag written on every prediction the model makes.
    data_hash : str
        SHA-256 of the training data.
    trained_at : datetime.datetime
        When the model was fitted (UTC).
    trained_through : datetime.date
        Last month in the training rows.
    commit : str
        Code commit that trained it, ``unknown`` when unavailable.
    scores : dict of str to SplitScores
        Scores on the ``train`` and ``holdout`` rows routed to xgboost.
    """

    model_version: str
    data_hash: str
    trained_at: datetime
    trained_through: date
    commit: str
    scores: dict[Literal["train", "holdout"], SplitScores]


def model_version(data_hash: str, trained_at: datetime) -> str:
    """Build a readable, unique version tag from the fit date and the data hash.

    Parameters
    ----------
    data_hash : str
        SHA-256 of the training data.
    trained_at : datetime.datetime
        When the model was fitted.

    Returns
    -------
    str
        e.g. ``xgb-20261007T0600-1a2b3c4d``.
    """
    return f"xgb-{trained_at:%Y%m%dT%H%M}-{data_hash[:8]}"


def code_commit() -> str:
    """Return the current commit from ``GIT_COMMIT`` or git, else ``unknown``.

    Returns
    -------
    str
        Short commit hash, or ``unknown`` inside a container without git.
    """
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


class ArtifactStore:
    """Reads and writes the champion and challenger folders under one root.

    Parameters
    ----------
    root : pathlib.Path
        Folder holding ``champion/`` and ``challenger/``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.champion_dir = root / "champion"
        self.challenger_dir = root / "challenger"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ArtifactStore:
        """Build the store from ``serving.artifact_dir``, relative to the project root.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        ArtifactStore
            Store rooted at the configured folder.
        """
        return cls(PROJECT_ROOT / str(config["serving"]["artifact_dir"]))

    def save_challenger(self, model: RoutedModel, metadata: ArtifactMetadata) -> Path:
        """Write a challenger, replacing the previous one; the champion is never touched.

        Parameters
        ----------
        model : RoutedModel
            Fitted challenger.
        metadata : ArtifactMetadata
            Its provenance and scores.

        Returns
        -------
        pathlib.Path
            The challenger folder.
        """
        self.challenger_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.challenger_dir / MODEL_FILE)
        (self.challenger_dir / METADATA_FILE).write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        return self.challenger_dir

    def load_challenger(self) -> tuple[RoutedModel, ArtifactMetadata]:
        """Load the current challenger.

        Returns
        -------
        tuple of (RoutedModel, ArtifactMetadata)
            The model and its metadata.

        Raises
        ------
        FileNotFoundError
            If no challenger has been saved.
        """
        return self._load(self.challenger_dir)

    def load_champion(self) -> tuple[RoutedModel, ArtifactMetadata] | None:
        """Load the live model, or None before the first promotion.

        Returns
        -------
        tuple of (RoutedModel, ArtifactMetadata) or None
            The champion and its metadata.
        """
        if not (self.champion_dir / METADATA_FILE).exists():
            return None
        return self._load(self.champion_dir)

    def champion_metadata(self) -> ArtifactMetadata | None:
        """Read only the champion's metadata, without unpickling the model.

        Returns
        -------
        ArtifactMetadata or None
            None before the first promotion.
        """
        path = self.champion_dir / METADATA_FILE
        if not path.exists():
            return None
        return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def promote(self) -> ArtifactMetadata:
        """Copy the challenger over the champion, one atomic file swap at a time.

        Returns
        -------
        ArtifactMetadata
            Metadata of the new champion.

        Raises
        ------
        FileNotFoundError
            If no challenger has been saved.
        """
        _, metadata = self.load_challenger()
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        # Model first, metadata last: the version check in _load catches a crash between the two.
        for name in (MODEL_FILE, METADATA_FILE):
            staged = self.champion_dir / f"{name}.tmp"
            shutil.copy2(self.challenger_dir / name, staged)
            os.replace(staged, self.champion_dir / name)
        logger.info(f"promoted {metadata.model_version} to {self.champion_dir}")
        return metadata

    def _load(self, folder: Path) -> tuple[RoutedModel, ArtifactMetadata]:
        """Load the model and metadata of one folder and check they describe each other.

        Parameters
        ----------
        folder : pathlib.Path
            Champion or challenger folder.

        Returns
        -------
        tuple of (RoutedModel, ArtifactMetadata)
            The model and its metadata.

        Raises
        ------
        TypeError
            If the file holds a model other than a RoutedModel.
        ValueError
            If the model's version differs from the metadata's.
        """
        metadata = ArtifactMetadata.model_validate_json(
            (folder / METADATA_FILE).read_text(encoding="utf-8")
        )
        model: Model = Model.load(folder / MODEL_FILE)
        if not isinstance(model, RoutedModel):
            raise TypeError(f"{folder / MODEL_FILE} holds a {type(model).__name__}")
        # A promotion that crashed between the two swaps leaves mismatched versions.
        if model.version != metadata.model_version:
            raise ValueError(
                f"{folder}: model is {model.version!r}, metadata is "
                f"{metadata.model_version!r}; rerun the promotion"
            )
        return model, metadata
