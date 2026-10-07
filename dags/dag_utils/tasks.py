"""One function per DAG task: each loads its inputs, logs what it did, returns XCom-safe values."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from dag_utils.artifacts import ArtifactMetadata, ArtifactStore, code_commit, model_version
from dag_utils.compare import Decision, PromotionRules, decide, first_run
from dag_utils.db import PerformanceRow, PredictionRow, ServingRepository
from dag_utils.live_performance import live_scores, refresh
from dag_utils.monitoring import MonitoringRules, check
from dag_utils.retrain import (
    build_model,
    data_hash,
    load_dataset,
    saved_params,
    split,
    xgboost_scores,
)
from dag_utils.routed_model import RoutedModel
from src.config.config import load_config
from src.data.data import Dataset, build_dataset


def _repository(config: dict[str, Any]) -> ServingRepository:
    """Return the repository with its tables guaranteed to exist.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.

    Returns
    -------
    ServingRepository
        Ready for reads and writes.
    """
    repository = ServingRepository.from_config(config)
    repository.ensure_tables()
    return repository


# --- Prediction DAG ---------------------------------------------------------------------------


def check_pending_rows() -> list[list[str]]:
    """List the feature rows newer than each user's checkpoint.

    Returns
    -------
    list of [str, str]
        ``[user_id, month ISO date]`` pairs; empty when there is no champion yet.
    """
    config = load_config()
    repository = _repository(config)
    champion = ArtifactStore.from_config(config).champion_metadata()
    if champion is None:
        logger.warning("no champion yet; run the retrain DAG first")
        return []

    # A user with no checkpoint starts after the champion's last training month.
    pending = repository.pending_rows(champion.trained_through)
    users = sorted({user for user, _ in pending})
    floor = champion.trained_through
    logger.info(f"{len(pending)} pending rows for {len(users)} users (floor {floor})")
    logger.debug(f"pending users: {users}")
    return [[user, month.isoformat()] for user, month in pending]


def run_predictions(pending: list[list[str]]) -> dict[str, Any]:
    """Predict every pending row and write each user's results and checkpoint as they finish.

    Parameters
    ----------
    pending : list of [str, str]
        Output of ``check_pending_rows``.

    Returns
    -------
    dict
        ``{"success": {user: n_rows}, "failed": [users], "skipped": [[user, month]]}``.

    Raises
    ------
    RuntimeError
        If every user failed, which points at a broken model or database rather than bad rows.
    """
    outcome: dict[str, Any] = {"success": {}, "failed": [], "skipped": []}
    if not pending:
        logger.info("nothing to predict")
        return outcome

    config = load_config()
    repository = _repository(config)
    loaded = ArtifactStore.from_config(config).load_champion()
    if loaded is None:
        raise RuntimeError("pending rows exist but no champion is saved")
    model, metadata = loaded

    keys = [(user, date.fromisoformat(month)) for user, month in pending]
    dataset = build_dataset(config, frame=repository.load_features(keys))
    routes = model.route(dataset)

    # Skipped rows have no last-month balance, so they are kept out of the model call.
    usable = np.flatnonzero((routes != "skip").to_numpy())
    predicted = np.full(dataset.n_rows, np.nan)
    if usable.size:
        predicted[usable] = model.predict(dataset.take(usable))

    frame = dataset.frame.assign(prediction=predicted, route=routes)
    for user, rows in frame.groupby(dataset.id_column, sort=True):
        try:
            served = rows.loc[rows["route"] != "skip"]
            predictions = [
                PredictionRow(
                    user_id=str(user),
                    month=row_month.date(),
                    prediction=float(value),
                    route=route,
                    model_version=metadata.model_version,
                )
                for row_month, value, route in zip(
                    served[dataset.time_column], served["prediction"], served["route"]
                )
            ]
            last_month = rows[dataset.time_column].max().date()
            repository.record_user(str(user), predictions, last_month)
        except Exception:
            # One user's failure must not stop the rest; their checkpoint stays put for the retry.
            logger.exception(f"user {user}: failed")
            outcome["failed"].append(str(user))
            continue
        skipped = rows.loc[rows["route"] == "skip", dataset.time_column]
        outcome["skipped"] += [[str(user), month.date().isoformat()] for month in skipped]
        outcome["success"][str(user)] = len(predictions)
        logger.info(
            f"user {user}: {len(predictions)} predicted, {len(skipped)} skipped, "
            f"up to {last_month}"
        )

    logger.info(
        f"success {len(outcome['success'])}, failed {len(outcome['failed'])}, "
        f"skipped rows {len(outcome['skipped'])} with {metadata.model_version}"
    )
    if outcome["failed"] and not outcome["success"]:
        raise RuntimeError(f"every user failed: {outcome['failed']}")
    return outcome


def refresh_live_performance() -> int:
    """Resolve predictions whose real balance is now known.

    Returns
    -------
    int
        Predictions newly resolved.
    """
    return refresh(_repository(load_config()))


def check_health_and_maybe_trigger_retrain() -> bool:
    """Measure the champion's live MAE decay and say whether the retrain DAG should fire.

    Returns
    -------
    bool
        True to trigger the retrain DAG now.
    """
    config = load_config()
    repository = _repository(config)
    champion = ArtifactStore.from_config(config).champion_metadata()
    if champion is None:
        logger.warning("no champion yet; nothing to monitor")
        return False

    health = check(
        errors=repository.live_errors(champion.model_version, route="xgboost"),
        baseline_mae=champion.scores["holdout"].mae,
        last_trained=repository.last_trained_at(),
        now=datetime.now(timezone.utc),
        rules=MonitoringRules.from_config(config),
    )
    log = logger.warning if health.retrain else logger.info
    log(f"health: {health.reason}; threshold {health.threshold:.0%}; retrain={health.retrain}")
    return health.retrain


# --- Retrain DAG ------------------------------------------------------------------------------


def load_data() -> str:
    """Validate the training data and return its hash.

    Returns
    -------
    str
        SHA-256 of the feature rows, passed on so later tasks prove they used the same data.
    """
    dataset = load_dataset(load_config())
    digest = data_hash(dataset.frame)
    logger.info(f"{dataset.summary()}; {dataset.frame.shape[1]} columns; hash {digest}")
    return digest


def train_test_split(expected_hash: str) -> dict[str, str]:
    """Cut the data in time and log both sides.

    Parameters
    ----------
    expected_hash : str
        Hash from ``load_data``.

    Returns
    -------
    dict of str to str
        First and last month of each side.

    Raises
    ------
    ValueError
        If the data changed since ``load_data``.
    """
    config = load_config()
    train, test = split(_checked_dataset(config, expected_hash), config)
    ranges = {
        "train": f"{train.months[0]:%Y-%m}..{train.months[-1]:%Y-%m}",
        "holdout": f"{test.months[0]:%Y-%m}..{test.months[-1]:%Y-%m}",
    }
    logger.info(
        f"train {train.n_rows:,} rows {ranges['train']}; "
        f"holdout {test.n_rows:,} rows {ranges['holdout']}"
    )
    return ranges


def train_and_save_model(expected_hash: str) -> str:
    """Refit the routed model with the saved parameters and save it as the challenger.

    Parameters
    ----------
    expected_hash : str
        Hash from ``load_data``.

    Returns
    -------
    str
        The challenger's model version.

    Raises
    ------
    ValueError
        If the data changed since ``load_data``.
    """
    config = load_config()
    train, test = split(_checked_dataset(config, expected_hash), config)
    started = datetime.now(timezone.utc)
    model = build_model(config, saved_params(config))
    model.fit(train)
    model.version = model_version(expected_hash, started)

    metadata = ArtifactMetadata(
        model_version=model.version,
        data_hash=expected_hash,
        trained_at=started,
        trained_through=train.months[-1].date(),
        commit=code_commit(),
        scores={"train": xgboost_scores(model, train), "holdout": xgboost_scores(model, test)},
    )
    folder = ArtifactStore.from_config(config).save_challenger(model, metadata)
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    holdout = metadata.scores["holdout"]
    logger.info(
        f"{model.version} trained in {seconds:.0f}s; holdout MAE {holdout.mae:,.0f}, "
        f"RMSE {holdout.rmse:,.0f}, WAPE {holdout.wape:.1f}%; saved to {folder}"
    )
    return model.version


def save_model_performance(expected_version: str) -> int:
    """Write the challenger's train and holdout scores to ``model_performance``.

    Parameters
    ----------
    expected_version : str
        Version from ``train_and_save_model``.

    Returns
    -------
    int
        Rows inserted; 0 on a rerun.
    """
    config = load_config()
    metadata = _challenger_metadata(config, expected_version)
    rows = [
        PerformanceRow(
            model_version=metadata.model_version,
            split=split_name,
            data_hash=metadata.data_hash,
            trained_at=metadata.trained_at,
            **scores.model_dump(),
        )
        for split_name, scores in metadata.scores.items()
    ]
    written = _repository(config).save_performance(rows)
    logger.info(f"{metadata.model_version}: wrote {written} rows: {metadata.scores}")
    return written


def compare_challenger_to_champion(expected_version: str) -> dict[str, str]:
    """Decide whether the challenger should replace the champion.

    Parameters
    ----------
    expected_version : str
        Version from ``train_and_save_model``.

    Returns
    -------
    dict of str to str
        The ``Decision`` as a dict, for ``promote_if_approved``.
    """
    config = load_config()
    store = ArtifactStore.from_config(config)
    challenger, challenger_meta = store.load_challenger()
    _check_version(challenger_meta, expected_version)
    champion = store.load_champion()
    if champion is None:
        decision = first_run()
    else:
        decision = _versus_champion(config, challenger, challenger_meta, *champion)
    logger.info(f"{challenger_meta.model_version}: {decision.outcome} ({decision.reason})")
    return decision.model_dump()


def promote_if_approved(decision: dict[str, str]) -> bool:
    """Copy the challenger over the champion when the comparison said so.

    Parameters
    ----------
    decision : dict of str to str
        Output of ``compare_challenger_to_champion``.

    Returns
    -------
    bool
        Whether a promotion happened.
    """
    verdict = Decision.model_validate(decision)
    if verdict.outcome != "promote":
        logger.info(f"promoted: false ({verdict.outcome}: {verdict.reason}); champion stays")
        return False
    config = load_config()
    metadata = ArtifactStore.from_config(config).promote()
    _repository(config).mark_promoted(metadata.model_version)
    logger.info(f"promoted: true ({verdict.reason})")
    return True


# --- Shared helpers ---------------------------------------------------------------------------


def _checked_dataset(config: dict[str, Any], expected_hash: str) -> Dataset:
    """Reload the data and fail if it is not what ``load_data`` hashed.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.
    expected_hash : str
        Hash from ``load_data``.

    Returns
    -------
    Dataset
        The same rows ``load_data`` saw.

    Raises
    ------
    ValueError
        If new rows landed between tasks.
    """
    dataset = load_dataset(config)
    if data_hash(dataset.frame) != expected_hash:
        raise ValueError("training data changed since load_data; rerun the DAG from the start")
    return dataset


def _challenger_metadata(config: dict[str, Any], expected_version: str) -> ArtifactMetadata:
    """Load the challenger's metadata and fail if a later retrain replaced it.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.
    expected_version : str
        Version this DAG run trained.

    Returns
    -------
    ArtifactMetadata
        The challenger's metadata.

    Raises
    ------
    ValueError
        If the saved challenger is a different version.
    """
    _, metadata = ArtifactStore.from_config(config).load_challenger()
    _check_version(metadata, expected_version)
    return metadata


def _check_version(metadata: ArtifactMetadata, expected_version: str) -> None:
    """Fail when the saved challenger is not the one this DAG run trained.

    Parameters
    ----------
    metadata : ArtifactMetadata
        Saved challenger metadata.
    expected_version : str
        Version this run trained.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a later retrain replaced the challenger.
    """
    if metadata.model_version != expected_version:
        raise ValueError(
            f"challenger is {metadata.model_version}, this run trained {expected_version}"
        )


def _versus_champion(
    config: dict[str, Any],
    challenger: RoutedModel,
    challenger_meta: ArtifactMetadata,
    champion: RoutedModel,
    champion_meta: ArtifactMetadata,
) -> Decision:
    """Score the champion live when it has enough resolved rows, else both on unseen holdout rows.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.
    challenger : RoutedModel
        Freshly trained model.
    challenger_meta : ArtifactMetadata
        Its metadata with its holdout scores.
    champion : RoutedModel
        Live model.
    champion_meta : ArtifactMetadata
        Its metadata.

    Returns
    -------
    Decision
        Outcome of ``decide``.
    """
    rules = PromotionRules.from_config(config)
    errors = _repository(config).live_errors(champion_meta.model_version, route="xgboost")
    if len(errors) >= rules.min_live_rows:
        return decide(challenger_meta.scores["holdout"], live_scores(errors), "live", rules)

    # Holdout months the champion never trained on, so neither model has seen them; both score them.
    _, test = split(load_dataset(config), config)
    unseen = np.flatnonzero((test.times.dt.date > champion_meta.trained_through).to_numpy())
    holdout = test.take(unseen)
    if holdout.n_rows == 0:
        return Decision(
            outcome="insufficient_data",
            basis="holdout",
            reason="no holdout month is new to the champion",
        )
    return decide(
        xgboost_scores(challenger, holdout), xgboost_scores(champion, holdout), "holdout", rules
    )
