from __future__ import annotations

import copy
import csv
import hashlib
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Demo mode (default) runs the full interface on the bundled 10-image set and
# never stores any data. Set DEMO_MODE=0 to run the batched study, which needs
# your own stimulus manifest and a Supabase project.
DEMO_MODE = _env_bool("DEMO_MODE", True)


def _stimuli_csv_path() -> Path:
    explicit = os.getenv("STIMULI_CSV", "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else BASE_DIR / path
    if DEMO_MODE:
        return BASE_DIR / "demo" / "stimuli.csv"
    return BASE_DIR / "stimuli" / "stimuli.csv"


STIM_CSV = _stimuli_csv_path()
N_REPS = 1
BREAK_EVERY = 50
DEFAULT_RATING = 5
BATCH_COUNT = 3
ANCHOR_TRIAL_COUNT = 15
UNIQUE_TRIALS_PER_BATCH = 93
QC_REPEAT_COUNT = 2
BASE_TRIALS_PER_BATCH = ANCHOR_TRIAL_COUNT + UNIQUE_TRIALS_PER_BATCH
TRIALS_PER_BATCH = BASE_TRIALS_PER_BATCH + QC_REPEAT_COUNT
TRIAL_ASSIGNMENT_VERSION = "anchor15_unique93_qc2_v1"

QC_MIN_GAP = max(1, _env_int("QC_REPEAT_MIN_GAP", 20))
QC_ALLOW_ANCHORS = _env_bool("QC_REPEAT_ALLOW_ANCHORS", True)

BASE_TRIAL_FIELDS = {
    "stim_id",
    "stimulus_id",
    "image_path",
    "blur_path",
    "gloss_index",
    "class_label",
    "material_class",
    "render_mode",
    "source_image_path",
    "source_image",
    "annotation_path",
    "target_label",
    "source_sample_id",
    "source_object_labels",
    "stimulus_name",
    "stimulus_display_name",
    "batch_id",
    "batch_number",
    "batch_position",
    "is_anchor",
    "is_qc_eligible",
}

TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
FALSY_VALUES = {"0", "false", "no", "n", "off"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_participant_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._")[:80]


def participant_seed(participant_id: str) -> int:
    normalized = normalize_participant_id(participant_id) or participant_id.strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assignment_seed(participant_id: str, batch_number: int | None = None) -> int:
    normalized = normalize_participant_id(participant_id) or participant_id.strip()
    seed_text = (
        f"{normalized}|batch={batch_number or 'all'}|"
        f"{TRIAL_ASSIGNMENT_VERSION}|gap={QC_MIN_GAP}|allow_anchor_qc={int(QC_ALLOW_ANCHORS)}"
    )
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def make_trial_key(stim_id: str, rep_index: int, *, qc_repeat_index: int | None = None) -> str:
    if qc_repeat_index is not None:
        return f"{stim_id}__qc{qc_repeat_index}"
    return f"{stim_id}__rep{rep_index}"


def stimulus_manifest_id(csv_path: Path = STIM_CSV) -> str:
    if not csv_path.exists():
        return "missing"
    digest = hashlib.sha256()
    digest.update(csv_path.read_bytes())
    digest.update(
        (
            f"|assignment={TRIAL_ASSIGNMENT_VERSION}"
            f"|batches={BATCH_COUNT}|anchors={ANCHOR_TRIAL_COUNT}"
            f"|unique={UNIQUE_TRIALS_PER_BATCH}|qc={QC_REPEAT_COUNT}"
            f"|gap={QC_MIN_GAP}|allow_anchor_qc={int(QC_ALLOW_ANCHORS)}"
        ).encode("utf-8")
    )
    return digest.hexdigest()[:16]


CURRENT_STIMULUS_MANIFEST_ID = stimulus_manifest_id(STIM_CSV)


def _parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in TRUTHY_VALUES:
        return True
    if text in FALSY_VALUES:
        return False
    return default


def _row_batch_id(row: dict) -> int | None:
    raw = (row.get("batch_id") or row.get("batch_number") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= BATCH_COUNT else None


def _uses_anchor_qc_design(rows: list[dict]) -> bool:
    if not rows:
        return False
    fields = set(rows[0].keys())
    return {"batch_id", "is_anchor"}.issubset(fields)


def build_observer_stimulus_label(
    source_image_path: str,
    class_label: str,
    source_object_labels: str,
    fallback_name: str = "",
) -> str:
    image_name = Path(source_image_path).stem if source_image_path else ""
    image_name = image_name or fallback_name or "stimulus"
    object_part = (source_object_labels or "").strip() or "unknown object"
    material_part = (class_label or "").strip() or "Object"
    return f"{image_name} | {material_part} | {object_part}"


def _spread_positions(total: int, count: int) -> list[int]:
    positions: list[int] = []
    used: set[int] = set()
    for i in range(count):
        pos = int(round(((i + 0.5) * total / count) - 0.5))
        pos = max(0, min(total - 1, pos))
        while pos in used and pos < total - 1:
            pos += 1
        while pos in used and pos > 0:
            pos -= 1
        positions.append(pos)
        used.add(pos)
    return sorted(positions)


def _trial_from_row(
    row: dict,
    *,
    rep_index: int,
    batch_number: int | None,
    manifest_row_index: int,
    is_anchor: bool,
    is_qc_eligible: bool,
    stimulus_batch_id: str,
) -> dict:
    stim_id = row.get("stim_id") or row.get("stimulus_id")
    if not stim_id:
        raise ValueError(f"Stimulus row {manifest_row_index} is missing stim_id/stimulus_id.")

    stimulus_id = row.get("stimulus_id") or stim_id
    source_image = row.get("source_image") or row.get("source_image_path", "")
    material_class = row.get("material_class") or row.get("class_label", "Object") or "Object"
    observer_stimulus_label = row.get("observer_stimulus_label", "").strip() or build_observer_stimulus_label(
        source_image,
        material_class,
        row.get("source_object_labels", ""),
        fallback_name=row.get("source_sample_id", stimulus_id),
    )
    extra_fields = {
        key: value
        for key, value in row.items()
        if key not in BASE_TRIAL_FIELDS and value not in (None, "")
    }
    return {
        "trial_key": make_trial_key(stim_id, rep_index),
        "stim_id": stim_id,
        "stimulus_id": stimulus_id,
        "image_path": row["image_path"],
        "blur_path": row.get("blur_path", ""),
        "gloss_index": float(row["gloss_index"]),
        "class_label": row.get("class_label", material_class) or material_class,
        "material_class": material_class,
        "render_mode": row.get("render_mode", ""),
        "source_image_path": row.get("source_image_path", source_image),
        "source_image": source_image,
        "annotation_path": row.get("annotation_path", ""),
        "target_label": row.get("target_label", material_class),
        "source_sample_id": row.get("source_sample_id", ""),
        "source_object_labels": row.get("source_object_labels", ""),
        "stimulus_name": row.get("stimulus_name", stimulus_id),
        "stimulus_display_name": row.get("stimulus_display_name", row.get("stimulus_name", stimulus_id)),
        "observer_stimulus_label": observer_stimulus_label,
        "batch_id": int(batch_number) if batch_number is not None else "",
        "batch_number": int(batch_number) if batch_number is not None else "",
        "stimulus_batch_id": stimulus_batch_id,
        "manifest_row_index": manifest_row_index,
        "is_anchor": bool(is_anchor),
        "is_qc_eligible": bool(is_qc_eligible),
        "is_qc_repeat": False,
        "has_qc_repeat": False,
        "qc_repeat_index": None,
        "repeated_from_trial_key": None,
        "repeated_from_trial_index": None,
        "repeated_from_presented_order_index": None,
        "qc_gap": None,
        "assignment_role": "anchor" if is_anchor else "batch_unique",
        "rep_index": rep_index,
        **extra_fields,
    }


def _legacy_load_trials_from_rows(
    rows: list[dict],
    *,
    n_reps: int,
    participant_id: str | None,
    batch_number: int | None,
) -> list[dict]:
    if batch_number is not None and len(rows) != BATCH_COUNT * TRIALS_PER_BATCH:
        raise RuntimeError(
            f"Expected {BATCH_COUNT * TRIALS_PER_BATCH} stimuli for "
            f"{BATCH_COUNT} x {TRIALS_PER_BATCH} batching, found {len(rows)}."
        )

    indexed_rows = list(enumerate(rows))
    if batch_number is not None:
        indexed_rows = [
            (global_index, row)
            for global_index, row in indexed_rows
            if (global_index % BATCH_COUNT) + 1 == int(batch_number)
        ]
        if len(indexed_rows) != TRIALS_PER_BATCH:
            raise RuntimeError(
                f"Batch {batch_number} should contain {TRIALS_PER_BATCH} stimuli, "
                f"found {len(indexed_rows)}."
            )

    trials: list[dict] = []
    for global_index, row in indexed_rows:
        row_batch_number = (global_index % BATCH_COUNT) + 1
        row_batch_position = (global_index // BATCH_COUNT) + 1
        for rep_index in range(n_reps):
            trial = _trial_from_row(
                row,
                rep_index=rep_index,
                batch_number=row.get("batch_number", "") or row_batch_number,
                manifest_row_index=global_index,
                is_anchor=_parse_bool(row.get("is_anchor"), False),
                is_qc_eligible=_parse_bool(row.get("is_qc_eligible"), True),
                stimulus_batch_id=str(row.get("batch_id", "") or row_batch_number),
            )
            trial["batch_position"] = row.get("batch_position", "") or row_batch_position
            trials.append(trial)

    rng = random.Random(participant_seed(participant_id)) if participant_id else random.Random()
    rng.shuffle(trials)
    for idx, trial in enumerate(trials):
        trial["planned_trial_index"] = idx
        trial["presented_order_index"] = idx + 1
        trial["batch_position"] = idx + 1
    return trials


def _select_anchor_qc_rows(rows: list[dict], batch_number: int) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    anchors: list[tuple[int, dict]] = []
    uniques: list[tuple[int, dict]] = []

    for row_index, row in enumerate(rows):
        if _parse_bool(row.get("is_anchor"), False):
            anchors.append((row_index, row))
            continue
        if _row_batch_id(row) == batch_number:
            uniques.append((row_index, row))

    if len(anchors) != ANCHOR_TRIAL_COUNT:
        raise RuntimeError(
            f"Expected {ANCHOR_TRIAL_COUNT} shared anchors, found {len(anchors)} in stimuli.csv."
        )
    if len(uniques) != UNIQUE_TRIALS_PER_BATCH:
        raise RuntimeError(
            f"Batch {batch_number} should contain {UNIQUE_TRIALS_PER_BATCH} unique stimuli, "
            f"found {len(uniques)}."
        )
    return anchors, uniques


def _build_base_anchor_qc_sequence(
    anchors: list[tuple[int, dict]],
    uniques: list[tuple[int, dict]],
    *,
    batch_number: int,
    rng: random.Random,
) -> list[dict]:
    anchors = list(anchors)
    uniques = list(uniques)
    rng.shuffle(anchors)
    rng.shuffle(uniques)

    anchor_trials = [
        _trial_from_row(
            row,
            rep_index=0,
            batch_number=batch_number,
            manifest_row_index=row_index,
            is_anchor=True,
            is_qc_eligible=_parse_bool(row.get("is_qc_eligible"), True),
            stimulus_batch_id="anchor",
        )
        for row_index, row in anchors
    ]
    unique_trials = [
        _trial_from_row(
            row,
            rep_index=0,
            batch_number=batch_number,
            manifest_row_index=row_index,
            is_anchor=False,
            is_qc_eligible=_parse_bool(row.get("is_qc_eligible"), True),
            stimulus_batch_id=str(_row_batch_id(row) or batch_number),
        )
        for row_index, row in uniques
    ]

    sequence: list[dict | None] = [None] * BASE_TRIALS_PER_BATCH
    for pos, trial in zip(_spread_positions(BASE_TRIALS_PER_BATCH, ANCHOR_TRIAL_COUNT), anchor_trials):
        sequence[pos] = trial

    unique_iter = iter(unique_trials)
    for idx in range(BASE_TRIALS_PER_BATCH):
        if sequence[idx] is None:
            sequence[idx] = next(unique_iter)

    return [trial for trial in sequence if trial is not None]


def _select_qc_original_keys(base_sequence: list[dict], rng: random.Random) -> list[str]:
    latest_original_index = len(base_sequence) - QC_MIN_GAP - 1
    candidates = [
        (idx, trial)
        for idx, trial in enumerate(base_sequence)
        if idx <= latest_original_index
        and trial.get("is_qc_eligible", True)
        and (QC_ALLOW_ANCHORS or not trial.get("is_anchor", False))
    ]
    rng.shuffle(candidates)

    selected: list[str] = []
    selected_stimuli: set[str] = set()
    for _, trial in candidates:
        stimulus_id = str(trial.get("stimulus_id") or trial.get("stim_id"))
        if stimulus_id in selected_stimuli:
            continue
        selected.append(trial["trial_key"])
        selected_stimuli.add(stimulus_id)
        if len(selected) == QC_REPEAT_COUNT:
            return selected

    raise RuntimeError(
        f"Could not select {QC_REPEAT_COUNT} QC repeats with min gap {QC_MIN_GAP}. "
        "Check is_qc_eligible fields or lower QC_REPEAT_MIN_GAP."
    )


def _insert_qc_repeats(sequence: list[dict], rng: random.Random) -> list[dict]:
    final_sequence = list(sequence)
    original_keys = _select_qc_original_keys(sequence, rng)

    for qc_index, original_key in enumerate(original_keys, start=1):
        original_index = next(
            idx for idx, trial in enumerate(final_sequence) if trial["trial_key"] == original_key
        )
        original = final_sequence[original_index]
        min_insert_pos = original_index + QC_MIN_GAP + 1
        possible_positions = list(range(min_insert_pos, len(final_sequence) + 1))
        if not possible_positions:
            raise RuntimeError(
                f"QC original at index {original_index} has no valid repeat position with gap {QC_MIN_GAP}."
            )
        insert_pos = rng.choice(possible_positions)
        repeat = copy.deepcopy(original)
        repeat["trial_key"] = make_trial_key(repeat["stim_id"], 1, qc_repeat_index=qc_index)
        repeat["is_qc_repeat"] = True
        repeat["qc_repeat_index"] = qc_index
        repeat["assignment_role"] = "qc_repeat"
        repeat["rep_index"] = 1
        repeat["repeated_from_trial_key"] = original_key
        repeat["observer_stimulus_label"] = original["observer_stimulus_label"]
        final_sequence.insert(insert_pos, repeat)

    return final_sequence


def _finalize_sequence_metadata(sequence: list[dict], batch_number: int) -> list[dict]:
    by_key = {trial["trial_key"]: trial for trial in sequence}
    index_by_key = {trial["trial_key"]: idx for idx, trial in enumerate(sequence)}

    for idx, trial in enumerate(sequence):
        trial["planned_trial_index"] = idx
        trial["presented_order_index"] = idx + 1
        trial["batch_position"] = idx + 1
        trial["batch_id"] = int(batch_number)
        trial["batch_number"] = int(batch_number)

    for trial in sequence:
        if not trial.get("is_qc_repeat"):
            continue
        original_key = trial.get("repeated_from_trial_key")
        original = by_key.get(original_key)
        if original is None:
            continue
        original_index = index_by_key[original_key]
        repeat_index = index_by_key[trial["trial_key"]]
        gap = repeat_index - original_index
        trial["repeated_from_trial_index"] = original_index
        trial["repeated_from_presented_order_index"] = original_index + 1
        trial["qc_gap"] = gap
        original["has_qc_repeat"] = True
        original["qc_repeat_trial_key"] = trial["trial_key"]
        original["qc_repeat_trial_index"] = repeat_index
        original["qc_repeat_presented_order_index"] = repeat_index + 1
        original["qc_gap"] = gap

    return sequence


def _load_anchor_qc_trials_from_rows(
    rows: list[dict],
    *,
    participant_id: str,
    batch_number: int,
) -> list[dict]:
    anchors, uniques = _select_anchor_qc_rows(rows, batch_number)
    rng = random.Random(assignment_seed(participant_id, batch_number))
    base_sequence = _build_base_anchor_qc_sequence(
        anchors,
        uniques,
        batch_number=batch_number,
        rng=rng,
    )
    sequence = _insert_qc_repeats(base_sequence, rng)
    sequence = _finalize_sequence_metadata(sequence, batch_number)
    validate_trial_assignment(sequence, strict=True)
    return sequence


def load_trials_from_csv(
    csv_path: Path = STIM_CSV,
    n_reps: int = N_REPS,
    *,
    participant_id: str | None = None,
    batch_number: int | None = None,
) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Stimulus CSV not found: {csv_path}")
    if n_reps < 1:
        raise ValueError("n_reps must be at least 1.")
    if batch_number is not None and not 1 <= int(batch_number) <= BATCH_COUNT:
        raise ValueError(f"batch_number must be between 1 and {BATCH_COUNT}.")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("stimuli.csv is empty.")

    if batch_number is not None and _uses_anchor_qc_design(rows):
        if n_reps != 1:
            raise ValueError("Anchor/QC assignment supports n_reps=1; QC repeats are added internally.")
        if not participant_id:
            raise ValueError("participant_id is required for deterministic anchor/QC assignment.")
        return _load_anchor_qc_trials_from_rows(
            rows,
            participant_id=participant_id,
            batch_number=int(batch_number),
        )

    return _legacy_load_trials_from_rows(
        rows,
        n_reps=n_reps,
        participant_id=participant_id,
        batch_number=batch_number,
    )


def assignment_debug_summary(trials: list[dict]) -> dict:
    stimulus_positions: dict[str, list[int]] = defaultdict(list)
    for idx, trial in enumerate(trials):
        stimulus_id = str(trial.get("stimulus_id") or trial.get("stim_id"))
        stimulus_positions[stimulus_id].append(idx)

    duplicate_stimuli = {
        stimulus_id: positions
        for stimulus_id, positions in stimulus_positions.items()
        if len(positions) > 1
    }
    qc_repeats = [trial for trial in trials if trial.get("is_qc_repeat")]
    qc_repeat_pairs = [
        {
            "stimulus_id": trial.get("stimulus_id"),
            "repeat_trial_key": trial.get("trial_key"),
            "repeat_position": idx,
            "repeat_presented_order_index": idx + 1,
            "repeated_from_trial_key": trial.get("repeated_from_trial_key"),
            "repeated_from_trial_index": trial.get("repeated_from_trial_index"),
            "repeated_from_presented_order_index": trial.get("repeated_from_presented_order_index"),
            "gap": trial.get("qc_gap"),
            "is_anchor": bool(trial.get("is_anchor")),
        }
        for idx, trial in enumerate(trials)
        if trial.get("is_qc_repeat")
    ]

    base_trials = [trial for trial in trials if not trial.get("is_qc_repeat")]
    anchor_base_positions = [
        idx for idx, trial in enumerate(trials) if trial.get("is_anchor") and not trial.get("is_qc_repeat")
    ]
    unique_base_positions = [
        idx for idx, trial in enumerate(trials) if not trial.get("is_anchor") and not trial.get("is_qc_repeat")
    ]
    role_counts = Counter(str(trial.get("assignment_role") or "") for trial in trials)

    validation = validate_trial_assignment(trials, strict=False)
    return {
        "valid": validation["ok"],
        "errors": validation["errors"],
        "total_trials": len(trials),
        "base_trial_count": len(base_trials),
        "anchor_base_count": len(anchor_base_positions),
        "unique_base_count": len(unique_base_positions),
        "qc_repeat_count": len(qc_repeats),
        "role_counts": dict(role_counts),
        "duplicate_stimuli": duplicate_stimuli,
        "anchor_positions": anchor_base_positions,
        "anchor_presented_order_indexes": [pos + 1 for pos in anchor_base_positions],
        "unique_positions_sample": unique_base_positions[:20],
        "qc_repeat_pairs": qc_repeat_pairs,
        "qc_min_gap": QC_MIN_GAP,
        "qc_allow_anchors": QC_ALLOW_ANCHORS,
    }


def validate_trial_assignment(trials: list[dict], *, strict: bool = False) -> dict:
    errors: list[str] = []
    total = len(trials)
    base_trials = [trial for trial in trials if not trial.get("is_qc_repeat")]
    qc_repeats = [trial for trial in trials if trial.get("is_qc_repeat")]
    anchors = [trial for trial in base_trials if trial.get("is_anchor")]
    uniques = [trial for trial in base_trials if not trial.get("is_anchor")]

    if total != TRIALS_PER_BATCH:
        errors.append(f"total trial count should be {TRIALS_PER_BATCH}, found {total}.")
    if len(anchors) != ANCHOR_TRIAL_COUNT:
        errors.append(f"anchor base count should be {ANCHOR_TRIAL_COUNT}, found {len(anchors)}.")
    if len(uniques) != UNIQUE_TRIALS_PER_BATCH:
        errors.append(f"unique base count should be {UNIQUE_TRIALS_PER_BATCH}, found {len(uniques)}.")
    if len(qc_repeats) != QC_REPEAT_COUNT:
        errors.append(f"QC repeat count should be {QC_REPEAT_COUNT}, found {len(qc_repeats)}.")

    base_ids = [str(trial.get("stimulus_id") or trial.get("stim_id")) for trial in base_trials]
    repeated_ids = [str(trial.get("stimulus_id") or trial.get("stim_id")) for trial in qc_repeats]
    base_counts = Counter(base_ids)
    repeated_counts = Counter(repeated_ids)

    unintended_base_duplicates = sorted(stimulus_id for stimulus_id, count in base_counts.items() if count > 1)
    if unintended_base_duplicates:
        errors.append(f"base trials contain duplicate stimuli: {unintended_base_duplicates}.")
    repeated_twice = sorted(stimulus_id for stimulus_id, count in repeated_counts.items() if count > 1)
    if repeated_twice:
        errors.append(f"same stimulus selected for more than one QC repeat: {repeated_twice}.")

    trial_key_counts = Counter(str(trial.get("trial_key", "")) for trial in trials)
    duplicate_keys = sorted(key for key, count in trial_key_counts.items() if count > 1)
    if duplicate_keys:
        errors.append(f"trial keys are not unique: {duplicate_keys}.")

    index_by_key = {trial["trial_key"]: idx for idx, trial in enumerate(trials) if trial.get("trial_key")}
    for repeat in qc_repeats:
        original_key = repeat.get("repeated_from_trial_key")
        if not original_key or original_key not in index_by_key:
            errors.append(f"QC repeat {repeat.get('trial_key')} has no valid original link.")
            continue
        original_index = index_by_key[original_key]
        repeat_index = index_by_key[repeat["trial_key"]]
        if repeat_index <= original_index:
            errors.append(f"QC repeat {repeat.get('trial_key')} appears before its original.")
        if repeat_index - original_index < QC_MIN_GAP:
            errors.append(
                f"QC repeat {repeat.get('trial_key')} gap is {repeat_index - original_index}, "
                f"minimum is {QC_MIN_GAP}."
            )
        original = trials[original_index]
        if (original.get("stimulus_id") or original.get("stim_id")) != (
            repeat.get("stimulus_id") or repeat.get("stim_id")
        ):
            errors.append(f"QC repeat {repeat.get('trial_key')} stimulus does not match its original.")

    duplicated_stimuli = {
        stimulus_id: count
        for stimulus_id, count in Counter(base_ids + repeated_ids).items()
        if count > 1
    }
    for stimulus_id, count in duplicated_stimuli.items():
        if repeated_counts.get(stimulus_id) != 1 or base_counts.get(stimulus_id) != 1 or count != 2:
            errors.append(f"unexpected duplicate stimulus collision for {stimulus_id}: {count} occurrences.")

    result = {"ok": not errors, "errors": errors}
    if strict and errors:
        raise RuntimeError("Invalid trial assignment: " + " ".join(errors))
    return result


def create_session(
    participant_id: str,
    *,
    storage_backend: str,
    csv_path: Path = STIM_CSV,
    n_reps: int = N_REPS,
    batch_number: int | None = None,
) -> dict:
    now = utc_now_iso()
    participant_storage_id = normalize_participant_id(participant_id)
    trials = load_trials_from_csv(
        csv_path,
        n_reps,
        participant_id=participant_id,
        batch_number=batch_number,
    )
    return {
        "participant_id": participant_id.strip(),
        "participant_storage_id": participant_storage_id,
        "session_id": participant_storage_id,
        "stimulus_manifest_id": stimulus_manifest_id(csv_path),
        "trial_assignment_version": TRIAL_ASSIGNMENT_VERSION,
        "storage_backend": storage_backend,
        "n_reps": n_reps,
        "batch_number": int(batch_number) if batch_number is not None else None,
        "batch_id": int(batch_number) if batch_number is not None else None,
        "batch_count": BATCH_COUNT,
        "batch_trial_count": len(trials),
        "qc_min_gap": QC_MIN_GAP,
        "qc_repeat_count": QC_REPEAT_COUNT,
        "qc_allow_anchors": QC_ALLOW_ANCHORS,
        "assignment_summary": assignment_debug_summary(trials) if batch_number is not None else {},
        "started_at": now,
        "updated_at": now,
        "completed": False,
        "completed_at": None,
        "current_index": 0,
        "trial_started_at": time.time(),
        "trials": trials,
        "responses": {},
    }


def ensure_session_defaults(session: dict, storage_backend: str) -> dict:
    session.setdefault("participant_storage_id", normalize_participant_id(session.get("participant_id", "")))
    session.setdefault("session_id", session.get("participant_storage_id", ""))
    session.setdefault("trial_assignment_version", TRIAL_ASSIGNMENT_VERSION)
    session.setdefault("n_reps", N_REPS)
    session.setdefault("batch_number", None)
    session.setdefault("batch_id", session.get("batch_number"))
    session.setdefault("batch_count", BATCH_COUNT)
    session.setdefault("batch_trial_count", TRIALS_PER_BATCH if session.get("batch_number") else None)
    session.setdefault("qc_min_gap", QC_MIN_GAP)
    session.setdefault("qc_repeat_count", QC_REPEAT_COUNT)
    session.setdefault("qc_allow_anchors", QC_ALLOW_ANCHORS)
    session.setdefault("started_at", utc_now_iso())
    session.setdefault("updated_at", utc_now_iso())
    session.setdefault("completed", False)
    session.setdefault("completed_at", None)
    session.setdefault("current_index", 0)
    session.setdefault("responses", {})
    session["storage_backend"] = storage_backend

    # Reconstruct trials from CSV if missing. Compact Supabase sessions store
    # only trial_keys; the deterministic assignment code rebuilds the same
    # candidate set and this block restores the saved order exactly.
    if not session.get("trials"):
        pid = session.get("participant_id", "")
        saved_keys = session.pop("trial_keys", None)
        try:
            session["trials"] = load_trials_from_csv(
                STIM_CSV,
                session.get("n_reps", N_REPS),
                participant_id=pid,
                batch_number=session.get("batch_number"),
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            session["trials"] = []

        if saved_keys and session["trials"]:
            by_key = {t["trial_key"]: t for t in session["trials"]}
            ordered = [by_key[k] for k in saved_keys if k in by_key]
            if len(ordered) == len(session["trials"]):
                session["trials"] = ordered

    if session["trials"]:
        session["current_index"] = max(0, min(session["current_index"], len(session["trials"]) - 1))
        session["batch_trial_count"] = len(session["trials"])
        session["assignment_summary"] = assignment_debug_summary(session["trials"])
    else:
        session["current_index"] = 0
        session.setdefault("assignment_summary", {})

    # Reaction times should restart when a participant resumes the page.
    session["trial_started_at"] = time.time()
    return session


def session_uses_current_stimulus_manifest(session: dict) -> bool:
    return (session.get("stimulus_manifest_id") or "").strip() == CURRENT_STIMULUS_MANIFEST_ID


def get_category_label(value: float, type_: str) -> str:
    if type_ == "gloss":
        if value <= 3:
            return "Matte"
        if value <= 7:
            return "Semi-Glossy"
        return "Glossy"
    if type_ == "translucency":
        if value <= 3:
            return "Opaque"
        if value <= 7:
            return "Semi-Translucent"
        return "Highly Translucent"
    if type_ == "texture":
        if value <= 3:
            return "Uniform"
        if value <= 7:
            return "Moderately Textured"
        return "Highly Textured"
    return ""


def current_trial(session: dict) -> dict:
    return session["trials"][session["current_index"]]


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


def progress_label(session: dict) -> str:
    total = len(session["trials"])
    if total == 0:
        return "No trials loaded"
    return f"Trial {session['current_index'] + 1} / {total}"


def break_message(session: dict) -> str:
    index = session["current_index"]
    total = len(session["trials"])
    messages: list[str] = []

    if index > 0 and index % BREAK_EVERY == 0:
        messages.append("Take a short break if needed. Your progress is already saved.")
    if total and index == total - 1:
        messages.append("This is the final trial. Press Submit when you are done.")

    return " ".join(messages)


def build_response(
    session: dict,
    *,
    gloss_rating: float,
    translucency_rating: float,
    texture_rating: float,
    rt_sec: float,
) -> dict:
    trial = current_trial(session)
    gloss_value = round(float(gloss_rating), 2)
    translucency_value = round(float(translucency_rating), 2)
    texture_value = round(float(texture_rating), 2)
    demographics = session.get("demographics", {}) or {}
    response = {
        key: value
        for key, value in trial.items()
        if key not in {"gloss_index"}
    }
    response.update(
        {
            "participant_id": session["participant_id"],
            "observer_id": session["participant_id"],
            "session_id": session.get("session_id", session.get("participant_storage_id", "")),
            "participant_storage_id": session["participant_storage_id"],
            "trial_index": session["current_index"],
            "presented_order_index": session["current_index"] + 1,
            "trial_key": trial["trial_key"],
            "batch_id": session.get("batch_id", session.get("batch_number", trial.get("batch_number", ""))),
            "batch_number": session.get("batch_number", trial.get("batch_number", "")),
            "batch_count": session.get("batch_count", BATCH_COUNT),
            "batch_position": trial.get("batch_position", ""),
            "stim_id": trial["stim_id"],
            "stimulus_id": trial.get("stimulus_id", trial["stim_id"]),
            "image_path": trial["image_path"],
            "class_label": trial["class_label"],
            "material_class": trial.get("material_class", trial["class_label"]),
            "render_mode": trial.get("render_mode", ""),
            "source_image_path": trial.get("source_image_path", ""),
            "source_image": trial.get("source_image", trial.get("source_image_path", "")),
            "annotation_path": trial.get("annotation_path", ""),
            "target_label": trial.get("target_label", trial["class_label"]),
            "source_sample_id": trial.get("source_sample_id", ""),
            "source_object_labels": trial.get("source_object_labels", ""),
            "stimulus_name": trial.get("stimulus_name", trial["stim_id"]),
            "stimulus_display_name": trial.get(
                "stimulus_display_name", trial.get("stimulus_name", trial["stim_id"])
            ),
            "is_anchor": bool(trial.get("is_anchor", False)),
            "is_qc_eligible": bool(trial.get("is_qc_eligible", True)),
            "is_qc_repeat": bool(trial.get("is_qc_repeat", False)),
            "has_qc_repeat": bool(trial.get("has_qc_repeat", False)),
            "qc_repeat_index": trial.get("qc_repeat_index"),
            "repeated_from_trial_key": trial.get("repeated_from_trial_key"),
            "repeated_from_trial_index": trial.get("repeated_from_trial_index"),
            "repeated_from_presented_order_index": trial.get("repeated_from_presented_order_index"),
            "qc_gap": trial.get("qc_gap"),
            "assignment_role": trial.get("assignment_role", ""),
            "stimulus_batch_id": trial.get("stimulus_batch_id", ""),
            "gloss_index_precomputed": trial["gloss_index"],
            "rep_index": trial["rep_index"],
            "gloss_rating": gloss_value,
            "gloss_category": get_category_label(gloss_value, "gloss"),
            "translucency_rating": translucency_value,
            "translucency_category": get_category_label(translucency_value, "translucency"),
            "texture_rating": texture_value,
            "texture_category": get_category_label(texture_value, "texture"),
            "rt_sec": round(max(rt_sec, 0.0), 3),
            "response_time_ms": int(round(max(rt_sec, 0.0) * 1000)),
            "timestamp": utc_now_iso(),
            "device": demographics.get("device"),
            "age": demographics.get("age"),
            "gender": demographics.get("gender"),
            "color_vision_deficiency": demographics.get("color_vision_deficiency"),
            "cvd_type": demographics.get("cvd_type"),
            "glasses": demographics.get("glasses"),
            "color_scientist": demographics.get("color_scientist"),
        }
    )
    return response


def ordered_results(session: dict) -> list[dict]:
    ordered: list[dict] = []
    responses = session.get("responses", {})
    for trial in session.get("trials", []):
        response = responses.get(trial["trial_key"])
        if response is not None:
            ordered.append(response)
    return ordered
