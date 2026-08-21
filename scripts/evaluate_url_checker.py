from __future__ import annotations

from dataclasses import is_dataclass, asdict
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# Config (easy to edit)
# -----------------------------
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "urls_train.csv"
USE_SAMPLE = True
SAMPLE_SIZE = 1000
RANDOM_STATE = 42

FAILED_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "failed_urls.csv"


# -----------------------------
# URL checker adapter
# -----------------------------
def check_url_label(url: str) -> int:
    """
    Return predicted label for a URL:
    - 1 = malicious/suspicious/phishing/fraud
    - 0 = benign/safe

    Default behavior:
    1) Try backend.analyzer.url_analyzer.analyze_url (project API logic)
    2) Fallback to url_checker.dataset rule_check if backend path is unavailable
    """
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from backend.analyzer.url_analyzer import analyze_url

        result = analyze_url(url)
        return to_binary_label(result)
    except Exception:
        # Fallback for standalone testing inside url_checker/
        url_checker_root = Path(__file__).resolve().parents[1]
        if str(url_checker_root) not in sys.path:
            sys.path.insert(0, str(url_checker_root))

        from dataset.utils.url_normalize import normalize_url
        from dataset.utils.url_rules import rule_check

        info = normalize_url(url)
        is_suspicious, _reasons, _domain_valid, _unusual_findings = rule_check(info)
        return int(bool(is_suspicious))


def to_binary_label(prediction) -> int:
    """Convert various checker outputs into strict binary labels (1/0)."""
    if isinstance(prediction, bool):
        return int(prediction)

    if isinstance(prediction, (int, float)):
        return 1 if int(prediction) == 1 else 0

    if isinstance(prediction, str):
        value = prediction.strip().lower()
        positive_tokens = {
            "1",
            "true",
            "fraud",
            "phishing",
            "malicious",
            "suspicious",
            "danger",
            "unsafe",
            "blocked",
        }
        return 1 if value in positive_tokens else 0

    if is_dataclass(prediction):
        return to_binary_label(asdict(prediction))

    if isinstance(prediction, dict):
        # Prefer explicit binary fields if present.
        for key in ("label", "prediction", "predicted_label"):
            if key in prediction:
                return to_binary_label(prediction[key])

        # Common API-style fields.
        if "verdict" in prediction:
            verdict = str(prediction["verdict"]).strip().lower()
            return 1 if verdict in {"fraud", "malicious", "phishing", "suspicious", "unsafe"} else 0

        if "risk_score" in prediction:
            try:
                return 1 if float(prediction["risk_score"]) >= 50 else 0
            except Exception:
                return 0

        return 0

    # Object-like return from checker (e.g., dataclass instance without conversion above).
    if hasattr(prediction, "verdict"):
        verdict = str(getattr(prediction, "verdict", "")).strip().lower()
        return 1 if verdict in {"fraud", "malicious", "phishing", "suspicious", "unsafe"} else 0

    if hasattr(prediction, "label"):
        return to_binary_label(getattr(prediction, "label"))

    if hasattr(prediction, "prediction"):
        return to_binary_label(getattr(prediction, "prediction"))

    return 0


# -----------------------------
# Evaluation flow
# -----------------------------
def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required_cols = {"url", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    # Keep only valid rows and normalize label to int 0/1.
    eval_df = df[["url", "label"]].dropna(subset=["url", "label"]).copy()
    eval_df["label"] = eval_df["label"].astype(int).clip(lower=0, upper=1)

    if USE_SAMPLE:
        sample_size = min(SAMPLE_SIZE, len(eval_df))
        eval_df = eval_df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Using random sample: {sample_size} rows (RANDOM_STATE={RANDOM_STATE})")
    else:
        eval_df = eval_df.reset_index(drop=True)
        print(f"Using full dataset: {len(eval_df)} rows")

    y_true: list[int] = []
    y_pred: list[int] = []
    failed_rows: list[dict[str, int | str]] = []

    total = len(eval_df)
    for idx, row in eval_df.iterrows():
        url = str(row["url"])
        true_label = int(row["label"])

        try:
            pred_label = int(to_binary_label(check_url_label(url)))
        except Exception:
            # Treat checker errors as benign prediction for deterministic output.
            pred_label = 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        if pred_label != true_label:
            failed_rows.append(
                {
                    "url": url,
                    "expected_label": true_label,
                    "predicted_label": pred_label,
                }
            )

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    failed_df = pd.DataFrame(failed_rows, columns=["url", "expected_label", "predicted_label"])
    failed_df.to_csv(FAILED_OUTPUT_PATH, index=False)
    print(f"\nSaved mismatches: {len(failed_df)} rows -> {FAILED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
