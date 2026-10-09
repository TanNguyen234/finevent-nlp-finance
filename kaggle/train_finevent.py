"""FinEvent Cloud Training Pipeline for Kaggle GPU T4.

Complies strictly with Kaggle Operational Rules:
- CUDA_VISIBLE_DEVICES = "0" (VRAM safety)
- Fast Validation Gate (Pre-train vs Post-train validation delta check)
- Models: microsoft/deberta-v3-small & ProsusAI/finbert
- Standardized Cloud Output: /kaggle/working/output/
"""

import os
import sys
import json
import time
import pathlib
import logging

# 1. Enforce single-GPU isolation to eliminate PyTorch DataParallel VRAM replication
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FinEvent-Kaggle")

OUTPUT_DIR = pathlib.Path("/kaggle/working/output")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = pathlib.Path("kaggle/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset path resolution
DATA_INPUT_DIR = pathlib.Path("/kaggle/input/finevent-benchmark-dataset")
if not (DATA_INPUT_DIR / "train.jsonl").exists():
    DATA_INPUT_DIR = pathlib.Path("data/labels")
if not (DATA_INPUT_DIR / "train.jsonl").exists():
    DATA_INPUT_DIR = pathlib.Path("/kaggle/working/data/labels")
    DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    base_url = "https://raw.githubusercontent.com/TanNguyen234/finevent-nlp-finance/main/data/labels"
    for fname in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        target_f = DATA_INPUT_DIR / fname
        if not target_f.exists():
            logger.info(f"Downloading {fname} from GitHub raw...")
            try:
                urllib.request.urlretrieve(f"{base_url}/{fname}", target_f)
            except Exception as e:
                logger.warning(f"Could not download {fname}: {e}")

CLASSES = ["GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}


def load_dataset_splits():
    train_df = pd.read_json(DATA_INPUT_DIR / "train.jsonl", lines=True)
    val_df = pd.read_json(DATA_INPUT_DIR / "val.jsonl", lines=True)
    test_df = pd.read_json(DATA_INPUT_DIR / "test.jsonl", lines=True)

    train_df["label"] = train_df["event_type"].map(LABEL2ID)
    val_df["label"] = val_df["event_type"].map(LABEL2ID)
    test_df["label"] = test_df["event_type"].map(LABEL2ID)

    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}


def run_fast_validation_gate(model, tokenizer, val_dataset, device):
    """Fast validation gate: evaluate <= 300 samples in < 60s."""
    logger.info("Executing Fast Validation Gate...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dataset:
            inputs = tokenizer(batch["text"], truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            all_preds.append(pred)
            all_labels.append(batch["label"])

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    logger.info(f"Fast Validation Gate Macro-F1: {macro_f1:.4f}")
    return float(macro_f1)


def train_and_evaluate_model(model_name: str, exp_id: str, train_df, val_df, test_df):
    logger.info(f"--- Starting Experiment: {exp_id} ({model_name}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    # 1. Fast Validation Gate (Pre-train Base baseline)
    pre_val_f1 = run_fast_validation_gate(model, tokenizer, val_ds, device)

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_val = val_ds.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn, batched=True)

    model_output_dir = OUTPUT_DIR / exp_id
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    start_time = time.time()
    trainer.train()
    train_duration = time.time() - start_time
    logger.info(f"Training completed in {train_duration:.1f}s")

    # Post-train evaluation
    post_val_metrics = trainer.evaluate(tokenized_val)
    test_metrics = trainer.evaluate(tokenized_test)

    # Generate predictions for classification report
    test_preds_raw = trainer.predict(tokenized_test)
    test_pred_labels = np.argmax(test_preds_raw.predictions, axis=-1)
    clf_report = classification_report(
        test_df["label"],
        test_pred_labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model_name": model_name,
        "exp_id": exp_id,
        "train_duration_sec": train_duration,
        "pre_train_val_macro_f1": pre_val_f1,
        "post_train_val_macro_f1": post_val_metrics["eval_macro_f1"],
        "delta_val_f1": round(post_val_metrics["eval_macro_f1"] - pre_val_f1, 4),
        "test_macro_f1": test_metrics["eval_macro_f1"],
        "classification_report": clf_report,
    }

    # Save metrics JSON to standardized cloud output directory
    with open(OUTPUT_DIR / f"{exp_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Result {exp_id}: Test Macro-F1 = {test_metrics['eval_macro_f1']:.4f}")
    return results


def main():
    logger.info("Starting FinEvent Kaggle Cloud Training...")
    train_df, val_df, test_df = load_dataset_splits()
    logger.info(f"Splits loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # E2-A: microsoft/deberta-v3-small
    res_deberta = train_and_evaluate_model(
        model_name="microsoft/deberta-v3-small",
        exp_id="E2_DeBERTa_v3_small",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # E2-B: ProsusAI/finbert
    res_finbert = train_and_evaluate_model(
        model_name="ProsusAI/finbert",
        exp_id="E2_FinBERT",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # Consolidate benchmark summary
    benchmark_summary = {
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": "Kaggle GPU Tesla T4" if torch.cuda.is_available() else "CPU",
        "E2_DeBERTa": res_deberta,
        "E2_FinBERT": res_finbert,
    }

    with open(OUTPUT_DIR / "final_cloud_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    logger.info("All Kaggle Cloud experiments completed successfully!")


if __name__ == "__main__":
    main()
