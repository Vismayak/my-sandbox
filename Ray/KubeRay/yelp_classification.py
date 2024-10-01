import os

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


# [1] Encapsulate data preprocessing, training, and evaluation
# logic in a training function
# ============================================================
def train_func():
    # Datasets
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=315)

    small_train_dataset = (
        dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
    )
    small_eval_dataset = (
        dataset["test"].select(range(1000)).map(tokenize_function, batched=True)
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", num_labels=5
    )

    # Evaluation Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="test_trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # [2] Report Metrics and Checkpoints to Ray Train
    # ===============================================
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    # [3] Prepare Transformers Trainer
    # ================================
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Start Training
    trainer.train()


def train_yelp_classification(num_workers=4, cpus_per_worker=2, use_gpu=False):
    global_batch_size = 32

    train_config = {
        "lr": 1e-3,
        "epochs": 3,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker}
    )

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")

    print("Training completed. Final evaluation metrics:", result.metrics)         

if __name__ == "__main__":
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", "2"))
    train_yelp_classification(num_workers, cpus_per_worker)                     