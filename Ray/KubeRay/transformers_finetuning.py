import os
from typing import Dict

import ray.train
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

import ray
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the dataset and preprocess it
def load_data(dataset_name, model_name):
    # Load and shuffle the dataset
    dataset = load_dataset(dataset_name)
    
    # Limit dataset size for quicker training
    ds_size = 10000
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(ds_size))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(ds_size))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocess function for tokenization
    def preprocess(examples):
        # Tokenize text with padding and truncation
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Preprocess the datasets (train and test)
    dataset["train"] = dataset["train"].map(preprocess, batched=True)
    dataset["test"] = dataset["test"].map(preprocess, batched=True)
    
    # Convert Hugging Face datasets to Ray datasets for distributed processing
    ray_train_ds = ray.data.from_huggingface(dataset["train"])
    ray_eval_ds = ray.data.from_huggingface(dataset["test"])
    
    return ray_train_ds, ray_eval_ds


def train_func_per_worker(config: Dict):

    model_config = AutoConfig.from_pretrained(config["model_name"], num_labels=config["num_labels"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], config=model_config)

    # Build Ray Data iterables
    train_dataset = ray.train.get_dataset_shard("train")
    eval_dataset = ray.train.get_dataset_shard("evaluation")

    train_iterable_ds = train_dataset.iter_torch_batches(batch_size=config["per_device_train_batch_size"])
    eval_iterable_ds = eval_dataset.iter_torch_batches(batch_size=config["per_device_eval_batch_size"])
    
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        weight_decay=0.01,
        learning_rate=2e-5,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        max_steps=100, #TODO -
    )

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_iterable_ds,
        eval_dataset=eval_iterable_ds
    )

    # Inject Ray Train Report Callback
    trainer.add_callback(RayTrainReportCallback())

    trainer = prepare_trainer(trainer)
    trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()
    ray.train.report(metrics=eval_results)


def train_yelp_classifier(num_workers=4, cpus_per_worker=2, use_gpu=False):
    
    global_batch_size = 32

    # Load the dataset and preprocess it
    ray_train_ds, ray_eval_ds = load_data("yelp_review_full", "distilbert-base-uncased")

    train_config = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 5,
        "per_device_train_batch_size": global_batch_size // num_workers,
        "per_device_eval_batch_size": global_batch_size // num_workers,
        "num_train_epochs": 3,
        "metric_for_best_model": "accuracy",
    }

    # Configure computation resources
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker}
    )


    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        datasets={"train": ray_train_ds, "evaluation": ray_eval_ds}
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    result = trainer.fit()
    print(f"Training result: {result}")

    print("Training completed. Final evaluation metrics:", result.metrics)


if __name__ == "__main__":
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", "2"))
    train_yelp_classifier(num_workers=num_workers, cpus_per_worker=cpus_per_worker)

    