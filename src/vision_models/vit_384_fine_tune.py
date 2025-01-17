"""
This script is used to fine-tune a ViT model with 384x384 resolution on the custom dataset.
"""
from datasets import load_dataset, load_metric
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
from PIL import Image
import json


def load_datasets(data_dir):
    """
    Load the dataset from the given directory.

    Parameters
    ----------
    data_dir : str
        The directory where the dataset is stored.

    Returns
    -------
    dataset_train : datasets.Dataset
        The training dataset.

    dataset_val : datasets.Dataset
        The testing dataset.
    """
    dataset_train = load_dataset("imagefolder", data_dir=data_dir, split="train")
    dataset_val = load_dataset("imagefolder", data_dir=data_dir, split="validation")
    return dataset_train, dataset_val


def transform(batch):
    """
    Transform the given example batch.

    Parameters
    ----------
    batch : dict
        The example batch.

    Returns
    -------
    dict
        The transformed example batch.
    """
    inputs = processor([x.convert("RGB") for x in batch["image"]], return_tensors="pt")
    # print("Processed tensor shape:", inputs["pixel_values"].shape)
    inputs["labels"] = batch["label"]
    return inputs


def collate_fn(batch):
    """
    Collate the given batch.

    Parameters
    ----------
    batch : list
        The batch.

    Returns
    -------
    dict
        The collated batch.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(p):
    """
    Compute the metrics.

    Parameters
    ----------
    p : EvalPrediction
        The evaluation prediction.

    Returns
    -------
    float
        The accuracy.
    """
    metric = load_metric("accuracy")
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


def main():
    # List of models to try
    model_list = [
        "google/vit-base-patch16-384",
    ]

    data_dir = "../folder_raw"

    dataset_name = data_dir.split("/")[-1]

    # Define the new size
    image_res = {"height": 384, "width": 384}

    image_res_name = image_res["height"]

    for model_name_or_path in model_list:
        output_dir = f"../vision_models/{model_name_or_path.split('/')[-1]}_{image_res_name}_{dataset_name}_fine_tuned"

        # Load datasets
        dataset_train, dataset_val = load_datasets(data_dir)

        # Image Processor
        global processor
        processor = ViTImageProcessor(size=image_res)

        prepared_ds_train = dataset_train.with_transform(transform)
        prepared_ds_val = dataset_val.with_transform(transform)

        # Model Setup
        labels = dataset_train.features["label"].names
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
            ignore_mismatched_sizes=True,
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=4,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=5e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
        )

        # Trainer Setup
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds_train,
            eval_dataset=prepared_ds_val,
            tokenizer=processor,
        )

        # Train and Evaluate
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        metrics = trainer.evaluate(prepared_ds_val)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        print(f"Training and evaluation complete for {model_name_or_path}!")


if __name__ == "__main__":
    main()
