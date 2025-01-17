"""
This Script is to test the vision models on the test dataset
"""
from glob import glob
import os
import torch
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm
from transformers import (
    ViTImageProcessor,
    ConvNextImageProcessor,
    RegNetForImageClassification,
    ResNetForImageClassification,
    SwinForImageClassification,
    ViTForImageClassification,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_model(model_path, resolution):
    """
    Load a fine-tuned vision model based on the model name extracted from the model path and set the processor accordingly.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model.
    resolution : int
        Resolution for the image processor. Used for square resizing for ViT and Swin, and resizing with maintaining aspect ratio for ResNet and RegNet.

    Returns
    -------
    tuple
        A tuple containing the loaded model and processor.
    """
    # Extract model name from the last part of the model path
    model_name = model_path.split("/")[-1].lower()

    # Define the size based on the model name and set the processor
    if "vit" in model_name or "swin" in model_name:
        size = {"height": resolution, "width": resolution}
        processor = ViTImageProcessor(size=size)
    elif "resnet" in model_name or "regnet" in model_name:
        size = {"shortest_edge": resolution}
        processor = ConvNextImageProcessor(size=size)
    else:
        raise ValueError(
            "Invalid model path or model path does not contain a recognized keyword"
        )

    # Load the model based on the model name
    if "vit" in model_name:
        model = ViTForImageClassification.from_pretrained(model_path)
    elif "swin" in model_name:
        model = SwinForImageClassification.from_pretrained(model_path)
    elif "regnet" in model_name:
        model = RegNetForImageClassification.from_pretrained(model_path)
    elif "resnet" in model_name:
        model = ResNetForImageClassification.from_pretrained(model_path)
    else:
        raise ValueError(
            "Invalid model path or model path does not contain a recognized model keyword"
        )

    return model, processor


def classify_image(model, processor, img_path, device):
    """
    Classify an image using a fine-tuned model.

    Parameters
    ----------
    model : ImageClassification
        A fine-tuned model.
    processor : ImageProcessor
        An ImageProcessor.
    img_path : str
        Path to the image to classify.
    device : torch.device
        Device to use for inference.

    Returns
    -------
    str
        The class label of the image.
    """
    img = Image.open(img_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    output = model(**inputs)
    proba = output.logits.softmax(1)
    preds = proba.argmax(1)

    return preds


def evaluate_performance(y_true, y_pred):
    """
    Evaluate the performance of a model.

    Parameters
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.

    Returns
    -------
    dict
        A dictionary containing accuracy, precision, and recall.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def main():
    # Read the CSV
    df = pd.read_csv(
        "../test_captions_and_desc_with_image_path.csv"
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup (replace with your actual model and path)
    model_path = "../vision_models/regnet-y-320-seer-in1k_512_folder_raw_aug_fine_tuned"
    resolution = 512
    model, processor = load_model(model_path, resolution)
    model = model.to(device)
    # Extract model name from path for file naming
    model_name = model_path.split("/")[-1]  # Adjust the split logic as needed

    # Lists to store true labels and predictions
    true_labels = []
    predictions = []

    # Iterate over the DataFrame
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            image_path = row["image_path"]
            true_label = row["label"]
            prediction = classify_image(model, processor, image_path, device)

            true_labels.append(true_label)
            predictions.append(prediction.item())
        except Exception as e:
            print(f"Error processing image at {image_path}: {e}")
            continue

    # Calculate metrics
    metrics = evaluate_performance(true_labels, predictions)

    # Save metrics to a JSON file
    metrics_filename = f"../vision_models/test_metrics/test_metrics_{model_name}.json"
    with open(metrics_filename, "w") as file:
        json.dump(metrics, file)

    print(f"Metrics saved in '{metrics_filename}'.")

    if __name__ == "__main__":
        main()
