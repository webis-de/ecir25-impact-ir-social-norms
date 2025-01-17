"""
This script is used to fine-tune the BERT model for sequence classification.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import json


def load_data(filepath, text):
    """
    Load and preprocess data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    data : pd.DataFrame
        Preprocessed DataFrame.
    """
    data = pd.read_csv(filepath)
    data = data[["id", text, "label"]]
    data = data.dropna()
    return data


def preprocess_text(text, tokenizer):
    """
    Tokenize and encode text using BERT tokenizer.

    Parameters
    ----------
    text : str
        Text to be tokenized and encoded.
    tokenizer : BertTokenizer
        Tokenizer to be used.

    Returns
    -------
    encoding : dict
        Dictionary containing input_ids, attention_mask.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoding


def create_data_loaders(train_set, val_set, batch_size):
    """
    Create DataLoaders for training and validation sets.

    Parameters
    ----------
    train_set : TensorDataset
        Training dataset.
    val_set : TensorDataset
        Validation dataset.
    batch_size : int
        Batch size for the dataloaders.

    Returns
    -------
    train_dataloader, validation_dataloader : tuple
        Training and validation DataLoader.
    """
    train_dataloader = DataLoader(
        train_set, sampler=RandomSampler(train_set), batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_set, sampler=SequentialSampler(val_set), batch_size=batch_size
    )
    return train_dataloader, validation_dataloader


def train_model(model, train_dataloader, device, optimizer):
    """
    Train the BERT model.

    Parameters
    ----------
    model : BertForSequenceClassification
        BERT model for sequence classification.
    train_dataloader : DataLoader
        DataLoader for training data.
    device : torch.device
        Device to run the model on.
    optimizer : torch.optim
        Optimizer for the model.

    Returns
    -------
    average_loss : float
        Average training loss.
    """
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()

        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    return average_loss


def b_metrics(preds, labels):
    """
    Calculate performance metrics based on predictions and actual labels.

    Parameters
    ----------
    preds : np.ndarray
        Predicted labels.
    labels : np.ndarray
        Actual labels.

    Returns
    -------
    accuracy, precision, recall : tuple
        Accuracy, precision and recall.
    """
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall


def validate_model(model, validation_dataloader, device):
    """
    Validate the model on the validation set.

    Parameters
    ----------
    model : BertForSequenceClassification
        BERT model for sequence classification.
    validation_dataloader : DataLoader
        DataLoader for validation data.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    val_accuracy, val_precision, val_recall : tuple
        Validation accuracy, precision and recall.
    """
    model.eval()
    val_accuracy = []
    val_precision = []
    val_recall = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        b_accuracy, b_precision, b_recall = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        val_precision.append(b_precision)
        val_recall.append(b_recall)

    return np.mean(val_accuracy), np.mean(val_precision), np.mean(val_recall)


def test_model(model, test_data, text, tokenizer, device):
    """
    Test the model on new data and update the DataFrame with predictions.

    Parameters
    ----------
    model : BertForSequenceClassification
        BERT model for sequence classification.
    test_data : pd.DataFrame
        DataFrame containing test data.
    tokenizer : BertTokenizer
        Tokenizer to be used.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    test_data : pd.DataFrame
        Updated DataFrame with predictions.
    """
    model.eval()
    test_data["prediction"] = None

    for index, row in test_data.iterrows():
        encoding = preprocess_text(row[text], tokenizer)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(
                input_ids, token_type_ids=None, attention_mask=attention_mask
            )

        prediction = np.argmax(output.logits.cpu().numpy()).flatten().item()
        test_data.at[index, "prediction"] = prediction

    return test_data


def calculate_metrics_test_data(df):
    """
    Calculate the accuracy of predictions in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame)
        DataFrame containing 'label' and 'prediction' columns.

    Returns
    -------
    accuracy, precision, recall : tuple
        Calculated accuracy, precision and recall.
    """
    correct_predictions = (df["prediction"] == df["label"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    tp = ((df["prediction"] == 1) & (df["label"] == 1)).sum()
    fp = ((df["prediction"] == 1) & (df["label"] == 0)).sum()
    fn = ((df["prediction"] == 0) & (df["label"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall


def save_results_to_json(
    train_loss,
    val_accuracy,
    val_precision,
    val_recall,
    test_accuracy,
    test_precision,
    test_recall,
    file_path,
):
    """
    Save the training, validation, and test results to a JSON file.

    Parameters
    ----------
    train_loss : float
        Training loss from the last epoch.
    val_accuracy : float
        Validation accuracy from the last epoch.
    val_precision : float
        Validation precision from the last epoch.
    val_recall : float
        Validation recall from the last epoch.
    test_accuracy : float
        Test accuracy.
    test_precision : float
        Test precision.
    test_recall : float
        Test recall.
    file_path : str
        Path to the JSON file to save results.

    Returns
    -------
    None
    """
    results = {
        "train_loss": train_loss,
        "validation_accuracy": val_accuracy,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }

    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)


def save_model(model, tokenizer, save_path):
    """
    Save the model and tokenizer.

    Parameters
    ----------
    model : BertForSequenceClassification
        The trained BERT model.
    tokenizer : BertTokenizer
        The tokenizer used for the model.
    save_path : str
        The path to save the model and tokenizer.

    Returns
    -------
    None
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def main():
    """
    Main function to orchestrate the workflow of training, validating, and testing the BERT model.
    """
    # Load and preprocess data
    train_filepath = "../train_captions_and_desc.csv"
    test_filepath = "../test_captions_and_desc.csv"
    val_filepath = "../valid_captions_and_desc.csv"
    text = "reformulated_caption"
    train_data = load_data(train_filepath, text)
    test_data = load_data(test_filepath, text)
    val_data = load_data(val_filepath, text)

    # Model
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    # Split the string at the slash and take the second part
    split_name = model_name.split("/")
    if len(split_name) > 1:
        save_name = split_name[1]
    else:
        save_name = split_name[0]

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)

    # Preprocess the text for train and val data
    train_texts = train_data[text].values
    train_labels = train_data["label"].values
    val_texts = val_data[text].values
    val_labels = val_data["label"].values

    # Tokenization and encoding of the dataset
    train_encodings = [preprocess_text(text, tokenizer) for text in train_texts]
    val_encodings = [preprocess_text(text, tokenizer) for text in val_texts]

    # Convert to tensors
    train_input_ids = torch.cat([enc["input_ids"] for enc in train_encodings], dim=0)
    train_attention_masks = torch.cat(
        [enc["attention_mask"] for enc in train_encodings], dim=0
    )
    train_labels = torch.tensor(train_labels)
    val_input_ids = torch.cat([enc["input_ids"] for enc in val_encodings], dim=0)
    val_attention_masks = torch.cat(
        [enc["attention_mask"] for enc in val_encodings], dim=0
    )
    val_labels = torch.tensor(val_labels)

    # Create TensorDataset for train and validation
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    # Create DataLoader
    batch_size = 16
    train_dataloader, validation_dataloader = create_data_loaders(
        train_dataset, val_dataset, batch_size
    )

    # Initialize model and optimizer
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and Validation
    epochs = 4
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_model(model, train_dataloader, device, optimizer)
        print(f"Training loss: {train_loss}")

        val_accuracy, val_precision, val_recall = validate_model(
            model, validation_dataloader, device
        )
        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Validation Precision: {val_precision}")
        print(f"Validation Recall: {val_recall}")
        print("----")

    # Testing
    updated_test_data = test_model(model, test_data, text, tokenizer, device)
    test_accuracy, test_precision, test_recall = calculate_metrics_test_data(
        updated_test_data
    )
    print("\n")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print("----")

    # Save model
    save_path = f"../bert_models/{text}_{save_name}_fine_tuned"
    save_model(model, tokenizer, save_path)
    print(f"Model saved to {save_path}")
    print("\n")

    # Save results to JSON
    results_path = (
        f"../bert_models/metrics/{text}_{save_name}_results.json"
    )
    save_results_to_json(
        train_loss,
        val_accuracy,
        val_precision,
        val_recall,
        test_accuracy,
        test_precision,
        test_recall,
        results_path,
    )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
