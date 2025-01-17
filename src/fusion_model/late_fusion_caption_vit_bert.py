"""
This script is used to create an late fusion model using the caption and image features
"""
from transformers import (
    BertModel,
    ViTForImageClassification,
    BertTokenizer,
    ViTImageProcessor,
)
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from PIL import Image
import json

# Global Variables
MODEL_SAVE_PATH = "../fusion_model/caption_late_fusion_vit_bert_base_fine_tuned.pth"


def prepare_data(text, image_path, tokenizer, feature_extractor):
    """
    Tokenize text and convert them into tensors.

    Parameters
    ----------
    text : str
        Text to be tokenized.

    image_path : str
        Path to the image.

    tokenizer : transformers.BertTokenizer
        Tokenizer to be used for tokenizing the text.

    feature_extractor : transformers.ViTImageProcessor
        Feature extractor to be used for processing the image.

    Returns
    -------
    text_input : dict
        Tokenized text input, containing input_ids and attention_mask.

    image_input : dict
        Processed image input, containing pixel_values.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = " "

    # Tokenize text
    text_input = tokenizer(text, padding=False, truncation=True, return_tensors=None)

    # Resize and process image
    image = Image.open(image_path).convert("RGB")
    image_input = feature_extractor(images=image, return_tensors="pt")

    return text_input, image_input


def prepare_dataset(dataset, tokenizer, feature_extractor):
    """
    Prepare the dataset for training.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to be prepared.

    tokenizer : transformers.BertTokenizer
        Tokenizer to be used for tokenizing the text.

    feature_extractor : transformers.ViTImageProcessor
        Feature extractor to be used for processing the image.

    Returns
    -------
    text_inputs : list
        List of tokenized text inputs.

    image_inputs : list
        List of resized image inputs.

    torch.Tensor
        Labels.
    """
    # Pocess each text and image pair and store their processed forms
    text_inputs = []
    image_inputs = []
    labels = []

    for _, row in tqdm(dataset.iterrows()):
        text, image_path, label = row["caption"], row["image_path"], row["label"]
        text_input, image_input = prepare_data(
            text, image_path, tokenizer, feature_extractor
        )

        # Debugging: Print shapes and types
        # print(f"Text input IDs shape: {len(text_input['input_ids'])}, Type: {type(text_input['input_ids'])}")
        # print(f"Image input shape: {image_input['pixel_values'].shape}, Type: {type(image_input['pixel_values'])}")

        text_inputs.append(text_input)
        image_inputs.append(image_input)
        labels.append(label)

    return text_inputs, image_inputs, torch.tensor(labels)


def custom_collate_fn(batch, tokenizer):
    """
    Custom collate function for batching text and image data.

    This function is used to dynamically pad the text inputs to the same length and to batch
    the image data. It is passed to the DataLoader.

    Reference for pad_sequence:
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

    Parameters
    ----------
    batch : list
        A list of tuples, where each tuple contains the text data, image data,
                    and label for a single sample.

    tokenizer : transformers.BertTokenizer
        Tokenizer to be used for tokenizing the text.

    Returns
    -------
    dict, torch.Tensor, torch.Tensor: A
        batch of text data (input_ids and attention masks), image data, and labels.
    """
    text_inputs, image_inputs, labels = zip(*batch)

    # Pad text inputs
    input_ids = pad_sequence(
        [torch.tensor(text["input_ids"]) for text in text_inputs],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = pad_sequence(
        [torch.tensor(text["attention_mask"]) for text in text_inputs],
        batch_first=True,
        padding_value=0,
    )

    # Stack image inputs and labels
    image_inputs = torch.stack([img["pixel_values"].squeeze(0) for img in image_inputs])
    labels = torch.tensor(labels)

    return (
        {"input_ids": input_ids, "attention_mask": attention_masks},
        image_inputs,
        labels,
    )


class TextImageDataset(Dataset):
    """
    A custom Dataset class for handling text and image data together.
    The class inherits from torch.utils.data.Dataset.
    Reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    texts : list
        List of tokenized text inputs.

    images : list
        List of image tensors.

    labels : torch.Tensor
        Labels corresponding to the text and image inputs.

    Returns
    -------
    torch.utils.data.Dataset
    """

    def __init__(self, texts, images, labels):
        """
        Initializes the TextImageDataset.
        """
        self.texts = texts
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.

        Parameters:
            idx (int): The index of the item to be retrieved.

        Returns:
            tuple: A tuple containing the tokenized text, image tensor, and label for the given index.
        """
        text_input = self.texts[idx]
        image_input = self.images[idx]
        label = self.labels[idx]
        return text_input, image_input, label


class FusionModel(torch.nn.Module):
    """
    A custom model that fuses the outputs of BERT and ViT.

    Parameters
    ----------
    bert_model : transformers.BertModel
        A pretrained BERT model.

    vit_model : transformers.ViTModel
        A pretrained ViT model.

    num_labels : int
        The number of labels in the dataset.
    """

    def __init__(self, bert_model, vit_model, num_labels=2):
        super(FusionModel, self).__init__()
        self.bert = bert_model
        self.vit = vit_model
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size + self.vit.config.hidden_size, num_labels
        )

    def forward(self, text_input, image_input):
        """
        Defines the forward pass of the FusionModel.
        The method processes the input text and image through their respective models (BERT and ViT),
        extracts the pooler_output from both, concatenates them, and passes the combined vector
        through a classifier to obtain the final logits.

        Parameters
        ----------
            text_input (dict):
                The tokenized input text data. It contains 'input_ids' and 'attention_mask'.
            image_input (Tensor):
                The processed input image data.

        Returns
        -------
            Tensor:
                The logits representing the output predictions of the model.
        """
        # Process text input through BERT and use the last hidden state
        # Bert provides last_hidden_state, pooler_output, hidden_states, and attentions as possible outputs.
        # last_hidden_state[:, 0] extracts the hidden state of the first token ([CLS] token) from the last layer. This represents the transformed [CLS] token embedding, capturing the essence of the input text.
        text_outputs = self.bert(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
        ).last_hidden_state[:, 0]

        # Process image input through ViT and request hidden states
        # ViT provides several outputs including last_hidden_state, pooler_output, hidden_states, and attentions.
        image_outputs = self.vit(
            pixel_values=image_input, output_hidden_states=True
        ).hidden_states[-1][:, 0]

        # Concatenate the outputs
        combined = torch.cat((text_outputs, image_outputs), dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits


def train_model(model, dataloader, optimizer, loss_fn):
    """
    Trains the given model for one epoch.

    Parameters
    ----------
    model : FusionModel
        The model to be trained.

    dataloader : torch.utils.data.DataLoader
        The DataLoader object containing the training data.

    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.

    loss_fn : torch.nn.modules.loss
        The loss function to be used for training.

    Returns
    -------
    float
        The average training loss for the epoch.
    """
    # Check if GPU is available and set the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the selected device
    # model = model.to(device)

    model.train()  # Set the model to training mode
    total_train_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        text_input, image_input, labels = batch
        # Move data to the same device as the model
        # text_input = {k: v.to(device) for k, v in batch[0].items()}  # Assuming batch[0] is text_input
        # image_input = batch[1].to(device)  # Adjust according to your dataloader structure
        # labels = batch[2].to(device)       # Adjust according to your dataloader structure

        optimizer.zero_grad()  # Clear previous gradients

        # Forward pass
        logits = model(text_input, image_input)
        loss = loss_fn(logits.view(-1, model.num_labels), labels.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    avg_train_loss = total_train_loss / len(dataloader)
    train_accuracy = total_correct / total_samples
    return avg_train_loss, train_accuracy


def validate_model(model, dataloader, loss_fn):
    """
    Validates the given model.

    Parameters
    ----------
    model : FusionModel
        The model to be validated.

    dataloader : torch.utils.data.DataLoader
        The DataLoader object containing the validation data.

    loss_fn : torch.nn.modules.loss
        The loss function to be used for validation.

    Returns
    -------
    avg_val_loss : float
        The average validation loss.

    avg_accuracy : float
        The average validation accuracy.
    """
    # Check if GPU is available and set the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the selected device
    # model = model.to(device)

    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_accuracy = 0

    with torch.no_grad():  # No gradient computation
        for batch in tqdm(dataloader, desc="Validation"):
            text_input, image_input, labels = batch
            # Move data to the same device as the model
            # text_input = {k: v.to(device) for k, v in batch[0].items()}  # Assuming batch[0] is text_input
            # image_input = batch[1].to(device)  # Adjust according to your dataloader structure
            # labels = batch[2].to(device)       # Adjust according to your dataloader structure

            # Forward pass
            logits = model(text_input, image_input)
            loss = loss_fn(logits.view(-1, model.num_labels), labels.view(-1))

            total_val_loss += loss.item()

            # Calculate accuracy
            preds = logits.argmax(dim=1)
            total_accuracy += (preds == labels).sum().item()

    avg_val_loss = total_val_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader.dataset)
    return avg_val_loss, avg_accuracy


def test_model(model, dataloader):
    """
    Tests the given model and calculates accuracy, recall and precision.

    Parameters
    ----------
    model : FusionModel
        The model to be tested.

    dataloader : torch.utils.data.DataLoader
        The DataLoader object containing the test data.

    Returns
    -------
    avg_test_accuracy : float
        The average test accuracy.

    recall : float
        The overall recall across all classes.

    precision : float
        The overall precision across all classes.
    """
    total_correct = 0
    total_samples = len(dataloader.dataset)
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            text_input, image_input, labels = batch

            logits = model(text_input, image_input)
            preds = logits.argmax(dim=1)

            # if preds == labels, then correct += 1
            total_correct += (preds == labels).sum().item()

            # if preds == labels == 1, then TP += 1
            total_TP += ((preds == labels) & (preds == 1)).sum().item()

            # if preds == labels == 0, then TN += 1
            total_TN += ((preds == labels) & (preds == 0)).sum().item()

            # if preds != labels == 1, then FP += 1
            total_FP += ((preds != labels) & (preds == 1)).sum().item()

            # if preds != labels == 0, then FN += 1
            total_FN += ((preds != labels) & (preds == 0)).sum().item()

    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    accuracy = (
        (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        if (total_TP + total_TN + total_FP + total_FN) > 0
        else 0
    )

    return accuracy, recall, precision


# Main Function
def main():
    # Load models, tokenizers, and feature extractors
    bert_model = BertModel.from_pretrained(
        "../bert_models/caption_bert-base-multilingual-uncased-sentiment_fine_tuned"
    )
    vit_model = ViTForImageClassification.from_pretrained(
        "../vision_models/vit-base-patch16-384_384_folder_raw_fine_tuned"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "../bert_models/caption_bert-base-multilingual-uncased-sentiment_fine_tuned"
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        "../vision_models/vit-base-patch16-384_384_folder_raw_fine_tuned"
    )

    # Read and prepare datasets
    train_data = pd.read_csv(
        "../train_captions_and_desc_with_image_path.csv"
    )
    train_data = train_data[["id", "caption", "image_path", "label"]]
    test_data = pd.read_csv(
        "../test_captions_and_desc_with_image_path.csv"
    )
    test_data = test_data[["id", "caption", "image_path", "label"]]
    val_data = pd.read_csv(
        "../valid_captions_and_desc_with_image_path.csv"
    )
    val_data = val_data[["id", "caption", "image_path", "label"]]

    # Prepare each dataset
    train_text, train_images, train_labels = prepare_dataset(
        train_data, tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    val_text, val_images, val_labels = prepare_dataset(
        val_data, tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    test_text, test_images, test_labels = prepare_dataset(
        test_data, tokenizer=tokenizer, feature_extractor=feature_extractor
    )

    # Create Dataset objects
    train_dataset = TextImageDataset(train_text, train_images, train_labels)
    val_dataset = TextImageDataset(val_text, val_images, val_labels)
    test_dataset = TextImageDataset(test_text, test_images, test_labels)

    # Prepare DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
    )

    # Initialize Fusion Model
    fusion_model = FusionModel(bert_model, vit_model)

    # Metrics
    epoch_metrics = {
        "train_loss": [],
        "train_accuracy": [],  #
        "val_loss": [],
        "val_accuracy": [],
    }

    # Training and Validation
    optimizer = AdamW(fusion_model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    epochs = 4
    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(
            fusion_model, train_dataloader, optimizer, loss_fn
        )
        val_loss, val_accuracy = validate_model(fusion_model, val_dataloader, loss_fn)

        # Update metrics
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["train_accuracy"].append(train_accuracy)
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["val_accuracy"].append(val_accuracy)

        # Print metrics for this epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Training Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}\n"
        )

        # Save metrics to a JSON file
        metrics_file = f"../fusion_model/metrics/train_validate_metrics_epoch_{epoch+1}_caption_vit384_bert_fine_tuned_fusion.json"
        with open(metrics_file, "w") as file:
            json.dump(epoch_metrics, file, indent=4)
        print(f"Metrics saved to {metrics_file}")

    # Test the model on the test dataset
    test_accuracy, test_recall, test_precision = test_model(
        fusion_model, test_dataloader
    )
    print(f"Test Metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Recall: {json.dumps(test_recall, indent=4)}")
    print(f"Precision: {json.dumps(test_precision, indent=4)}\n")

    # Save the metrics to a file
    metrics = {
        "accuracy": test_accuracy,
        "recall": test_recall,
        "precision": test_precision,
    }
    with open(
        "../fusion_model/metrics/test_metrics_caption_vit384_bert_fusion_fine_tuned.json",
        "w",
    ) as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved to test_metrics.json")

    # Save Model
    torch.save(fusion_model, MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
