"""
This script is used to evaluate the zero-shot performance of CLIP 336.
"""
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm.auto import tqdm

def initialize_model(model_id="openai/clip-vit-large-patch14-336"):
    """
    Initialize the CLIP model and processor.

    Parameters
    ----------
    model_id : str
        The model id to use. Defaults to "openai/clip-vit-large-patch14-336".

    Returns
    -------
    processor : CLIPProcessor
        The CLIP processor.
    model : CLIPModel
        The CLIP model.
    device : str
        The device to use. Defaults to "cuda" if available, otherwise "cpu".
    """
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def load_data(data_dir="../raw"):
    """
    Load the dataset.

    Parameters
    ----------
    data_dir : str
        The directory where the dataset is stored.

    Returns
    -------
    train_dataset : DatasetDict
        The dataset.
    """
    return load_dataset("imagefolder", data_dir=data_dir)

def process_labels(train_dataset, processor, model, device):
    """
    Process the labels and generate embeddings.

    Parameters
    ----------
    train_dataset : DatasetDict
        The dataset.
    processor : CLIPProcessor
        The CLIP processor.
    model : CLIPModel
        The CLIP model.
    device : str
        The device to use.

    Returns
    -------
    label_emb : np.ndarray
        The label embeddings.
    """
    labels = train_dataset['train'].info.features['label'].names
    clip_labels = [f"a photo of a {label} instagram post" for label in labels]
    
    label_tokens = processor(text=clip_labels, padding=True, return_tensors='pt').to(device)
    label_emb = model.get_text_features(**label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
    return label_emb

def process_image(image, processor, device):
    """
    Process the image.

    Parameters
    ----------
    image : str
        The path to the image.
    processor : CLIPProcessor
        The CLIP processor.
    device : str
        The device to use.

    Returns
    -------
    processed_image : torch.Tensor
        The processed image.
    """
    processed_image = processor(text=None, images=image, return_tensors='pt')['pixel_values']
    if isinstance(processed_image, list):
        processed_image = torch.tensor(processed_image)
    return processed_image.to(device)

def predict_image_label(image, model, label_emb, processor, device):
    """
    Predict the label of an image.

    Parameters
    ----------
    image : str
        The path to the image.
    model : CLIPModel
        The CLIP model.
    label_emb : np.ndarray
        The label embeddings.
    processor : CLIPProcessor
        The CLIP processor.
    device : str
        The device to use.

    Returns
    -------
    pred : int
        The predicted label.    
    """
    image = process_image(image, processor, device)
    img_emb = model.get_image_features(image)
    img_emb = img_emb.detach().cpu().numpy()
    scores = np.dot(img_emb, label_emb.T)
    return np.argmax(scores)

def batch_predict(train_dataset, processor, model, label_emb, device, batch_size=32):
    """
    Predict the labels of a batch of images.

    Parameters
    ----------
    train_dataset : DatasetDict
        The dataset.
    processor : CLIPProcessor
        The CLIP processor.
    model : CLIPModel
        The CLIP model.
    label_emb : np.ndarray
        The label embeddings.
    device : str
        The device to use.
    batch_size : int
        The batch size. Defaults to 32.

    Returns
    -------
    preds : list
        The predicted labels.
    """
    preds = []
    for i in tqdm(range(0, len(train_dataset['train']), batch_size)):
        i_end = min(i + batch_size, len(train_dataset['train']))
        images = [train_dataset['train'][j]['image'] for j in range(i, i_end)]
        for image in images:
            pred = predict_image_label(image, model, label_emb, processor, device)
            preds.append(pred)
    return preds

def calculate_accuracy(train_dataset, preds):
    """
    Calculate the accuracy.

    Parameters
    ----------
    train_dataset : DatasetDict
        The dataset.
    preds : list
        The predicted labels.

    Returns
    -------
    accuracy : float
        The accuracy.
    """
    true_preds = [1 if label == preds[i] else 0 for i, label in enumerate(train_dataset['train']['label'])]
    return sum(true_preds) / len(true_preds)

def calculate_precision_recall(train_dataset, preds):
    """
    Calculate the precision and recall.

    Parameters
    ----------
    train_dataset : DatasetDict
        The dataset.
    preds : list
        The predicted labels.

    Returns
    -------
    precision : float
        The precision.
    recall : float
        The recall.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i, label in enumerate(train_dataset['train']['label']):
        if label == preds[i]:
            if label == 1:  # Assuming label 1 is 'positive'
                true_positive += 1
        else:
            if preds[i] == 1:
                false_positive += 1
            if label == 1:
                false_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return precision, recall

def main():
    # Initialize model and processor
    processor, model, device = initialize_model()

    # Load the dataset
    train_dataset = load_data()

    # Process labels and generate embeddings
    label_emb = process_labels(train_dataset, processor, model, device)

    # Batch prediction
    preds = batch_predict(train_dataset, processor, model, label_emb, device)

    # Calculate accuracy, precision, and recall
    accuracy = calculate_accuracy(train_dataset, preds)

    precision, recall = calculate_precision_recall(train_dataset, preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    main()