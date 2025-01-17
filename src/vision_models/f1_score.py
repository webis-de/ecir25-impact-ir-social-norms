"""
This script calculates the F1 score for each model and dataset combination and saves the results to a CSV file.
"""
import pandas as pd

# Data
data = {
    'Model': ['RegNet_224', 'RegNet_224', 'RegNet_384', 'RegNet_384', 'RegNet_512', 'RegNet_512', 'ResNet_224', 'ResNet_224', 'ResNet_384', 'ResNet_384', 'ResNet_512', 'ResNet_512', 'ViT_224', 'ViT_224', 'ViT_384', 'ViT_384', 'Swin_224', 'Swin_224', 'Swin_384', 'Swin_384', 'Clip_336'],
    'Dataset': ['raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw', 'aug', 'raw'],
    'Train Runtime': [486.83, 4727.00, 493.54, 3787.36, 556.19, 3864.71, 390.53, 2561.90, 374.59, 2562.85, 305.97, 2071.81, 704.62, 1302.00, 787.94, 7097.82, 703.08, 4282.59, 787.94, 7097.82, None],
    'Training Loss': [0.20, 0.10, 0.18, 0.09, 0.18, 0.09, 0.67, 0.46, 0.68, 0.41, 0.68, 0.40, 0.20, 0.09, 0.14, 0.05, 0.20, 0.08, 0.18, 0.05, None],
    'Validation Loss': [0.47, 0.48, 0.29, 0.47, 0.31, 0.42, 0.69, 0.50, 0.68, 0.46, 0.68, 0.39, 0.58, 0.62, 0.24, 0.31, 0.23, 0.48, 0.14, 0.31, None],
    'Validation Accuracy': [0.80, 0.87, 0.87, 0.86, 0.90, 0.86, 0.58, 0.75, 0.65, 0.81, 0.67, 0.85, 0.85, 0.78, 0.89, 0.91, 0.92, 0.87, 0.94, 0.91, None],
    'Test Accuracy': [0.78, 0.84, 0.87, 0.86, 0.86, 0.86, 0.65, 0.77, 0.59, 0.78, 0.59, 0.79, 0.75, 0.73, 0.84, 0.84, 0.80, 0.84, 0.84, 0.84, 0.49],
    'Test Precision': [0.75, 0.82, 0.87, 0.85, 0.84, 0.83, 0.65, 0.77, 0.60, 0.78, 0.59, 0.78, 0.71, 0.70, 0.82, 0.82, 0.75, 0.80, 0.81, 0.83, 0.50],
    'Test Recall': [0.87, 0.87, 0.87, 0.87, 0.89, 0.91, 0.83, 0.79, 0.57, 0.81, 0.64, 0.83, 0.87, 0.83, 0.87, 0.87, 0.94, 0.91, 0.89, 0.85, 0.83]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the difference between training and validation loss for each model
df['Loss Difference (Val-Train)'] = df['Validation Loss'] - df['Training Loss']

df['Test F1 Score'] = 2 * (df['Test Precision'] * df['Test Recall']) / (df['Test Precision'] + df['Test Recall'])

# round to 2 decimal places
df = df.round(2)

# make the following order of columns: Model, Dataset, Train Runtime, Training Loss, Validation Loss, Loss Difference, Validation Accuracy, Test Accuracy, Test Precision, Test Recall, Test F1 Score
df = df[['Model', 'Dataset', 'Train Runtime', 'Training Loss', 'Validation Loss', 'Loss Difference (Val-Train)', 'Validation Accuracy', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score']]

df.to_csv('../results/vision_models/test_metrics/all_results.csv', index=False)