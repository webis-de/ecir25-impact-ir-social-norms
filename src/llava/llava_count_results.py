"""
This script counts the number of True Positives, True Negatives, False Positives, and False Negatives
from the JSON files in the directory and calculates the Accuracy, Precision, and Recall.
"""
import json
import os
import pandas as pd

# Path to the directory
directory = "../results/llava/llava_results_3_img_caption_cust"

# test dataframe
df = pd.read_csv("../data/test_captions_and_desc.csv")

# Initialize counters for TP, TN, FP, FN
TP = TN = FP = FN = 0

# Function to safely get the value from nested dictionary
def get_aligns_with_beauty_standards_value(model_data):
    try:
        return model_data["model"]["aligns_with_beauty_standards"]
    except KeyError:
        return None  # Return None if the value is not found


# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    # Check if it's a file and it's a JSON file
    if os.path.isfile(file_path) and filename.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)

            # Go through each entry in the data
            for key, model_data in data.items():
                value = get_aligns_with_beauty_standards_value(model_data)
                if value is None:
                    continue  # Skip if value is not found

                # Determine the correct label based on the file name
                correct_label = 0 if "divers" in filename else 1

                # Count TP, TN, FP, FN
                if value == correct_label:
                    if correct_label == 1:
                        TP += 1  # True Positive
                    else:
                        TN += 1  # True Negative
                else:
                    if correct_label == 1:
                        FN += 1  # False Negative
                    else:
                        FP += 1  # False Positive

# Calculate Accuracy, Precision, and Recall
total = TP + TN + FP + FN
accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Write the results to a text file
output_path = "../results/llava/llava_results_3_img_caption_cust/llava_results_3_img_caption_cust_final_metrics.txt"
with open(output_path, "w") as f:
    f.write(f"Total: {total}\n")
    f.write(f"True Positives: {TP}\n")
    f.write(f"True Negatives: {TN}\n")
    f.write(f"False Positives: {FP}\n")
    f.write(f"False Negatives: {FN}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")

print(f"Metrics have been saved to {output_path}.")

# Test Metrics
ids = df["id"].tolist()
ids = [str(i) for i in ids]

# Initialize counters for TP, TN, FP, FN
TP = TN = FP = FN = 0

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    # Check if it's a file and it's a JSON file
    if os.path.isfile(file_path) and filename.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)

            # Go through each entry in the data
            for key, model_data in data.items():
                if key in ids:
                    value = get_aligns_with_beauty_standards_value(model_data)
                    if value is None:
                        continue  # Skip if value is not found

                    # Determine the correct label based on the file name
                    correct_label = 0 if "divers" in filename else 1

                    # Count TP, TN, FP, FN
                    if value == correct_label:
                        if correct_label == 1:
                            TP += 1  # True Positive
                        else:
                            TN += 1  # True Negative
                    else:
                        if correct_label == 1:
                            FN += 1  # False Negative
                        else:
                            FP += 1  # False Positive
                else:
                    continue

# Calculate Accuracy, Precision, and Recall
total = TP + TN + FP + FN
accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Write the results to a text file
output_path = "../results/llava/llava_results_3_img_caption_cust/llava_results_3_img_caption_cust_final_metrics_test.txt"
with open(output_path, "w") as f:
    f.write(f"Total: {total}\n")
    f.write(f"True Positives: {TP}\n")
    f.write(f"True Negatives: {TN}\n")
    f.write(f"False Positives: {FP}\n")
    f.write(f"False Negatives: {FN}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")

print(f"Metrics have been saved to {output_path}.")
