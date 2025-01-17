"""
This script calculates the accuracy, precision, and recall of the LLAVA model
"""
import json
import os
import pandas as pd

# Path to the directory
directory = (
    "../thesis-achkar/results/llava/llava_results_custom_1"
)

# test dataframe
df = pd.read_csv("../data/test_captions_and_desc.csv")

# Initialize counters for TP, TN, FP, FN
TP = TN = FP = FN = 0


def get_aligns_with_beauty_standards_value(model_data):
    if (
        model_data["model"] is not None
        and "aligns_with_beauty_standards" in model_data["model"]
    ):
        return model_data["model"]["aligns_with_beauty_standards"]
    else:
        return None


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
output_path = "../thesis-achkar/results/llava/llava_results_custom_1/llava_custom_1_final_metrics.txt"
with open(output_path, "w") as f:
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
output_path = "../thesis-achkar/results/llava/llava_results_custom_1/llava_custom_1_final_metrics_test.txt"
with open(output_path, "w") as f:
    f.write(f"True Positives: {TP}\n")
    f.write(f"True Negatives: {TN}\n")
    f.write(f"False Positives: {FP}\n")
    f.write(f"False Negatives: {FN}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")

print(f"Metrics have been saved to {output_path}.")
