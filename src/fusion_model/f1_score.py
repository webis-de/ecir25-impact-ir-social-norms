"""
This script is used to calculate the F1 score of the models and save the results in a csv file.
"""
import pandas as pd

data = {
    "Model": ["caption_vit384_bert_base", "caption_vit384_bert_fine_tuned"],
    "Learning Rate": [5e-5, 5e-5],
    "Training Loss": [0.001, 0.02],
    "Validation Loss": [0.32, 0.36],
    "Validation Accuracy": [0.89, 0.86],
    "Test Accuracy": [0.88, 0.92],
    "Test Precision": [0.89, 0.91],
    "Test Recall": [0.87, 0.93]
}

df = pd.DataFrame(data)

# add a new column to the DataFrame f1 score
df["Test F1 Score"] = 2 * (df["Test Precision"] * df["Test Recall"]) / (df["Test Precision"] + df["Test Recall"])
                                                                        
# save the DataFrame to a csv file in /Users/pierreachkar/Documents/thesis-achkar/results/caption
df.to_csv("../results/late_fusion_model/total_results.csv", index=False)
                                                                        