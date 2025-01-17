"""
This script is used to calculate the F1 score for the experiments of the LLAVA dataset.
"""
import pandas as pd

# Creating a DataFrame with the provided data
data_total = {
    "Exp": ["llava_1_img", "llava_1_img_cust", "llava_1_img_caption", "llava_1_img_caption_cust", 
            "llava_3_img", "llava_3_img_cust", "llava_3_img_caption", "llava_3_img_caption_cust"],
    "Model Type": ["Vision", "Vision", "Vision + Text", "Vision + Text", "Vision", "Vision", "Vision + Text", "Vision + Text"],
    "Total Accuracy": [0.72, 0.80, 0.74, 0.82, 0.79, 0.56, 0.81, 0.78],
    "Total Precision": [0.65, 0.73, 0.67, 0.76, 0.77, 0.55, 0.81, 0.79],
    "Total Recall": [0.99, 0.97, 0.98, 0.94, 0.85, 0.80, 0.82, 0.78]
}

data_test = {
    "Exp": ["llava_1_img", "llava_1_img_cust", "llava_1_img_caption", "llava_1_img_caption_cust", 
            "llava_3_img", "llava_3_img_cust", "llava_3_img_caption", "llava_3_img_caption_cust"],
    "Model Type": ["Vision", "Vision", "Vision + Text", "Vision + Text", "Vision", "Vision", "Vision + Text", "Vision + Text"],
    "Total Accuracy": [0.67, 0.79, 0.73, 0.83, 0.75, 0.48, 0.82, 0.85],
    "Total Precision": [0.61, 0.72, 0.65, 0.76, 0.75, 0.49, 0.80, 0.84],
    "Total Recall": [1.00, 0.98, 1.00, 0.96, 0.77, 0.72, 0.85, 0.87]
}


df_total = pd.DataFrame(data_total)
df_test = pd.DataFrame(data_test)

# add a new column to the DataFrame f1 score
df_total["Total F1 Score"] = 2 * (df_total["Total Precision"] * df_total["Total Recall"]) / (df_total["Total Precision"] + df_total["Total Recall"])
df_test["Total F1 Score"] = 2 * (df_test["Total Precision"] * df_test["Total Recall"]) / (df_test["Total Precision"] + df_test["Total Recall"])

# save the DataFrame to a csv file 
df_total.to_csv("../results/llava/llava_total_f1.csv", index=False)
df_test.to_csv("../results/llava/llava_test_f1.csv", index=False)
