"""
This script checks if each key in a json file has model.align_with_beauty_standards, if not it prints the key. Also, it checks if the value of model.align_with_beauty_standards is an integer, if not it prints the key.
"""
import json

def check_keys(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        for key in data:
            model_data = data[key].get("model", {})
            align_with_beauty_standards = model_data.get("aligns_with_beauty_standards", None)
            if align_with_beauty_standards is None:
                print(key)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def check_align_with_beauty_standards(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        for key in data:
            model_data = data[key].get("model", {})
            align_with_beauty_standards = model_data.get("aligns_with_beauty_standards", None)
            if align_with_beauty_standards is not None and not isinstance(align_with_beauty_standards, int):
                print(key)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


file_path = "../results/llava/llava_results_3_img_caption_cust/prompt_3_img_caption_cust_norm_beauty.json"
print("Checking keys...")
check_keys(file_path)
print("--------------------")
print("Checking align_with_beauty_standards...")
check_align_with_beauty_standards(file_path)
print("--------------------")
