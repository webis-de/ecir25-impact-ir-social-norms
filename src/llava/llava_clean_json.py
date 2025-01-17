"""
This script cleans the llava JSON data by flattening the 'model' field and correcting boolean values.
"""
import json
import os

def clean_model_field(data):
    """
    Cleans and validates the 'model' field in the JSON data, correcting boolean values.

    Parameters
    ----------
    data : dict
        The dictionary containing the data to be cleaned.

    Returns
    -------
    dict
        Updated dictionary with cleaned data.
    """
    for key, value in data.items():
        try:
            if isinstance(value["model"], str):
                # Correcting the boolean values, replace TRUE or true with 1, and FALSE or false with 0
                corrected_str = value["model"].replace("TRUE", "1").replace(
                    "true", "1").replace("True", "1"
                )
                corrected_str = corrected_str.replace("FALSE", "0").replace(
                    "false", "0").replace("False", "0"
                )
                # Correcting the JSON string, replace \n with newline, \\ with \, and \_ with _
                corrected_str = (
                    corrected_str.replace("\\n", "\n")
                    .replace("\\\\", "\\")
                    .replace("\\_", "_")
                )

                # Attempt to decode the JSON string
                data[key]["model"] = json.loads(corrected_str)
            elif isinstance(value["model"], dict):
                # The model field is already a dict, so no action needed
                pass
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON for key {key}: {e}")
            # Handle other potential issues here
            data[key]["model"] = None  # or use a placeholder like {}
        except Exception as e:
            print(f"Unexpected error for key {key}: {e}")
            data[key]["model"] = None  # or use a placeholder like {}

    return data


def results_1_update_boolean_values(data):
    """
    Updates 'boolean_aligns_with_beauty_standards' values in the JSON data.

    Parameters
    ----------
    data : dict
        The dictionary containing the data to be updated.

    Returns
    -------
    None
    """
    for key in data:
        model_data = data[key].get("model", {})
        boolean_value = model_data.get("boolean_aligns_with_beauty_standards", None)

        # Check if the value is a string and then convert to lowercase
        if isinstance(boolean_value, str):
            boolean_value = boolean_value.lower()

        # Update values based on boolean logic
        if boolean_value in ["true", True]:
            model_data["boolean_aligns_with_beauty_standards"] = 1
        elif boolean_value in ["false", False]:
            model_data["boolean_aligns_with_beauty_standards"] = 0


def results_2_update_beauty_norms_alignment(data):
    try:
        for key, model_data in data.items():
            beauty_norms_alignment = model_data["model"]["beauty_norms_alignment"][
                "beauty_norms_alignment"
            ]

            # Trim spaces and convert to lowercase for comparison
            beauty_norms_alignment = str(beauty_norms_alignment).strip().lower()

            if beauty_norms_alignment == "true":
                model_data["model"]["beauty_norms_alignment"][
                    "beauty_norms_alignment"
                ] = 1
            elif beauty_norms_alignment == "false":
                model_data["model"]["beauty_norms_alignment"][
                    "beauty_norms_alignment"
                ] = 0
            else:
                # Log entries that don't match expected values
                print(
                    f"Key {key} has an unexpected beauty_norms_alignment value: '{beauty_norms_alignment}'"
                )
    except Exception as e:
        print(f"An error occurred: {e}")

    return data


def results_3_update_boolean_values(data):
    """
    Updates 'boolean_aligns_with_beauty_standards' values in the JSON data.

    Parameters
    ----------
    data : dict
        The dictionary containing the data to be updated.

    Returns
    -------
    None
    """
    for key in data:
        model_data = data[key].get("model", {})
        boolean_value = model_data.get("align_with_beauty_standards", None)

        # Check if the value is a string and then convert to lowercase
        if isinstance(boolean_value, str):
            boolean_value = boolean_value.lower()

        # Update values based on boolean logic
        if boolean_value in ["true", True]:
            model_data["align_with_beauty_standards"] = 1
        elif boolean_value in ["false", False]:
            model_data["align_with_beauty_standards"] = 0


# Directory path
directory = "../results/llava/llava_results_3_img_caption_cust"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        clean_model_field(data)
        # results_3_update_boolean_values(data)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Processed and cleaned {filename}")
