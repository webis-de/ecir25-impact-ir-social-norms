"""
This script is used to classify images into 'norm_beauty' and 'divers' categories using the LLAVA model based on the image and the caption.
"""
import os
import json
import torch
from PIL import Image
from io import BytesIO
from typing import Dict, Tuple
from textwrap import fill
from tqdm import tqdm
import pandas as pd

# Import necessary modules from llava library
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates, Conversation
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# Disable torch's automatic initialization to save memory
disable_torch_init()

# Global Variables
MODEL = "4bit/llava-v1.5-13b-3GB"  # https://huggingface.co/4bit/llava-v1.5-13b-3GB
CONV_MODE = "llava_v0"
IMAGE_DIRECTORY = "../raw/norm_beauty"
RESULT_JSON_PATH = "../llava_results/missing_prompt_2_cust_img_caption_norm_beauty.json"


def load_model(model_path):
    """
    load the model, tokenizer and image processor.

    Parameters
    ----------
    model_path : str
        path to the model

    Returns
    -------
    Tuple
        tokenizer, model, image_processor
    """
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name=model_name, load_4bit=True
    )
    return tokenizer, model, image_processor


def process_image(image: Image, image_processor, model):
    """
    process the image and convert it to a tensor.

    Parameters
    ----------
    image : Image
        PIL image

    Returns
    -------
    torch.Tensor
        image tensor
    """
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)


def create_prompt(prompt):
    """
    create the prompt for the model.

    Parameters
    ----------
    prompt : str
        prompt for the model

    Returns
    -------
    Tuple[str, Dict]
        prompt, conversation
    """
    # For Custom Instructions Use Steps 1-3
    # Step 1: Define your custom system prompt
    custom_system_prompt = "In the role of a social scientist, you are tasked with classifying social media images into 'norm_beauty' and 'divers' categories. 'norm_beauty' images typically feature conventional beauty standards and poses, while 'divers' images represent a wider spectrum of human diversity, including unique physical traits and unconventional styles. Analyze these images critically, focusing on their representation of societal norms and diversity."
    # Step 2: Create a new instance of Conversation with your custom prompt
    custom_conversation = Conversation(
        system=custom_system_prompt,
        roles=("Human", "AI Social Scientist"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    # Step 3: Append your custom prompt to the conversation
    conv = custom_conversation.copy()
    # For Default Instructions Use Step 4
    # Step 4: Use the default prompt
    # conv = conv_templates[CONV_MODE].copy()

    # Same for both Custom and Default Instructions
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def ask_image(tokenizer, model, image_processor, image: Image, prompt: str):
    """
    ask the model to describe the image.

    Parameters
    ----------
    tokenizer : tokenizer
        tokenizer

    model : model
        model

    image_processor : image_processor
        image processor

    image : Image
        PIL image

    prompt : str
        prompt for the model

    Returns
    -------
    str
        description of the image
    """
    image_tensor = process_image(image, image_processor, model)
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def is_image_file(filename: str):
    """
    Check if a file is an image based on its extension.

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    bool
        True if the file is an image, False otherwise
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    return os.path.splitext(filename)[1].lower() in valid_extensions


def save_results_to_json(results, file_path):
    """
    Save the results to a JSON file.

    Parameters
    ----------
    results : dict
        results dictionary

    file_path : str
        path to the JSON file

    Returns
    -------
    None
    """
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    # Load the model
    tokenizer, model, image_processor = load_model(MODEL)

    # Load captions
    df = pd.read_csv("../total_captions_and_desc.csv")
    df.set_index("id", inplace=True)

    # Initialize the results dictionary
    results = {}

    # Get all the image files
    image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if is_image_file(f)]

    # Loop through each file in the image directory
    for file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(IMAGE_DIRECTORY, file)
        with Image.open(image_path) as image:
            # Convert image file name to integer for matching with DataFrame index
            try:
                image_id = int(
                    os.path.splitext(file)[0]
                )  # Convert the image ID to integer
            except ValueError:
                print(
                    f"Skipping file {file} as its name cannot be converted to integer."
                )
                continue

            # Construct the prompt with the corresponding caption
            if image_id in df.index:
                caption = df.loc[image_id, "caption"]

                # PROMPT_1 = f"Analyze the person in the image, considering the Instagram caption written by the person who posted it: '{caption}'. Provide a JSON response with the following fields: 'pose_and_posture': Description of their stance, 'body_prominence': How their body is displayed, 'skin_appearance': Details of skin texture and features, 'body_features': Information on weight, slimness, muscularity, and facial characteristics, 'disability_or_syndrome': Indicators of any disabilities or syndromes, 'aligns_with_beauty_standards': boolean value (0 or 1) for whether the individual's appearance aligns with traditional beauty standards on social media, 'explanation': brief explanation for aligns_with_beauty_standards."

                PROMPT_2 = (
                    f"Analyze the person in the image, considering the Instagram caption written by the person who posted it: '{caption}'. Beyond the immediate visual elements, reflect on the cultural, social, or personal significance conveyed in the image and caption. Does the image challenge or conform to traditional beauty norms? How does the caption complement or contrast with the visual message? Provide a JSON response with the following fields: "
                    "'pose_and_posture': Describe the physical stance and any implied emotions or attitudes, "
                    "'body_prominence': Detail how the body is displayed, including context such as clothing and setting, "
                    "'skin_appearance': Note skin texture, features, makeup, tattoos, or other adornments, "
                    "'body_features': Assess weight, slimness, muscularity, and facial characteristics, focusing on conformity or divergence from beauty standards, "
                    "'disability_or_syndrome': dentify any disabilities or syndromes, focusing on representation, "
                    "'aligns_with_beauty_standards': Boolean (0 or 1) indicating if the appearance aligns with traditional social media beauty norms, "
                    "'explanation': a brief explanation for the 'aligns_with_beauty_standards' decision, linking observed elements to beauty norms."
                )

                # prompt = f"Analyze the following Instagram image with a focus on pose and posture, body prominence, body image, hair style and skin appearance. The image was posted on Instagram with the caption: \"{caption}\". Provide a JSON response with the following fields: description (TEXT) and align_with_beauty_standards (BOOLEAN: TRUE or FALSE)"
                description = ask_image(
                    tokenizer, model, image_processor, image, prompt=PROMPT_2
                )

                # Add the result to the dictionary
                results[image_id] = {
                    "id": image_id,
                    "caption": caption,
                    "model": description,
                }

                # Save the results after each iteration to ensure progress is saved
                save_results_to_json(results, RESULT_JSON_PATH)

    print(f"Processing complete. Results saved to {RESULT_JSON_PATH}")


if __name__ == "__main__":
    main()
