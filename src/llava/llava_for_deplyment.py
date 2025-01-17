# This Script is structured for llava deployment.
import os
import json
"""
This script is a prpared for deployment of the LLAVA model.
"""

import torch
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, Conversation
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


class Llava:
    def __init__(self, model_path: str):
        self.tokenizer, self.model, self.image_processor = self.load_model(model_path)
        self.model_path = model_path

    @staticmethod
    def load_model(model_path):
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_4bit=True,
        )
        return tokenizer, model, image_processor

    def process_image(self, image: Image):
        args = {"image_aspect_ratio": "pad"}
        image_tensor = process_images([image], self.image_processor, args)
        return image_tensor.to(self.model.device, dtype=torch.float16)

    def create_prompt(self, prompt, custom_system_prompt):
        custom_conversation = Conversation(
            system=custom_system_prompt,
            roles=("Human", "AI Social Scientist"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        conv = custom_conversation.copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), conv

    def ask_image(self, image: Image, prompt: str, custom_system_prompt: str):
        image_tensor = self.process_image(image)
        prompt, conv = self.create_prompt(prompt, custom_system_prompt)
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            keywords=[stop_str], tokenizer=self.tokenizer, input_ids=input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()


def is_image_file(filename: str):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    return os.path.splitext(filename)[1].lower() in valid_extensions


def classify_image(
    image_path: str, prompt: str, custom_system_prompt: str, model: Llava
):
    if not is_image_file(image_path):
        return "File is not a valid image."
    with Image.open(image_path) as image:
        response = model.ask_image(image, prompt, custom_system_prompt)
        return response


if __name__ == "__main__":
    MODEL_PATH = "4bit/llava-v1.5-13b-3GB"
    model = Llava(MODEL_PATH)
    IMAGE_PATH = "path/to/image.jpg"
    PROMPT = (
        "Analyze the person in the image. Provide a JSON response with the following fields: "
        "'pose_and_posture': Description of their stance, "
        "'body_prominence': How their body is displayed, "
        "'skin_appearance': Details of skin texture and features, "
        "'body_features': Information on weight, slimness, muscularity, and facial characteristics, "
        "'disability_or_syndrome': Indicators of any disabilities or syndromes, "
        "'aligns_with_beauty_standards': boolean value (0 or 1) for whether the individual's appearance aligns with traditional beauty standards on social media, "
        "'explanation': brief explanation for aligns_with_beauty_standards."
    )
    CUSTOM_SYSTEM_PROMPT = "In the role of a social scientist, you are tasked with classifying social media images into 'norm_beauty' and 'divers' categories. 'norm_beauty' images typically feature conventional beauty standards and poses, while 'divers' images represent a wider spectrum of human diversity, including unique physical traits and unconventional styles. Analyze these images critically, focusing on their representation of societal norms and diversity."
    result = classify_image(IMAGE_PATH, PROMPT, CUSTOM_SYSTEM_PROMPT, model)
    print(result)
