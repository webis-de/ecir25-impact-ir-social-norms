"""
This script is used to remove the background of an image using the DeepLabV3 model
"""
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class BackgroundRemover:
    def __init__(self):
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "deeplabv3_resnet101", pretrained=True
        )
        self.model.eval()

    @staticmethod
    def make_transparent_foreground(pic, mask):
        b, g, r = cv2.split(np.array(pic).astype("uint8"))
        a = np.ones(mask.shape, dtype="uint8") * 255
        alpha_im = cv2.merge([b, g, r, a], 4)
        bg = np.zeros(alpha_im.shape)
        alpha_mask = np.stack(
            [mask] * 3 + [mask], axis=2
        )  # Adding the mask as alpha channel
        composite_image = np.where(
            alpha_mask,
            alpha_im if np.max(mask) == 255 else bg,
            bg if np.max(mask) == 255 else alpha_im,
        ).astype(np.uint8)
        return composite_image

    def remove_background(self, input_image):
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            self.model.to("cuda")

        with torch.no_grad():
            output = self.model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        mask = output_predictions.byte().cpu().numpy()
        bin_mask = np.where(mask, 255, np.zeros(mask.shape)).astype(np.uint8)

        # Create the foreground image
        foreground = self.make_transparent_foreground(input_image, bin_mask)

        # Create the background image
        inverse_bin_mask = np.where(mask, np.zeros(mask.shape), 255).astype(np.uint8)
        background_image = self.make_transparent_foreground(
            input_image, inverse_bin_mask
        )

        return foreground, background_image

    def separate_background_single(self, image):
        input_image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        foreground, background_image = self.remove_background(input_image_pil)
        foreground_cv = cv2.cvtColor(np.array(foreground), cv2.COLOR_RGBA2BGRA)
        background_image_cv = cv2.cvtColor(
            np.array(background_image), cv2.COLOR_RGBA2BGRA
        )
        return foreground_cv, background_image_cv
