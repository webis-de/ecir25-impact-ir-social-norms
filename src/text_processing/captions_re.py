"""
This script uses the GPT-4 model to reformulate the captions in the dataset.
"""
import pandas as pd
import os
import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your dataset
df = pd.read_csv("../data/1_captions.csv", sep=";")

# Define the system message
system_msg = "You are a helpful assistant who understands linguistics and language."


# Function to process each caption
def process_caption(caption):
    user_msg = (
        "Reformulate the following Instagram caption in one sentence to capture the essence of it. Focus on the meaning of the caption, the sentiment, and the context."
        "If the caption has emojis, remove them and explain them in natural language. "
        "If the caption has hashtags, remove them and explain them in natural language. "
        "If the caption is not in English, translate it to English and reformulate."
        'Caption: "{}"'.format(caption)
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error processing caption: {caption}. Error: {e}")
        return caption


gpt_4_responses = []
# Process and print results for the first 10 rows with tqdm progress bar
for i in tqdm(range(len(df)), desc="Processing captions"):
    original_caption = df["caption"][i]
    reformulated_caption = process_caption(original_caption)
    gpt_4_responses.append(reformulated_caption)

# add reformulated caption column to the dataframe
df["reformulated_caption"] = gpt_4_responses

# save the dataframe to csv file
df.to_csv("../data/1_reformulated_captions.csv", index=False)
