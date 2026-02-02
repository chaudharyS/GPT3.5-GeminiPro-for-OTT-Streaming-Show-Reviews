import pandas as pd
import openai
import numpy as np
from openai import OpenAI
import os
import time
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")  # Access environment variable
api_key = os.getenv('OPENAI_API_KEY')
# print(f"API Key: {api_key}")

def analyze_gpt35(text):
    time.sleep(3) # Wait for 3 seconds before making the next call
    client = OpenAI()
    system_message = {
        "role": "system",
        "content": """You are trained to analyze and predict the rating from the review text.
                    You will be provided with a movie review text, and your task is to predict the rating.
                    The rating can be anywhere between 1 and 5, in increments of 1. Note that 1 denotes
                    a very bad review and 5 denotes an excellent review. Use the text sentiment and content to
                    determine the appropriate rating."""
    }
    user_message = {
        "role": "user",
        "content": f"""Analyze the following movie review and predict its rating. Your response should be an integer
                    value between 1 and 5. Return only the rating and no additional explanation: {text}"""
    }

    # Getting examples for few-shot prompting
    input_file = '../Rating-Prediction/audience_reviews_train.csv'
    df = pd.read_csv(input_file)
    random_rows = df[['Rating', 'Review']].sample(n=8)

    # Initialize the examples list
    examples = []
    # Iterate over the random rows to create the examples array
    for index, row in random_rows.iterrows():
        review_text = row['Review']
        rating = row['Rating']
        examples.append({"role": "user", "content": review_text})
        examples.append({"role": "assistant", "content": str(rating)})

    messages = [system_message] + examples + [user_message]

    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        max_tokens = 10,
        temperature = 0
    )
    sentiment = response.choices[0].message.content.strip().lower()
    return sentiment

test_file = '../Rating-Prediction/audience_reviews_test.csv'
# test_file = '../Rating-Prediction/test.csv'
df = pd.read_csv(test_file)

tqdm.pandas(desc="Processing reviews through GPT3.5", unit=" reviews")

df['Predicted_Rating_fs_gpt'] = df['Review'].progress_apply(analyze_gpt35)

output_file = '../Rating-Prediction/audience_reviews_test_fs_gpt.csv'
# output_file = '../Sentiment Analysis/test_gpt35.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

# Print the output path of the CSV file
print("CSV file saved at:", os.path.abspath(output_file))
