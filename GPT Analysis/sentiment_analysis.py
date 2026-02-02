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
    messages = [

        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given text.
                                        You will be provided with a movie review text, and your task is to classify its sentiment.
                                        You use 1 to denote positive and 0 to denote negative."""},
        {"role": "user", "content": f"""Analyze the following movie review and determine if the sentiment is: positive (1) or negative (0). 
                                        Return answer only as 1 or 0. Return 1 for positive and 0 for negative: {text}"""}
                                        
    ]
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        max_tokens = 10,
        temperature = 0
    )
    sentiment = response.choices[0].message.content.strip().lower()
    return sentiment

test_file = '../Sentiment Analysis/critic_reviews_test.csv'
# test_file = '../Sentiment Analysis/test.csv'
df = pd.read_csv(test_file)

tqdm.pandas(desc="Processing reviews through GPT3.5", unit=" reviews")

df['Predicted_Sentiment_gpt'] = df['Review'].progress_apply(analyze_gpt35)

output_file = '../Sentiment Analysis/critic_reviews_test_gpt35.csv'
# output_file = '../Sentiment Analysis/test_gpt35.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

# Print the output path of the CSV file
print("CSV file saved at:", os.path.abspath(output_file))
