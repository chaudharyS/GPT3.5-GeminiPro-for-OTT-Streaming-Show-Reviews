import pandas as pd
import pathlib
import textwrap
import os
import time
from tqdm import tqdm
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

def analyze_gemini(text):
    try:
        time.sleep(3) # Wait for 3 seconds before making the next call
        model = genai.GenerativeModel('gemini-1.0-pro')
        system_message = """You are trained to analyze and detect the sentiment of given text.
                            You will be provided with a movie review text, and your task is to classify its sentiment.
                            You use 1 to denote positive and 0 to denote negative."""

        # Getting examples for few-shot prompting
        input_file = '../Sentiment Analysis/critic_reviews_train.csv'
        df = pd.read_csv(input_file)
        random_rows = df[['Sentiment', 'Review']].sample(n=8)

        examples = ""
        # Iterate over the random rows to create the examples string
        for index, row in random_rows.iterrows():
            review_text = row['Review']
            sentiment = row['Sentiment']
            examples += f"User: {review_text}\nAssistant: {sentiment}\n"

        user_message = f"""
                Analyze the following movie review and determine if the sentiment is: positive (1) or negative (0). 
                Return answer only as 1 or 0. Return 1 for positive and 0 for negative: {text}"""

        prompt = f"{system_message}\n\n{examples}\n\n{user_message}"

        response = model.generate_content(
            prompt,
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=10,
                temperature=0
            )
        )
        sentiment = response.text
        return sentiment
    except Exception as e:
        print(f"Error fetching sentiment for '{text}': {e}")
        return None

test_file = '../Sentiment Analysis/critic_reviews_test.csv'
# test_file = '../Sentiment Analysis/test.csv'
df = pd.read_csv(test_file)

tqdm.pandas(desc="Processing reviews through Gemini Pro", unit=" reviews")

df['Predicted_Sentiment_gemini'] = df['Review'].progress_apply(analyze_gemini)

output_file = '../Sentiment Analysis/critic_reviews_test_fs_gemini.csv'
# output_file = '../Sentiment Analysis/test_gemini.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

# Print the output path of the CSV file
print("CSV file saved at:", os.path.abspath(output_file))
