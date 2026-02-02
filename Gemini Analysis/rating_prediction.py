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
        prompt = f"""
        You are trained to analyze and predict the rating from the review text. You will be provided 
        with a movie review text, and your task is to predict the rating. The rating can be anywhere 
        between 1 and 5, in increments of 1. Note that 1 denotes a very bad review and 5
        denotes an excellent review. Use the text sentiment and content to determine the appropriate rating.
        Analyze the following movie review and predict its rating. Your response should be an integer
        value between 1 and 5. Return only the rating and no additional explanation: {text}"""
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

test_file = '../Rating-Prediction/audience_reviews_test.csv'
# test_file = '../Rating-Prediction/test.csv'
df = pd.read_csv(test_file)

tqdm.pandas(desc="Processing reviews through Gemini Pro", unit=" reviews")

df['Predicted_Sentiment_gemini'] = df['Review'].progress_apply(analyze_gemini)

output_file = '../Rating-Prediction/audience_reviews_testint_gemini.csv'
# output_file = '../Rating-Prediction/test_gemini.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

# Print the output path of the CSV file
print("CSV file saved at:", os.path.abspath(output_file))
