# also decraeses tokens which is good
import pandas as pd
import string
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import tiktoken
import os

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the CSV file
input_file = '../Rating-Prediction/audience_reviews_raw.csv'
df_raw = pd.read_csv(input_file)

# # Drop rows with missing review text
df_raw.dropna(subset=['Review'], inplace=True)

# # Remove rows with duplicate review text
df_raw.drop_duplicates(subset=['Review'], inplace=True)

# # Remove rows that don't contain the full review
df_raw.drop(df_raw[df_raw['Review'].str.contains('Full Review', case=False, na=False)].index, inplace=True)

# # # Picking out 4000 random rows from the datset of 14790 rows
# # # Split the DataFrame based on 'Sentiment' values
df_sentiment_0 = df_raw[df_raw['Sentiment'] == 0]
df_sentiment_1 = df_raw[df_raw['Sentiment'] == 1]

# # # Sample an equal number of rows from each DataFrame
df_sampled_0 = df_sentiment_0.sample(n=2000, random_state=11)
df_sampled_1 = df_sentiment_1.sample(n=2000, random_state=11)

# # # Concatenate the sampled DataFrames together
df = pd.concat([df_sampled_0, df_sampled_1]).sample(frac=1, random_state=21)

df = df_raw.sample(n=4000, random_state=11)

# def pre_process(review):
#     review = review.strip()
#     review = review.lower()
#     # Remove \n and \t
#     review = review.replace('\\n', ' ').replace('\\t', ' ')
#     # Remove extra white spaces
#     review = ' '.join(review.split())
#     review = contractions.fix(review)
#     # Remove special characters and delimiters 
#     review = re.sub(r'[^a-zA-Z0-9\s]', '', review)

#     return review

#     # # Tokenization
#     # tokens = word_tokenize(review)
    
#     # # Removing punctuation and special characters, removing numbers
#     # # normalized_tokens = [word for word in tokens if word.isalpha()]
    
#     # # Removing stop words
#     # stop_words = set(stopwords.words('english'))
#     # normalized_tokens = [word for word in tokens if word not in stop_words]

#     # # Lemmatize tokens
#     # lm = WordNetLemmatizer()
#     # normalized_tokens = [lm.lemmatize(word) for word in normalized_tokens]
    
#     # # Handle negations
#     # processed_tokens = []
#     # negation_flag = False
#     # for word in normalized_tokens:
#     #     if word in ['not', 'no', 'never']:
#     #         negation_flag = True
#     #     elif negation_flag:
#     #         processed_tokens.append('not_' + word)
#     #         negation_flag = False
#     #     else:
#     #         processed_tokens.append(word)

#     # # Joining tokens back into normalized text
#     # processed_text = ' '.join(processed_tokens)
#     # return processed_text

# # Apply normalization to the 'Review' column
df['Review'] = df['Review'].apply(pre_process)

# # # Use the TikToken Library to count tokens
# def num_tokens_from_string(string: str) -> int:
#     """Returns the number of tokens in a text string."""
#     # https://github.com/openai/tiktoken/tree/main
#     encoding_name = "cl100k_base"
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# df['Openai_Tokens'] = df['Review'].apply(num_tokens_from_string)

# Define output file path
output_file = '../Rating-Prediction/audience_reviews.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

# print("CSV file saved at:", os.path.abspath(output_file))
