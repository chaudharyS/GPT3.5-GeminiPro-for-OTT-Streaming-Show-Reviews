import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

def evaluate_model(model_name):
    # original = 'Sentiment'
    # prediction = 'Predicted_Sentiment_'
    # file_name = '../Sentiment Analysis/critic_reviews_test_'
    original = 'Rating'
    prediction = 'Predicted_Rating_'
    file_name = '../Rating-Prediction/audience_reviews_test_'
    # original = 'Sentiment'
    # prediction = 'Predicted_Sentiment_fs_'
    # file_name = '../Sentiment Analysis/critic_reviews_test_fs_'
    # original = 'Rating'
    # prediction = 'Predicted_Rating_fs_'
    # file_name = '../Rating-Prediction/audience_reviews_test_fs_'
    if model_name == 'gpt':
        file_name = file_name + model_name + '.csv'
        prediction = prediction + model_name
        df = pd.read_csv(file_name)
        accuracy = round(accuracy_score(df[original], df[prediction]), 4)
        precision = round(precision_score(df[original], df[prediction], average='weighted'), 4)
        recall = round(recall_score(df[original], df[prediction], average='weighted'), 4)
        f1 = round(f1_score(df[original], df[prediction], average='weighted'), 4)
    elif model_name == 'gemini':
        file_name = file_name + model_name + '.csv'
        prediction = prediction + model_name
        df = pd.read_csv(file_name)
        accuracy = round(accuracy_score(df[original], df[prediction]), 4)
        precision = round(precision_score(df[original], df[prediction], average='weighted'), 4)
        recall = round(recall_score(df[original], df[prediction], average='weighted'), 4)
        f1 = round(f1_score(df[original], df[prediction], average='weighted'), 4)
    # elif model_name == 'llama':
    #     file_name = file_name + model_name + '.csv'
    #     prediction = prediction + model_name
    #     df = pd.read_csv(file_name)
    #     accuracy = round(accuracy_score(df[original], df[prediction]), 4)
    #     precision = round(precision_score(df[original], df[prediction], average='weighted'), 4)
    #     recall = round(recall_score(df[original], df[prediction], average='weighted'), 4)
    #     f1 = round(f1_score(df[original], df[prediction], average='weighted'), 4)

    # Create a DataFrame with the evaluation results including the 'model' column
    evaluation_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1]
    })

    # Append the results to the existing CSV file or create a new one
    # model_evaluation_results = '../Sentiment Analysis/model_evaluation_results.csv'
    # model_evaluation_results = '../Rating-Prediction/model_evaluation_results.csv'
    # model_evaluation_results = '../Rating-Prediction/model_evaluation_results.csv'
    # model_evaluation_results = '../Rating-Prediction/model_evaluation_fs_results.csv'
    # evaluation_df.to_csv(model_evaluation_results, mode='a', header=not os.path.exists(model_evaluation_results), index=False)

    return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

print(f'base:gpt-3.5: ' + str(evaluate_model('gpt')))
print(f'base:gemini-1.0-pro: ' + str(evaluate_model('gemini')))
# print(f'base:llama-3-8b-instruct: ' + str(evaluate_model('llama')))
