import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import os
import seaborn as sns

def visulize(model_name):
    # original = 'Sentiment'
    # prediction = 'Predicted_Sentiment_'
    # file_name = '../Sentiment Analysis/critic_reviews_test_'
    # original = 'Rating'
    # prediction = 'Predicted_Rating_'
    # file_name = '../Rating-Prediction/audience_reviews_test_'
    # original = 'Sentiment'
    # prediction = 'Predicted_Sentiment_fs_'
    # file_name = '../Sentiment Analysis/critic_reviews_test_fs_'
    original = 'Rating'
    prediction = 'Predicted_Rating_fs_'
    file_name = '../Rating-Prediction/audience_reviews_test_fs_'
    if model_name == 'gpt':
        file_name = file_name + model_name + '.csv'
        prediction = prediction + model_name
        df = pd.read_csv(file_name)
        y_true = df[original]
        y_pred = df[prediction]
    elif model_name == 'gemini':
        file_name = file_name + model_name + '.csv'
        prediction = prediction + model_name
        df = pd.read_csv(file_name)
        y_true = df[original]
        y_pred = df[prediction]
    # elif model_name == 'llama':
    #     file_name = file_name + model_name + '.csv'
    #     prediction = prediction + model_name
    #     df = pd.read_csv(file_name)
    #     y_true = df[original]
    #     y_pred = df[prediction]

    mae = mean_absolute_error(y_true, y_pred)
    print(mae)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False,
                xticklabels=['1', '2', '3', '4', '5'],
                yticklabels=['1', '2', '3', '4', '5'])
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.title('Confusion Matrix for Gemini Pro: Few Shot')
    plt.show()

    # Scatter plot of actual vs. predicted sentiments
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    title = f"MAE: {mae}"
    plt.title(title)
    plt.show()

    # Calculate and print classification report
    print(classification_report(y_true, y_pred))

#visulize('gpt')
visulize('gemini')
