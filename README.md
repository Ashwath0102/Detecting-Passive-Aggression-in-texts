# Detecting-Passive-Aggression-in-texts

## NLP Program to Detect Passive-Aggressive Statements

### Introduction

The NLP Program to Detect Passive-Aggressive Statements is designed to identify passive-aggressive tones in text data. Passive-aggressive communication can be challenging to recognize, as it involves indirect expressions of hostility or resentment. This program aims to address this issue by employing various Natural Language Processing (NLP) techniques and machine learning models to accurately detect passive-aggressive statements.

### Problem Statement

Passive-aggressive behavior in communication can lead to misunderstandings and conflicts, especially in online interactions where tone and body language are absent. Detecting passive-aggressive statements is crucial for improving communication and fostering healthier online environments. This program addresses the challenge of identifying passive-aggressive tones in textual data.

### Architecture Description

The program employs several key components and techniques:

1. **Dataset Creation**: A synthetic dataset is created containing passive-aggressive and non-passive-aggressive statements. This dataset is used for training and testing the models.

2. **Text Preprocessing**: The statements undergo text preprocessing, including lowercase conversion, tokenization, punctuation removal, and stopword elimination.

3. **Feature Extraction**: Two feature extraction techniques are utilized:
   - **TF-IDF Vectorization**: Transforms text into numerical vectors, capturing the importance of words in the documents.
   - **Word2Vec Embeddings**: Word embeddings are created using Word2Vec, capturing semantic meanings of words.

4. **Machine Learning Models**:
   - **Logistic Regression**: A classic linear model used for binary classification tasks.
   - **Random Forest**: An ensemble learning method combining multiple decision trees.
   - **SVM (Support Vector Machine)**: A powerful classification algorithm finding the optimal hyperplane for separating classes.

5. **Deep Learning Model**: 
   - **GPT-3 (Generative Pre-trained Transformer 3)**: A state-of-the-art language processing model fine-tuned for passive-aggressiveness detection.

6. **Sentiment Analysis**: VADER sentiment analysis tool is used to understand the emotional tone of statements. However, it's observed that sentiment analysis alone cannot identify passive-aggressive or sarcastic tones effectively.

### Program Explanation

1. **Dataset Creation**: The program begins by generating a synthetic dataset containing both passive-aggressive and non-passive-aggressive statements.

2. **Exploratory Data Analysis (EDA)**: The dataset's shape, column information, and class distribution are analyzed. Additionally, the length distributions of statements in terms of words and characters are visualized.

3. **Sentiment Analysis**: VADER sentiment analysis is performed to understand the emotional tone of statements. However, this analysis proves insufficient for detecting passive-aggressiveness accurately.

4. **Text Preprocessing and Feature Extraction**: The statements are preprocessed, and TF-IDF vectors and Word2Vec embeddings are generated for training the machine learning models.

5. **Machine Learning Models**: Logistic Regression, Random Forest, and SVM models are trained and evaluated using TF-IDF vectors and Word2Vec embeddings.

6. **Deep Learning Model (GPT-3)**: The program utilizes GPT-3, a deep learning model fine-tuned for passive-aggressiveness detection, and compares its accuracy with traditional machine learning models.

7. **Accuracy Comparison**: The accuracy of all models (Logistic Regression, Random Forest, SVM, and GPT-3) is compared and visualized for better understanding.


### Why These Models

The presented NLP program stands out due to its comprehensive approach. By combining traditional machine learning models (Logistic Regression, Random Forest, SVM) with advanced techniques like Word2Vec embeddings and the state-of-the-art GPT-3 model, the program achieves a nuanced understanding of passive-aggressive tones.

While sentiment analysis falls short in capturing subtle nuances, the integration of multiple models allows for a more accurate detection of passive-aggressiveness. Additionally, the program's ability to visualize and compare various models provides valuable insights for choosing the most suitable approach in different contexts.


In conclusion, this NLP program's strength lies in its holistic approach, leveraging both traditional and modern methods, making it a robust and reliable solution for detecting passive-aggressive tones in textual communication.
