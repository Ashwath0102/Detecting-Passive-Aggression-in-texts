# Detecting-Passive-Aggression-in-texts

## NLP Program to Detect Passive-Aggressive Statements

### Introduction

The NLP Program to Detect Passive-Aggressive Statements is designed to identify passive-aggressive tones in text data. Passive-aggressive communication can be challenging to recognize, as it involves indirect expressions of hostility or resentment. This program aims to address this issue by employing various Natural Language Processing (NLP) techniques and machine learning models to accurately detect passive-aggressive statements.

### Problem Statement

Passive-aggressive behaviour can lead to misunderstandings and conflicts, especially in online interactions where tone and body language are absent. Detecting passive-aggressive statements is crucial for improving communication and fostering healthier online environments. This program addresses the challenge of identifying passive-aggressive tones in textual data.

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

6. **Sentiment Analysis**: The VADER sentiment analysis tool is used to understand the emotional tone of statements. However, it's observed that sentiment analysis alone cannot identify passive-aggressive or sarcastic tones effectively.

### Program Explanation

1. **Dataset Creation**: The program generates a synthetic dataset containing both passive-aggressive and non-passive-aggressive statements.

2. **Exploratory Data Analysis (EDA)**: The dataset's shape, column information, and class distribution are analyzed. Additionally, the length distributions of statements in terms of words and characters are visualized.

3. **Sentiment Analysis**: VADER sentiment analysis is performed to understand the emotional tone of statements. However, this analysis proves insufficient for detecting passive-aggressiveness accurately.

4. **Text Preprocessing and Feature Extraction**: The statements are preprocessed, and TF-IDF vectors and Word2Vec embeddings are generated for training the machine learning models.

5. **Machine Learning Models**: Logistic Regression, Random Forest, and SVM models are trained and evaluated using TF-IDF vectors and Word2Vec embeddings.

6. **Deep Learning Model (GPT-3)**: The program utilizes GPT-3, a deep learning model fine-tuned for passive-aggressiveness detection, and compares its accuracy with traditional machine learning models.

7. **Accuracy Comparison**: The accuracy of all models (Logistic Regression, Random Forest, SVM, and GPT-3) is compared and visualized for better understanding.


### Why These Models

The presented NLP program stands out due to its comprehensive approach. The program achieves a nuanced understanding of passive-aggressive tones by combining traditional machine learning models (Logistic Regression, Random Forest, SVM) with advanced techniques like Word2Vec embeddings and the state-of-the-art GPT-3 model.

While sentiment analysis falls short in capturing subtle nuances, the integration of multiple models allows for a more accurate detection of passive-aggressiveness. Additionally, the program's ability to visualize and compare various models provides valuable insights for choosing the most suitable approach in different contexts.

### Results


![download (2)](https://github.com/Ashwath0102/Detecting-Passive-Aggression-in-texts/assets/59199696/39780500-cc9a-40a6-b103-25630a43901f)

**Logistic Regression (TF-IDF):** Achieved an accuracy of 84.54%, making it the most accurate model in identifying passive-aggressive language using the TF-IDF approach.

**Random Forest (TF-IDF):** Demonstrated an accuracy of 81.44%, indicating a reliable performance in recognizing passive-aggressive tones.

**SVM (TF-IDF):** Attained an accuracy rate of 83.51%, showcasing its effectiveness in discerning passive-aggressive language in the dataset.

**GPT-3:** In the final iteration, the GPT-3 model achieved an accuracy of 50.52%, which suggests a random classification performance.


#### **Implications of the Results:**

1. **Reliable Communication**: The model’s 84.54% accuracy ensures consistent detection of passive-aggressive language, vital for clear communication in online forums and social media platforms.

2. **Conflict Prevention**: Early identification of passive-aggressive tones aids in preventing misunderstandings and conflicts, fostering healthier online interactions.

3. **Moderation Support**: Human moderators benefit from the model, making informed content moderation decisions by flagging potential passive-aggressive statements.

4. **User Experience Enhancement**: Platforms prioritizing positive experiences benefit by filtering out passive-aggressive content, creating a more respectful environment.

5. **User Feedback**: The model offers educational feedback, encouraging users to be mindful of communication styles, and promoting better online etiquette.

#### **Limitations and Future Improvements:**

1. **Nuance Challenges**: Subtle language nuances like sarcasm remain challenging. Continuous training with diverse datasets mitigates these limitations.

2. **Multimodal Integration**: Integrating images or user behaviours enhances accuracy, providing a broader context for communication nuances.

3. **Real-Time Scalability**: Optimizing real-time application and scalability ensures efficient deployment in high-traffic online platforms.

4. **Ethical Considerations**: Addressing biases and ensuring fairness through ongoing evaluation is crucial for diverse user groups.

*In summary, the Logistic Regression TF-IDF model's 84.54% accuracy demonstrates its effectiveness in detecting passive-aggressive language. Acknowledging its limitations, continuous refinement promises a more nuanced and applicable tool, fostering respectful online communication.*
