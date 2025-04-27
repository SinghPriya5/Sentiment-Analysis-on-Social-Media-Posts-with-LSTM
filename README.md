# Amazon Reviews Sentiment Analysis 📊💬

This project involves training and evaluating machine learning models to classify Amazon reviews into positive and negative sentiment categories using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. This solution provides valuable insights into customer sentiments, helping businesses improve their products, services, and customer relations by analyzing feedback.

## Table of Contents 📑

- [Problem Statement](#problem-statement)
- [Goal](#goal)
- [Approach](#approach)
- [Data Collection](#data-collection)
- [Project Various Steps](#project-various-steps)
  - [Data Preprocessing](#data-preprocessing)
  - [Data Visualization](#data-visualization)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Selection](#model-selection)
  - [Model Dump](#model-dump)
- [Idle Used](#idle-used)
- [Model Accuracy](#model-accuracy)
- [Continuous Improvement](#continuous-improvement)
- [Deployed](#deployed)
- [Model Interpretation](#model-interpretation)
- [Web View](#web-view)
- [Bug or Feature Request](#bug-or-feature-request)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)

## Problem Statement 🧐

The problem this project addresses is the classification of Amazon product reviews into **positive** or **negative** sentiment categories. In today’s digital era, reviews are critical for businesses to understand customer satisfaction. However, processing large volumes of text manually is time-consuming and inefficient. This project uses machine learning models to automate the classification of these reviews, providing businesses with a scalable and efficient tool to understand customer sentiment.

The challenge lies in:
- Understanding the textual content, which could include slang, typos, and context-specific language 🗣️.
- Classifying reviews with high accuracy, even when the text contains ambiguous or complex sentiments 🤔.

## Goal 🎯

The primary goal of this project is to:
- Develop a model capable of automatically classifying Amazon product reviews into sentiment categories (positive/negative).
- Leverage **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** models for sentiment analysis tasks 💻.
- Achieve a high level of accuracy while ensuring the model generalizes well across different types of reviews 📈.
- Enable easy deployment of the model for real-time sentiment analysis of new reviews 🌍.

## Approach 🔄

This project follows a structured approach to achieve its goals:
1. **Data Collection**: Use the publicly available Amazon review dataset to train and test the model 📂.
2. **Data Preprocessing**: Perform text cleaning, tokenization, and padding to prepare the data for model training 🧹.
3. **Model Training**: Train both RNN and LSTM models to identify patterns in review text and predict sentiment 🤖.
4. **Model Evaluation**: Evaluate the models based on performance metrics such as accuracy, precision, recall, and F1-score 📊.
5. **Model Selection**: Based on evaluation results, select the best model for deployment and real-time analysis 🏆.

## Data Collection 📦

The dataset used for this project comes from Kaggle and contains product reviews from Amazon. Each review is labeled as **positive** or **negative** sentiment. The dataset is in a compressed `.bz2` format and consists of:
- **Training Data**: `train.ft.txt.bz2`
- **Testing Data**: `test.ft.txt.bz2`

Each row in the dataset consists of a review text and its corresponding sentiment label. The reviews range from short comments to longer descriptions of the product experience, making this a real-world text classification challenge.

## Project Various Steps 🛠️

### Data Preprocessing 🧹

The data preprocessing phase includes the following steps:
- **Loading Data**: Extracting and loading the dataset from `.bz2` compressed files 📥.
- **Text Cleaning**: Removing special characters, extra spaces, and punctuation. We also lowercase all text and remove stopwords to ensure meaningful text input 🔤.
- **Stemming**: Using **PorterStemmer** to reduce words to their root form (e.g., "running" becomes "run") 🏃‍♂️.
- **Tokenization**: Splitting text into individual tokens (words) using the `NLTK` tokenizer 💬.
- **Padding**: To ensure consistent input size for the models, sequences are padded to the same length using `pad_sequences` 🔢.

### Data Visualization 📊

Visualization is a key part of understanding the data:
- **Sentiment Distribution**: A bar chart is used to visualize the balance of positive and negative reviews, helping ensure the dataset is not biased ⚖️.
- **Word Frequency**: A word cloud or frequency plot is used to show the most commonly used words in the reviews. This provides insights into the most discussed features of products 🌟.

### Model Training 🧠

We use two deep learning architectures for training the sentiment analysis model:
- **RNN**: A simple Recurrent Neural Network, suitable for handling sequences of data like text 🔁.
- **Bidirectional LSTM**: A more advanced model that processes sequences in both forward and reverse directions, making it more powerful in capturing long-term dependencies in text 🔀.

Both models are compiled with the **Adam optimizer** and a **binary cross-entropy loss function** since this is a binary classification problem ⚙️.

### Model Evaluation 📊

Once the models are trained, we evaluate them using:
- **Accuracy**: Measures the percentage of correct predictions ✅.
- **Precision**: Indicates the proportion of positive predictions that were actually correct 🔍.
- **Recall**: Shows the proportion of actual positive samples that were correctly predicted 💯.
- **F1-Score**: A balanced measure that combines precision and recall ⚖️.

### Model Selection 🎯

Based on the evaluation metrics, we select the **Bidirectional LSTM** model as it performed slightly better than the RNN in terms of recall and F1-score 🏆. We believe it can generalize better to unseen reviews, particularly when dealing with longer or more complex review texts.

### Model Dump 💾

After selecting the best model, we serialize it using **joblib** and save the trained model for future use. This ensures that we can load the model without retraining it each time and deploy it in production environments easily 🚀.

## Idle Used 🛠️

This project uses the following libraries and tools:
- **Python 3.11+** 🐍
- **TensorFlow 2.18.0**: For building and training deep learning models 📦.
- **Keras**: To simplify model creation and experimentation ⚙️.
- **NLTK**: Used for text preprocessing tasks like tokenization and stemming 📝.
- **Matplotlib**: For data visualizations such as bar charts and word clouds 🎨.
- **Pandas**: For data manipulation and analysis 🔢.
- **NumPy**: For numerical computations required during data preprocessing ➗.
- **joblib**: For saving and loading the trained model 🗂️.

## Model Accuracy 📈

The model was evaluated on the test dataset with the following results:
- **RNN Model**:
  - Accuracy: 82% ✔️
  - Precision: 0.81
  - Recall: 0.83
  - F1-Score: 0.82
- **Bidirectional LSTM Model**:
  - Accuracy: 82% ✔️
  - Precision: 0.82
  - Recall: 0.84
  - F1-Score: 0.83

The **Bidirectional LSTM** outperformed the **RNN** by a small margin, which is why it was selected as the final model 🥇.

## Continuous Improvement 🔄

Although the models performed well, there is always room for improvement:
- **Hyperparameter Tuning**: We can perform **GridSearchCV** or **RandomizedSearchCV** to fine-tune hyperparameters such as the number of LSTM units, batch size, and learning rate ⚙️.
- **Pretrained Word Embeddings**: Using pretrained embeddings like **GloVe** or **Word2Vec** could enhance the model's performance by providing richer word representations 🌐.
- **Ensemble Models**: Combine the predictions of multiple models (e.g., LSTM, RNN, SVM) to improve overall performance 🤝.

## Deployed 🌐

The model is not yet deployed, but the next steps would involve integrating it into a **Flask** or **Streamlit** app for real-time sentiment analysis. This would allow users to input Amazon reviews and receive an immediate sentiment classification 🚀.

## Model Interpretation 🔍

To interpret the model’s predictions:
- **Confusion Matrix**: Provides a breakdown of true positives, false positives, true negatives, and false negatives 📉.
- **Classification Report**: Summarizes the precision, recall, and F1-score of the model for each class 📄.

For advanced interpretability, we could consider techniques such as **LIME** (Local Interpretable Model-agnostic Explanations) to explain individual predictions 🧠.

## Web View 🌍

We aim to build a **Web Application** using **Streamlit** or **Flask** where users can enter a product review and get an immediate sentiment prediction (positive or negative). This will make the model accessible for real-time customer feedback analysis 💬.

## Bug or Feature Request 🐞

If you encounter bugs, have questions, or want to propose new features, feel free to open an issue or submit a pull request. Suggestions for additional features, like **multi-class classification** or **support for other languages**, are always welcome 💡.

## Future Scope 🚀

The project can be extended in various ways:
- **Multilingual Support**: Expand the project to classify reviews in other languages using multilingual embeddings 🌐.
- **Real-Time Deployment**: Deploy the model in a web application that analyzes real-time reviews from Amazon or other platforms 🛒.
- **Multi-Class Classification**: Extend the sentiment classification to more than two classes, such as **neutral**, **positive**, and **negative** 🏷️.
- **Fine-Tuning BERT**: Replace the LSTM model with a transformer-based model like **BERT** for even better performance 📊.

## Conclusion 🎉

This project demonstrates the power of deep learning techniques like **RNN** and **LSTM** in text classification tasks. While the **Bidirectional LSTM** model achieved a high level of accuracy, there are still opportunities for future improvements, such as tuning hyperparameters, using more advanced embeddings, and deploying the model for real-world applications 🌍. Ultimately, this solution can help businesses automatically process large volumes of customer feedback, improving customer service and product offerings 💼.

