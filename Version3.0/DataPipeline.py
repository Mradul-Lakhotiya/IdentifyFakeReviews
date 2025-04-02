import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class DataPipe:
    def __init__(self, df):
        self.df = df.copy()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        self.word2vec_model = None  # Will be initialized in fit()

    def preprocess_text(self, text):
        """
        Cleans and tokenizes text data.

        Args:
            text (str): Input text.

        Returns:
            list: Tokenized and lemmatized words.
        """
        if not isinstance(text, str) or pd.isna(text):  
            return []  # Return an empty list if text is NaN or not a string

        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower().strip()  # Convert to lowercase and remove extra spaces
        tokens = word_tokenize(text)  # Tokenize

        # Remove stopwords using set operation (faster lookup)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return tokens

    def text_to_vector(self, tokens):
        """
        Converts tokenized text into a numerical vector using Word2Vec.

        Args:
            tokens (list): List of words.

        Returns:
            np.array: Vector representation of text.
        """
        if self.word2vec_model is None:
            return np.zeros(100)  # Return zero vector if model is not trained

        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    def fit(self, df):
        """
        Fits encoders and trains the Word2Vec model.

        Args:
            df (DataFrame): Input dataset.
        """
        required_columns = {'category', 'label', 'text_'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        # Fit OneHotEncoder on 'category'
        self.one_hot_encoder.fit(df[['category']])

        # Fit LabelEncoder on 'label'
        self.label_encoder.fit(df['label'])

        # Tokenize text data
        tokenized_texts = df['text_'].apply(self.preprocess_text).tolist()

        # Initialize and train Word2Vec
        self.word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
        self.word2vec_model.build_vocab(tokenized_texts)
        self.word2vec_model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
        
    def c(self):
        """
        Returns the trained Word2Vec model.

        Returns:
            Word2Vec: Trained Word2Vec model.
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model has not been trained yet.")
        return self.word2vec_model

    def transform(self, df):
        """
        Transforms data using fitted encoders.

        Args:
            df (DataFrame): Input dataset.

        Returns:
            DataFrame: Encoded and vectorized dataframe.
        """
        df = df.copy()

        if 'category' not in df or 'label' not in df or 'text_' not in df:
            raise ValueError("Required columns 'category', 'label', or 'text_' are missing in the dataset.")

        # One-hot encode 'category'
        category_encoded = self.one_hot_encoder.transform(df[['category']])
        category_df = pd.DataFrame(category_encoded, columns=self.one_hot_encoder.get_feature_names_out(['category']))

        # Encode 'label'
        df['label'] = self.label_encoder.transform(df['label'])

        # Preprocess text and vectorize
        df['tokens'] = df['text_'].apply(self.preprocess_text)
        df['text_vector'] = df['tokens'].apply(self.text_to_vector)

        # Convert text vectors into DataFrame format
        text_vectors_df = pd.DataFrame(df['text_vector'].tolist())

        # Concatenate all processed features
        final_df = pd.concat([category_df, df[['rating', 'label']], text_vectors_df], axis=1)

        return final_df

    def fit_transform(self, df):
        """
        Fits and transforms the dataset.

        Args:
            df (DataFrame): Input dataset.

        Returns:
            DataFrame: Transformed dataframe with encoded features.
        """
        self.fit(df)
        return self.transform(df)
