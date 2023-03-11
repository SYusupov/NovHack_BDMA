from sentence_lemmatizer import SentenceLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TextPreprocesser:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = SentenceLemmatizer()

    def vectorize(self, dataset) -> pd.DataFrame:
        lemmatized = dataset.apply(self.lemmatizer.lemmatize)
        trained = self.vectorizer.fit_transform(lemmatized)
        feature_names = self.vectorizer.get_feature_names_out()
        dense = trained.todense()
        denselist = dense.tolist()
        return pd.DataFrame(denselist, columns=feature_names)
