# Methods modified from original https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258

import os
import pandas as pd
import mlflow
import requests
import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

class SentenceLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
        
    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize(self, sentence, tags_to_include=[wordnet.NOUN, wordnet.VERB]):
        sentence = sentence.lower()
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag in tags_to_include:    
                #else use the tag to lemmatize the token
                word_stem = self.stemmer.stem(word)
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word_stem, tag))
        return " ".join(lemmatized_sentence)
