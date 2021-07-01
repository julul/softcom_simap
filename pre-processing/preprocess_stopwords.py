# https://stackoverflow.com/questions/28716241/controlling-the-threshold-in-logistic-regression-in-scikit-learn
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import numpy as np
import json
import re
import ast
from nltk.tokenize import word_tokenize
from scipy.stats import randint
#import seaborn as sns # used for plot interactive graph. 
# https://stackoverflow.com/questions/3453188/matplotlib-display-plot-on-a-remote-machine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # should come after .use('somebackend')
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from langdetect import detect
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import ast
import re
import decimal
import pickle
from functools import reduce
from operator import or_
import unidecode
from sklearn.metrics import roc_curve, auc, roc_auc_score
import fasttext
from fasttext import load_model
import io
import time
import glob, os


################### prepreocess the gathered lists of stopwords ##################

################### some definitions

def clean_text(text):
    # lower text
    text = text.lower()
    # unaccent the words to make it language independent
    text = unidecode.unidecode(text)
    # text should contain only alphabetic chars
    # any non-alphabetic chars like ',' or '\n' should be replaced by a space
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)
    # remove multiple spaces and single chars
    text = ' '.join([w for w in text.split() if len(w)>1])
    return text

################### clean all meaningless words
languages = ["it","de","fr","en"]
for l in languages:
    stopwords_path = "./stopwords/sw_" + l + ".txt"
    cleaned_stopwords_path = "./stopwords/sw_" + l + "_cleaned.txt"
    f = open(stopwords_path,"r")
    text = f.read()
    # convert string representation of list into list
    words = ast.literal_eval(text)
    # clean words
    cleaned_words = clean_text(' '.join(words)).split()
    # convert list back tp string representation of list
    text = str(cleaned_words)
    f = open(cleaned_stopwords_path, 'w')
    f.write(text)