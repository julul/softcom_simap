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
from functools import partial
import itertools
import multiprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from langdetect import detect
from glob import iglob
from os import path
from filelock import FileLock



########## prepare data for language detection
## load unclean projects to detect language
# load labeled (unclean) data 
df_raw = pd.read_csv('./data/labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df_unclean = df_raw[['final_label', 'project_details','CPV','project_title']].copy()

# detect the language
lang_indicies = {}
for i in range(len(df_unclean)):
    t = df_unclean['project_details'][i]
    #l = TextBlob(t)  
    #lang = l.detect_language()
    lang = detect(t)
    if lang not in lang_indicies:
        lang_indicies[lang] = []
    lang_indicies[lang].append(i)
# print(lang_indicies.keys())  # show all languages found

## save the results 
results_path = './data/lang_indicies.json'
with io.open(results_path,'w+',encoding='utf8') as file:
    json.dump(lang_indicies, file) 


'''
# load language indicies
file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object = lang_indicies.copy()
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()
'''
