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
import multiprocessing
from functools import partial
import itertools
import multiprocessing




########################### CLEANING STEP #############################################
'''
We want project_details and project_title to be transformed into a single string of
meaningful words in low letters and without numbers in order to apply TF-IDF.
CPV is stays a number.
'''
################### sum up all meaningless words
languages = ["it","de","fr","en"]
stopwords_lists = dict.fromkeys(languages, [])
for l in languages:
    stopwords_path = "./stopwords/sw_" + l + "_cleaned.txt"
    f = open(stopwords_path,"r")
    text = f.read()
    words = ast.literal_eval(text) # convert string representation of list into list
    stopwords_lists[l] = words

meaningless_words = []
for _, sw in stopwords_lists.items():
    meaningless_words.extend(sw)
# drop duplicates
meaningless_words = list(dict.fromkeys(meaningless_words))


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
    # remove multiple spaces, words smaller than 3 letters, and meaningless words
    text = ' '.join([w for w in text.split() if len(w)>2 and w not in meaningless_words])
    return text


def preprocess_projects(partition_tuple, df):
    project_keys = list(df.keys())   # ['final_label', 'project_details','CPV','project_title']
    failed_extraction = []
    cleaned_projects = []
    start = partition_tuple[0]
    end = partition_tuple[1]
    for i in range(start,end):
        cleaned_project = dict.fromkeys(project_keys, []) # initialize as empty list
        cleaned_project['project_details'] = df['project_details'][i]
        # catch project whose extraction has failed
        if cleaned_project['project_details'] == '{}':
            failed_extraction.append(i)
            continue
        cleaned_project['final_label'] = df['final_label'][i] # '0' or '1'
        
        ######### TODO: adapt when adapted cpv is adapted in extraction part
        cpv_raw = df['CPV'][i] # e.g. "['50000000']"
        # convert string representation of list into list
        cpv_raw1 = ast.literal_eval(cpv_raw) # e.g ['50000000']
        cpv = cpv_raw1[0] # e.g. '50000000'
    
        cleaned_project['CPV'] = cpv # e.g. '50000000'
        cleaned_project['project_title'] = df['project_title'][i]
    
        # summarize those text (features) which need to be cleaned
        texts_keys = ['project_details', 'project_title']
        for t in range(len(texts_keys)):
            text = cleaned_project[texts_keys[t]]
            # assign cleaned text
            cleaned_project[texts_keys[t]] = clean_text(text)
            cleaned_projects.append(cleaned_project)
        return [cleaned_projects, failed_extraction]


def get_partition_tuples(num_processes):
    partition = int(len(df)/num_processes)
    partitions = []
    for i in range(0,num_processes):
        partitions.append(i*partition)
    partitions.append(len(df))
    partition_tuples = []
    for i in range(0,len(partitions)-1):
        t = (partitions[i],partitions[i+1])
        partition_tuples.append(t)
    return partition_tuples
    
    


################### load unclean projects
# load labeled data
df_raw = pd.read_csv('./input/merged_file.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()


################### clean projects

# catch project whose extraction has failed
failed_extraction = []
# save each cleaned project into list
all_cleaned_projects = []

num_processes = 13

partition_tuples = get_partition_tuples(num_processes)

pool = multiprocessing.Pool(processes=num_processes)
f=partial(preprocess_projects, df=df) 

list_of_results = pool.map(f, partition_tuples) #returns list of [cleaned_projects, failed_extraction]
for c, f in list_of_results:
    all_cleaned_projects.extend(c) 
    failed_extraction.extend(f)

df_cleaned = pd.DataFrame(all_cleaned_projects) # columns are ['final_label', 'project_details','CPV','project_title']

################### save cleaned projects
df_cleaned.to_csv('./input/merged_file_cleaned.csv', sep='\t', encoding = 'utf-8')



