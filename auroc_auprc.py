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
from sklearn.metrics import classification_report

# good references:
# https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
# https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248

'''
def get_cmap(n, name='hsv'):
    'Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'
    return plt.cm.get_cmap(name, n)
'''

def save_roc_auc_plot(data):
    #cmap =  get_cmap(len(data["roc_auc_curve"]))
    plt.figure(figsize=(5,5), dpi=100)
    for _, (fpr,tpr,mod,auc) in enumerate(data["roc_auc_curve"]):
        plt.plot(fpr, tpr, linestyle='--',color=np.random.rand(3,), label= mod + ' (auc = %0.3f)' % auc)
    # color = cmap(i)
    plt.title("ROC curve")
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.legend(loc='lower right')
    plt.savefig('./plots/roc_auc_curve.png') 

def save_precision_recall_plot(data):
    #cmap =  get_cmap(len(data["precision_recall_curve"]))
    plt.figure(figsize=(5,5), dpi=100)
    for _, (recall,precision,mod,auprc) in enumerate(data["precision_recall_curve"]):
        plt.plot(recall, precision, linestyle='--',color=np.random.rand(3,), label= mod + ' (auprc = %0.3f)' % auprc)
    plt.title("PR curve")
    plt.xlabel('Recall -->')
    plt.ylabel('Precision -->')
    plt.legend(loc='lower right')
    plt.savefig('./plots/precision_recall_curve.png') 

models = ["TFIDF_LogisticRegression", "TFIDF_RandomForestClassifier", "TFIDF_MultinomialNB", "TFIDF_LinearSVC", "Bert", "FastText"]
model_path = lambda num : pth + "results_" + str(num) + ".json"
data = {}
data["roc_auc_curve"] = []
data["precision_recall_curve"] = []

for model in models:
    num = 0
    pth = "./models/model_" + model + "/model_1/" 
    results_path = model_path(num)
    print(results_path + "before\n")
    # adapt results_path to most recent saved results path
    if os.path.exists(results_path):
        bn_list = list(map(path.basename,iglob(pth+"*.json")))
        num_list = []
        for bn in bn_list:
            num_list.extend(int(i) for i in re.findall('\d+', bn))
        max_num = max(num_list)
        results_path = model_path(max_num) 
    print(results_path + "after\n")
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    if "best_combination" not in results_object:
        print("third condition\n")
        continue
    if "best_averaged_results" not in results_object["best_combination"]:
        print("first condition\n")
        continue
    if "roc_curve" not in results_object["best_combination"]["best_averaged_results"]:
        print("second condition\n")
        continue
    roc_curve = results_object["best_combination"]["best_averaged_results"]["roc_curve"]
    auc_score = results_object["best_combination"]["best_averaged_results"]["auc"]
    precision_recall_curve = results_object["best_combination"]["best_averaged_results"]["precision_recall_curve"]
    auprc_score= results_object["best_combination"]["best_averaged_results"]["auprc"]
    fpr = roc_curve[0]
    tpr = roc_curve[1]
    recall = precision_recall_curve[1]
    precision = precision_recall_curve[0] 
    if "LogisticRegression" in model:
        m = "LR"
    elif "RandomForest" in model:
        m = "RF"
    elif "Multinomial" in model:
        m = "MNB"
    elif "Linear" in model:
        m = "LSVC"
    elif "Bert" in model:
        m = "B"
    elif "FastText" in model:
        m = "FT" 
    curve1 = [fpr, tpr, m, auc_score]
    curve2 = [recall, precision, m, auprc_score]
    data["roc_auc_curve"].append(curve1)
    data["precision_recall_curve"].append(curve2)

save_roc_auc_plot(data)
save_precision_recall_plot(data)



'''
file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object["multiple_best_results"] = results_object.pop("averaged_best_results")

file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()
'''