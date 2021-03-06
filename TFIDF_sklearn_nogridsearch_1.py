# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
'''
Here I won't use GridSearchCV for fine-tuning since it doesn't search and apply the best decision threshold.
See e.g. https://stats.stackexchange.com/questions/390186/is-decision-threshold-a-hyperparameter-in-logistic-regression

'''

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

################# definitions

def equal(X_test,y_test):
    ## make number of yes == number of no
    # take the difference of number of yes and no labels
    diff = abs(y_test.count(1)-y_test.count(0))
    y_test_new = []
    X_test_new = [] 
    c = 0
    if y_test.count(1) < y_test.count(0): # basically this case is true, since there are much more 0's than 1's
        for i in range(0,len(y_test)):
            if (y_test[i] == 1) or (y_test[i] == 0 and c >= diff):
                y_test_new.append(y_test[i])
                X_test_new.append(X_test[i])
            elif y_test[i] == 0 and c < diff:
                c = c + 1
    elif y_test.count(0) < y_test.count(1): 
        for i in range(0,len(y_test)):
            if (y_test[i] == 0) or (y_test[i] == 1 and c >= diff):
                y_test_new.append(y_test[i])
                X_test_new.append(X_test[i])
            elif y_test[i] == 1 and c < diff:
                c = c + 1
    return X_test_new, y_test_new

def get_train_test_sets(features,train_indicies=None, test_indicies=None,test_size= 0.2, random_state=0):
    if train_indicies is None:
        X_train, X_test, y_train, y_test = train_test_split(features, df['final_label'], test_size= test_size, random_state=random_state)
    else:  # for language 
        X_train= []
        X_test = []
        y_train = []
        y_test = []
        for i in train_indicies:
            X_train.append(features[i])
            y_train.append(df['final_label'][i])
        for j in test_indicies:
            X_test.append(features[j])
            y_test.append(df['final_label'][j])   
    #conversion
    splitted_set = [y_train,X_train,y_test, X_test]
    for p in range(len(splitted_set)):
        if isinstance(splitted_set[p], list):
            pass
        else:
            splitted_set[p] = splitted_set[p].tolist()
    y_train = splitted_set[0]
    X_train = splitted_set[1]
    y_test = splitted_set[2]
    X_test = splitted_set[3]
    ### make (number of yes) == (number of no) in test set
    X_test_1, y_test_1 = equal(X_test,y_test)
    X_test = X_test_1.copy()
    y_test = y_test_1.copy()
    return X_train, X_test, y_train, y_test

def find_best_prc_threshold(target, predicted):
    #https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    # calculate pr-curve
    precision, recall, thresholds = precision_recall_curve(target, predicted)
    # convert to f score
    fscores = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = argmax(fscores) 
    best_threshold = thresholds[ix]
    best_fscore = fscores[ix]
    print('Best prc Threshold=%f, F-Score=%.3f' % (best_threshold, best_fscore))
    return best_threshold, best_fscore

def find_best_roc_threshold(target, predicted):
    #https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    # calculate roc-curve
    fpr, tpr, thresholds = roc_curve(target, predicted)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    best_threshold = thresholds[ix]
    best_gmean = gmeans[ix]
    print('Best roc Threshold=%f, G-Mean=%.3f' % (best_threshold, best_gmean))
    return best_threshold, best_gmean

def get_predictions(threshold, prediction_scores):
    """ Apply the threshold to the prediction probability scores
        and return the resulting label predictions
    """
    predictions = []
    for score in prediction_scores:
        if score >= threshold:
            predictions.append(1)
        elif score < threshold:
            predictions.append(0)
    return predictions


def perform( fun, **args ):
    return fun( **args )

'''
example:
def sayhello(fname, sname):
    print('hello ' + fname + ' ' + sname)

params = {'fname':'Julia', 'sname':'Eigenmann'}
perform(sayhello, **params)
'''

# see https://stackoverflow.com/questions/47895434/how-to-make-pipeline-for-multiple-dataframe-columns
def get_results(params_representation, params_classification, classifier, train_indicies=None, test_indicies=None, test_size=0.2, random_state = 0, report = False, curve= False):
    vectorizer = TfidfVectorizer(sublinear_tf=True, **params_representation)
    ##### return no results if ValueError should occurr (should not anymore actually)
    # ValueError: "max_df corresponds to < documents than min_df"
    # ValueError: "After pruning, no terms remain. Try a lower min_df or a higher max_df"
    try:
        features = vectorizer.fit_transform(df['project_details'].tolist()).toarray()
    except ValueError:  
            print("ValueError occurred\n")
            return None
    X_train, X_test, y_train, y_test = get_train_test_sets(features,train_indicies=train_indicies, test_indicies=test_indicies,test_size=test_size, random_state=random_state)
    print("y_test " + str(y_test) +"\n")
    ### apply hyperparameter and train model
    classification_model = perform(classifier, **params_classification) # e.g. classifier == LogisticRegression
    classification_model.fit(X_train, y_train)
    ### find the optimal classification threshold and predict class labels on a set based on that threshold    
    #generate class probabilites
    '''
    try: 
        probs = classification_model.predict_proba(X_test) # 2 elements will be returned in probs array,
        y_scores = probs[:,1] # 2nd element gives probability for positive class
    except:
        y_scores = classification_model.decision_function(X_test) # for svc function
    '''
    if 'SVC' in str(classifier):
        print('passed ' + str(classifier))
        y_scores = classification_model.decision_function(X_test)
    else:
        probs = classification_model.predict_proba(X_test) # 2 elements will be returned in probs array,
        y_scores = probs[:,1] # 2nd element gives probability for positive class      
    # find optimal probability threshold in pr-curve and roc_curve
    best_prc_threshold, best_fscore = find_best_prc_threshold(y_test, y_scores)
    best_roc_threshold, best_gmean = find_best_roc_threshold(y_test, y_scores)
    # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
    y_prc_pred = get_predictions(best_prc_threshold, y_scores)
    y_roc_pred = get_predictions(best_roc_threshold, y_scores)
    ################## Evaluation
    ## based on threshold with best fscore in precision-recall-curve
    accuracy_prc = accuracy_score(y_test, y_prc_pred)
    precision_prc = precision_score(y_test, y_prc_pred)
    recall_prc = recall_score(y_test, y_prc_pred)
    f1_prc = f1_score(y_test, y_prc_pred)
    gmean_prc = geometric_mean_score(y_test, y_prc_pred)
    tn_prc, fp_prc, fn_prc, tp_prc = confusion_matrix(y_test, y_prc_pred).ravel()
    ## based on threshold with best gmean in fpr-tpr-curve (or roc-curve)
    accuracy_roc = accuracy_score(y_test, y_roc_pred)
    precision_roc = precision_score(y_test, y_roc_pred)
    recall_roc = recall_score(y_test, y_roc_pred)
    f1_roc = f1_score(y_test, y_roc_pred)
    gmean_roc = geometric_mean_score(y_test, y_roc_pred)
    tn_roc, fp_roc, fn_roc, tp_roc = confusion_matrix(y_test, y_roc_pred).ravel()    
    if curve == True:
        roc_curve = metrics.roc_curve(y_test, y_scores, pos_label=1)
        precision_recall_curve = metrics.precision_recall_curve(y_test, y_scores, pos_label=1)
    auc = metrics.roc_auc_score(y_test, y_scores)
    auprc = metrics.average_precision_score(y_test, y_scores)
    results = {}
    results['best_prc_threshold'] = 'Threshold=%.5f in precision-recall-curve with best F-Score=%.5f' % (best_prc_threshold, best_fscore)
    results['best_roc_threshold'] = 'Threshold=%.5f in fpr-tpr-curve with best G-Mean=%.5f' % (best_roc_threshold, best_gmean)
    results['accuracy_prc'] = round(float(accuracy_prc),5)
    results['precision_prc'] = round(float(precision_prc),5)
    results['recall_prc'] = round(float(recall_prc),5)
    results['f1_prc'] = round(float(f1_prc),5)
    results['gmean_prc'] = round(float(gmean_prc),5)
    results['accuracy_roc'] = round(float(accuracy_roc),5)
    results['precision_roc'] = round(float(precision_roc),5)
    results['recall_roc'] = round(float(recall_roc),5)
    results['f1_roc'] = round(float(f1_roc),5)
    results['gmean_roc'] = round(float(gmean_roc),5)
    results['tn_prc'] = int(tn_prc)
    results['fp_prc'] = int(fp_prc)
    results['fn_prc'] = int(fn_prc)
    results['tp_prc'] = int(tp_prc)
    results['tn_roc'] = int(tn_roc)
    results['fp_roc'] = int(fp_roc)
    results['fn_roc'] = int(fn_roc)
    results['tp_roc'] = int(tp_roc)
    if curve == True:
         results["roc_curve"] = [[float(i) for i in list(sublist)] for sublist in roc_curve]
         results["precision_recall_curve"] = [[float(i) for i in list(sublist)] for sublist in precision_recall_curve]
    results['auc'] = round(float(auc),5)
    results['auprc'] = round(float(auprc),5)
    if report == True:
        results['report_prc'] = classification_report(y_test, y_prc_pred)
        results['report_roc'] = classification_report(y_test, y_roc_pred)
    print(results)
    return results

def get_combination_with_results(combination,all_keys, keys_representation, classifier,test_size=0.2):
    params_representation = {}
    params_classification = {}
    d1={}
    d2={}
    d3={}
    d4={}
    for a, b in zip(all_keys, combination):
        if len(params_representation) != len(keys_representation):
            params_representation[a] = b
        else:
            params_classification[a] = b
    print("1a\n")
    results = get_results(params_representation, params_classification, classifier,test_size=test_size) # returns dict of accuracy, precision, recall, auc, auprc, f1
    print("1b\n")
    d1['params_representation'] = params_representation
    d2['params_classification'] = params_classification
    d3['results'] = results
    if results is not None:
        lock_path = results_path + ".lock" 
        with FileLock(lock_path):
            file = open(results_path,'r',encoding='utf8')
            results_object = json.load(file)
            file.close()
            
            number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
            comb_nr = "comb_" + str(number_of_combinations+1)
            d4[comb_nr] = {}
            d4[comb_nr].update(d1)
            d4[comb_nr].update(d2)
            d4[comb_nr].update(d3)
            results_object.update(d4)
            
            file = open(results_path,'w+',encoding='utf8')
            file.write(json.dumps(results_object))
            file.close()
        print(results)
    return [[params_representation,params_classification], results] 


def get_best_combination_with_results(param_grid_representation, param_grid_classification, score, classifier,test_size=0.2):
    keys_representation = list(param_grid_representation.keys())
    values_representation = list(param_grid_representation.values())
    keys_classification = list(param_grid_classification.keys())
    values_classification = list(param_grid_classification.values())
    all_values = values_representation + values_classification
    all_keys = keys_representation + keys_classification
    all_combinations = list(itertools.product(*all_values))
    print("A\n")
    num_available_cores = len(os.sched_getaffinity(0))
    num_cores = num_available_cores - 10
    print("B\n")
    pool = multiprocessing.Pool(processes=num_cores)
    print("C\n")
    f=partial(get_combination_with_results, all_keys=all_keys, keys_representation=keys_representation, classifier=classifier, test_size=test_size) 
    print("D\n")
    list_of_combination_results = pool.map(f, all_combinations) #returns list of [[params_representation_dict,params_classification_dict], results] 
    print("E\n")
    list_of_combination_results = [item for item in list_of_combination_results if item[1] is not None] 
    max_score_value = max(results[score] for combination,results in list_of_combination_results)
    max_comb_results = [[combination,results] for combination,results in list_of_combination_results if results[score] == max_score_value] # list of [[params_representation_dict,params_classification_dict], results] 
    print("Length of max_comb_results :" + str(len(max_comb_results)))
    best_results = max_comb_results[0][1].copy() 
    best_params_representation = max_comb_results[0][0][0].copy()
    best_params_classification = max_comb_results[0][0][1].copy()
    print(best_results)
    return best_params_representation, best_params_classification, best_results


def get_averaged_results(params_representation, params_classification,classifier,num_runs=5,train_indicies=None, test_indicies=None,test_size=0.2, report=False, curve=True):
    betw_results = {}
    final_results = {}
    random_state = 10
    multiple_best_results = {}
    for n in range(num_runs): # make report for the last run only
        if n < (num_runs-1):
            r = False
        elif (report == True) and (n == num_runs-1):
            r = True
        else : # (report == False) and (n == num_runs-1)
            r = False
        results = get_results(params_representation, params_classification,classifier,train_indicies=train_indicies,test_indicies= test_indicies,test_size=test_size,random_state = random_state+n, report=r, curve=curve)
        multiple_best_results["best_results_" + str(n)] = results        
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()        
        results_object["multiple_best_results"] = multiple_best_results
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
        #print("results run " + str(n) + ": " + str(results))
        for m in results:
            betw_results.setdefault(m,[]).append(results[m])
        #print("between results : " + str(betw_results))
    for m in results:
        a = betw_results[m] 
        if not any(isinstance(el, list) for el in a):
            final_results[m] = round(float(sum(a)/len(a)),5)
        else: # e.g. a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
            b = list(map(list, zip(*a))) # [[[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]]]
            c = [list(map(list, zip(*i))) for i in b]  # [[[1, 7], [2, 8]], [[3, 9], [4, 10]], [[5, 11], [6, 12]]]
            d = [[round(float(sum(subsubc)/len(subsubc)),5) for subsubc in subc] for subc in c] # subc = [[1, 7], [2, 8]], subsubc = [1, 7]
            final_results[m] = d
    if report == True:
        final_results['report'] = results['report']
    #print(str(final_results))
    return final_results

def lang_dependency_set(lang, test_size=0.2):
    dep_indicies = range(len(lang_indicies[lang]))
    k = len(lang_indicies[lang]) * test_size
    dep_test_indicies = random.sample(dep_indicies, int(k)) # at each run, a new random sampling
    test_indicies = []
    train_indicies = []    
    for i in range(len(lang_indicies[lang])):
        df_index = lang_indicies[lang][i]
        if i in dep_test_indicies:
            test_indicies.append(df_index)
        else:
            train_indicies.append(df_index)
     # avoid problem:
     #  While using new_list = my_list, any modifications to new_list changes my_list everytime. 
     # --> use list.copy()
    train_dep_indicies = train_indicies.copy()
    for l in lang_indicies:
        if l == lang:
            continue
        else:
            train_indicies.extend(lang_indicies[l])   
    return train_indicies, train_dep_indicies, test_indicies  


def compare_lang_dependency(lang,test_size=0.2):
    ### split train and test set indicies for 1st and 2nd set up
    train_indep_indicies, train_dep_indicies, test_indicies = lang_dependency_set(lang = lang, test_size = test_size)
    print("lang : " + lang + "\n")
    print("train_indep_indicies length: " + str(len(train_indep_indicies)) + "\n")
    print("train_dep_indicies length: " + str(len(train_dep_indicies)) + "\n")
    print("test_indicies length: " + str(len(test_indicies)) + "\n")
    ### apply best params and run num_runs times to take the average of the results 
    # set number of runs
    num_runs = 1
    # get results on 1st set up
    results_dep = get_averaged_results(best_params_representation,best_params_classification,classifier,num_runs=num_runs,train_indicies=train_dep_indicies, test_indicies=test_indicies, report=True) 
    # get results on 2nd set up
    results_indep = get_averaged_results(best_params_representation,best_params_classification,classifier, num_runs=num_runs, train_indicies=train_indep_indicies, test_indicies=test_indicies, report= True) 
    # compare results of 1st and 2nd set up
    print("results_dep: " + str(results_dep) + "\n")
    print("results_indep: " + str(results_indep) + "\n")
    if results_dep[score] > results_indep[score]:
        dependency_result = 1  # dependent is better
    else:
        dependency_result = 0  # independent is better
    ### save the results 
    lang_dependency_results = {}
    lang_dependency_results[lang + "_dependent"] = results_dep
    lang_dependency_results[lang + "_independent"] = results_indep
    lang_dependency_results["dependency_result"] = dependency_result
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object[lang + "_dependency_result"] = lang_dependency_results
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    return dependency_result



################## Apply TFIDF and different sklearn classification algorithms and fine tune the parameters for better results
'''
We consider two main steps for better results:
- tuning parameters for text representation (TF-IDF)
- tuning parameters for text classification (sklearn class algos like random forest)
The optimal classification threshold will also be applied
'''
############### Tuning parameters for text REPRESENTATION TF-IDF  ##################
'''
- For this step, you can use TF-IDF from sickit-learn. You need to try two options:
a- apply TF-IDF on all projects regardless of the language 
b- Divide the projects you have based on the language (group 1 french, group2 german…) 
    and apply TF-IDF on each group seperately

TfidfVectorizer class can be initialized with the following parameters:

min_df: remove the words from the vocabulary which have occurred in less than ‘min_df’ number of files.
   (float in range [0.0, 1.0] or int (default=1))
   When building the vocabulary ignore terms that have a document frequency strictly lower
   than the given threshold. This value is also called cut-off in the literature.
   If float, the parameter represents a proportion of documents, integer absolute counts.
   This parameter is ignored if vocabulary is not None.
max_df: remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of files in corpus.
   (float in range [0.0, 1.0] or int (default=1.0))
   When building the vocabulary ignore terms that have a document frequency strictly higher
   than the given threshold (corpus-specific stop words). If float, the parameter represents
   a proportion of documents, integer absolute counts. This parameter is ignored
   if vocabulary is not None
sublinear_tf: set to True to scale the term frequency in logarithmic scale.
stop_words: remove the predefined stop words in 'english'.
use_idf: weight factor must use inverse document frequency.
ngram_range: (1, 2) to indicate that unigrams and bigrams will be considered.
'''

### maybe there is a way to remove words 'except of some special words' like 'upu' etc (see important words in labeling_algo.py).
# check on https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/feature_extraction/text.py#L1519 
# def _limit_features in for loop 'for term, old_index in list(vocabulary.items()):'

############### Tuning parameters for TF-IDF word vectors
min_df = [round(float(i),3) for i in list(np.arange(0.001, 0.055, 0.01))]
max_df = [round(float(i),2) for i in list(np.arange(0.75, 1.0, 0.05))]
param_grid_tfidf= {"max_df":max_df,"min_df":min_df}

###############  Tuning parameters for text CLASSIFICATION with sklearn algorithms ############### 
'''
We compare following sklearn classification models
- Random Forest
- Linear Support Vector Machine
- Multinomial Naive Bayes
- Logistic Regression.
'''

#### Hyperparameter tuning Random Forest 
n_estimators = [int(i) for i in [5, 10, 20, 30, 50]]
max_depth = [int(i) for i in [5, 8, 15, 25, 30]]

min_samples_split = [int(i) for i in [2, 5, 10, 15, 100]]
min_samples_leaf = [int(i) for i in [1, 2, 5, 10]] 

param_grid_rf = {'n_estimators':n_estimators, 'max_depth':max_depth,  
              'min_samples_split':min_samples_split, 
             'min_samples_leaf':min_samples_leaf}


#### Hyperparameter tuning Logistic Regression
penalty = ['l1', 'l2']
C = [float(i) for i in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]]
class_weight = ['balanced']
solver = ['liblinear', 'saga']
param_grid_lg = {"C":C, "penalty":penalty, "class_weight":class_weight, 'solver':solver}


#### Hyperparameter tuning Multinomial Naive Bayes
alpha = [float(i) for i in np.linspace(0.5, 1.5, 6)]
fit_prior = [True, False]
param_grid_mnb = {'alpha': alpha,'fit_prior': fit_prior}



#### Hyperparameter tuning Linear Support Vector Machine
penalty = ['l1', 'l2']
C = [float(i) for i in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]]
class_weight = ['balanced']
param_grid_lsvc  = {'C':C, 'penalty':penalty,"class_weight":class_weight}


####### LOAD SOME DATA ####################################################################################################

# load cleaned labeled data
df_raw = pd.read_csv('./data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()

# load language indicies
file = open('./data/lang_indicies.json','r',encoding='utf8')
lang_indicies = json.load(file)
file.close()

######################################### ***** ADAPT THIS PART ***** ####################################################
#### test for one particular sklearn classification algo
# choose classifier
param_grid_classification = param_grid_lg # choose among param_grid_lg, param_grid_rf, param_grid_mnb, param_grid_lsvc
classifier = LogisticRegression  # choose respectively among LogisticRegression, RandomForestClassifier, MultinomialNB, LinearSVC 
num = 0
pth = "./models/model_TFIDF_LogisticRegression/model_1/" # adapt accordingly among LogisticRegression, RandomForestClassifier, MultinomialNB, LinearSVC resp.
model_path = lambda num : pth + "results_" + str(num) + ".json" # adapt path accordingly
results_path = model_path(num)
score = "auprc" # choose among 'auprc', 'auc', 'f1', 'accuracy', 'precision', 'recall'
################################################# ***** RUN THIS PART ***** ###############################################
###### EVENTUALLY REMOVE FILE MANUALLY ######
# check if file already exists, if yes create new one
# prepare framework for saving results
results_object={}
results_object['tune_param_representation'] = param_grid_tfidf
results_object['tune_param_classification'] = param_grid_classification
results_object['score'] = score
if os.path.exists(results_path):
    bn_list = list(map(path.basename,iglob(pth+"*.json")))
    num_list = []
    for bn in bn_list:
        num_list.extend(int(i) for i in re.findall('\d+', bn))
    max_num = max(num_list)
    num = max_num + 1
    results_path = model_path(num)

with io.open(results_path,'w+',encoding='utf8') as file:
    json.dump(results_object, file) 

############## get best parameter values along with the results 
    '''
    training of 80% of all projects and
    evaluating of 20% of randomly chosen projects (independently of the language)
    '''
best_params_representation, best_params_classification, best_results = get_best_combination_with_results(param_grid_tfidf, param_grid_classification, score, classifier)

############## OR load the (saved) best results
## adapt results_path to the most recent saved results path
if os.path.exists(results_path):
    bn_list = list(map(path.basename,iglob(pth+"*.json")))
    num_list = []
    for bn in bn_list:
        num_list.extend(int(i) for i in re.findall('\d+', bn))
    max_num = max(num_list)
    results_path = model_path(max_num)

# open results file
file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()


if "best_combination" not in results_object: # EITHER: extract best combination if not saved as best combination
    score_value = 0.0
    best_comb_name = ""
    for name,res in results_object.items():
        if 'comb' not in name:
            continue
        if 'results' not in res:
            continue
        v = results_object[name]['results'][score]
        if v > score_value:
            score_value = v
            best_comb_name = name
    best_params_representation = results_object[best_comb_name]["params_representation"]
    best_params_classification = results_object[best_comb_name]["params_classification"]
    best_results = results_object[best_comb_name]["results"]
else: # OR: get saved best combination
    best_params_representation= results_object["best_combination"]["best_params_representation"]
    best_params_classification = results_object["best_combination"]["best_params_classification"]
    best_results = results_object["best_combination"]["best_results"]
################## run 5 times with best parameter values and take the average 

averaged_results = get_averaged_results(best_params_representation,best_params_classification,classifier) # apply best params and run num_runs times and take the average of the results as best result

################## save best parameter values and the results 
best_combination = {}
best_combination["best_params_representation"] = best_params_representation
best_combination["best_params_classification"] = best_params_classification
best_combination["best_results"] =  best_results
best_combination["best_averaged_results"] = averaged_results


file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close() 
results_object["best_combination"] = best_combination
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()

################## OUTPUT the best parameter values and the results
print("Best representation parameter values according to " + score + ": \n")
for param in best_params_representation:
    best_value = best_params_representation[param]
    print(param + " : " + str(best_value) + "\n")

print("\n")
print("Best classification parameter values according to " + score + ": \n")

for param in best_params_classification:
    best_value = best_params_classification[param]
    print(param + " : " + str(best_value) + "\n")

print("\n")
print("Final (" + str(num_runs) + "-averaged) evaluation results: \n")

for metric in best_results:
    result = best_results[metric]
    print(metric + " : " + str(result) + "\n")

##################################### compare LANGUAGE DEPENDENCY with LANGUAGE INDEPENDENCY #######################################
'''
Test if applying TFIDF on each language separately gives better results or independently of the language

Compare 1st set up with 2nd set up
- 1st set up: train on 80% german, evaluate on 20% german
- 2nd set up: train on 100% italian, 100% french, 100% english, 80% german all together at once. evaluate on *the same* 20% german
same for italian, french and english

'''

########## test language (in)dependency and save the results
## set test size
test_size = 0.2
languages = ['de','fr','it','en']
half = int(len(languages)/2) # 2 if len(languages) is 5
dep_count = 0
for lang in languages:
    dep_result = compare_lang_dependency(lang, test_size=test_size) # returns 1 (dependent is better) or 0 (independent is better)
    dep_count = dep_count + dep_result

overall_dependency_result = dep_count/len(languages)
## save the results 

file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object["overall_dependency_result"] = overall_dependency_result
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()

## output the results
if dep_count > half:
    print("Dependency gives better results: " + str(dep_count) + " out of " + str(len(languages)))
else:
    print("Dependency does not give better results: " + str(dep_count) + " out of " + str(len(languages)))

