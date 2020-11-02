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
import multiprocessing
from functools import partial
import itertools
import multiprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from langdetect import detect
from glob import iglob

################# definitions
# The optimal cut-off would be where the true positive rate (tpr) is high
# and the false positive rate (fpr) is low,
# and tpr (- fpr) is zero or near to zero

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

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model
        related to the event rate

    Parameters
    ----------
    target: Matrix with dependent or target data, where rows are observations
    predicted_ Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def get_predictions(threshold, prediction_scores):
    """ Apply the threshold to the prediction probability scores
        and return the resulting label predictions

    Parameters
    ----------
    threshold: the threshold we want to apply
    prediction_scores: prediction probability scores, i.e. list of probabilites of being labeled positive

    Returns
    -------
    list of label predictions, i.e. list of 1's and 0's

    """
    predictions = []
    for score in prediction_scores:
        if score >= threshold[0]:
            predictions.append(1)
        elif score < threshold[0]:
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
def get_results(params_representation, params_classification, classifier, X_train, y_train, X_test, y_test):
    ### apply hyperparameter and train model
    print("2a\n")
    classification_model = perform(classifier, **params_classification) # e.g. classifier == LogisticRegression
    classification_model.fit(X_train, y_train)
    print("2b\n") 
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
        y_scores = classification_model.decision_function(X_test)
    else:
        probs = classification_model.predict_proba(X_test) # 2 elements will be returned in probs array,
        y_scores = probs[:,1] # 2nd element gives probability for positive class      
    # find optimal probability threshold
    opt_threshold = find_optimal_cutoff(y_test, y_scores)
    # apply optimal threshold to the prediction probability and get label predictions
    y_pred = get_predictions(opt_threshold, y_scores) 
    ################## Evaluation
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    auprc = metrics.average_precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    results = {}
    results['accuracy'] = round(float(accuracy),5)
    results['precision'] = round(float(precision),5)
    results['recall'] = round(float(recall),5)
    results['auc'] = round(float(auc),5)
    results['auprc'] = round(float(auprc),5)
    results['f1'] = round(float(f1),5)
    return results

def get_combination_with_results(combination,all_keys, keys_representation, classifier,X_train, y_train, X_test, y_test):
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
    results = get_results(params_representation, params_classification, classifier, X_train, y_train, X_test, y_test) # returns dict of accuracy, precision, recall, auc, auprc, f1
    print("1b\n")
    d1['params_representation'] = params_representation
    d2['params_classification'] = params_classification
    d3['results'] = results
    with io.open(results_path,'r+',encoding='utf8') as file:
        results_object = json.load(file)
        number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
        comb_nr = "comb_" + str(number_of_combinations+1)
        d4[comb_nr] = {}
        d4[comb_nr].update(d1)
        d4[comb_nr].update(d2)
        d4[comb_nr].update(d3)
        results_object.update(d4)
        file.seek(0)  # not sure if needed 
        json.dump(results_object, file)
    print(results)
    return [[params_representation,params_classification], results] 


def get_best_combination_with_results(param_grid_representation, param_grid_classification, score, classifier,X_train, y_train, X_test, y_test):
    keys_representation = list(param_grid_representation.keys())
    values_representation = list(param_grid_representation.values())
    keys_classification = list(param_grid_classification.keys())
    values_classification = list(param_grid_classification.values())
    all_values = values_representation + values_classification
    all_keys = keys_representation + keys_classification
    all_combinations = list(itertools.product(*all_values))
    print("A\n")
    num_available_cores = len(os.sched_getaffinity(0)) - 3
    print("B\n")
    pool = multiprocessing.Pool(processes=num_available_cores)
    print("C\n")
    f=partial(get_combination_with_results, all_keys=all_keys, keys_representation=keys_representation, classifier=classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test) 
    print("D\n")
    list_of_combination_results = pool.map(f, all_combinations) #returns list of [[params_representation_dict,params_classification_dict], results_dict] 
    print("E\n")
    accuracy_scores = [results["accuracy"] for combination,results in list_of_combination_results]
    precision_scores = [results["precision"] for combination,results in list_of_combination_results]
    recall_scores = [results["recall"] for combination,results in list_of_combination_results]
    auc_scores = [results["auc"] for combination,results in list_of_combination_results] # auc scores
    auprc_scores = [results["auprc"] for combination,results in list_of_combination_results] # auprc scores
    f1_scores = [results["f1"] for combination,results in list_of_combination_results] # f1 scores 
    if score == "auprc":
        index_max = max(range(len(auprc_scores)), key=auprc_scores.__getitem__)
    elif score == "auc":
        index_max = max(range(len(auc_scores)), key=auc_scores.__getitem__)
    elif score == "recall":
        index_max = max(range(len(recall_scores)), key=recall_scores.__getitem__)
    elif score == "precision":
        index_max = max(range(len(precision_scores)), key=precision_scores.__getitem__)
    elif score == "accuracy":
        index_max = max(range(len(accuracy_scores)), key=accuracy_scores.__getitem__)
    elif score == "f1":
        index_max = max(range(len(f1_scores)), key=f1_scores.__getitem__)
    best_results = {}
    best_results["auprc"] = auprc_scores[index_max]
    best_results["auc"] = auc_scores[index_max]
    best_results["recall"] = recall_scores[index_max]
    best_results["precision"] = precision_scores[index_max]
    best_results["accuracy"] = accuracy_scores[index_max]
    best_results["f1"] = f1_scores[index_max]
    best_combination = list_of_combination_results[index_max][0]  # [params_representation,params_classification]
    best_params_representation = best_combination[0]
    best_params_classification = best_combination[1]
    return best_params_representation, best_params_classification, best_results


def get_final_results(num_runs, params_representation, params_classification,classifier,X_train, y_train, X_test, y_test):
    metrics = ["accuracy","precision","recall","auc","auprc","f1"]
    betw_results = {}
    final_results = {}
    for n in range(num_runs):
        results = get_results(params_representation, params_classification,classifier,X_train, y_train, X_test, y_test)
        #print("results run " + str(n) + ": " + str(results))
        for m in metrics:
            betw_results.setdefault(m,[]).append(results[m])
        #print("between results : " + str(betw_results))
    for m in metrics:
        m_list = betw_results[m]
        final_results[m] = round(float(sum(m_list)/len(m_list)),5)
    #print(str(final_results))
    return final_results

def lang_dependency_set(test_size, lang):
    k = len(lang_indicies[lang]) * test_size
    indicies = random.sample(range(len(lang_indicies[lang])), int(k))
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    
    for i in range(len(lang_indicies[lang])):
        index = lang_indicies[lang][i]
        if i in indicies:
            X_test.append(df['project_details'][index])
            y_test.append(df['final_label'][index])
        else:
            X_train.append(df['project_details'][index])
            y_train.append(df['final_label'][index])
   
    X_train_dep = X_train
    y_train_dep = y_train      
    
    for l in lang_indicies:
        if l == lang:
            print(l)
            continue
        for i in range(len(lang_indicies[l])):
            index = lang_indicies[l][i]
            X_train.append(df['project_details'][index])
            y_train.append(df['final_label'][index])
    
    return X_train.tolist(), y_train.tolist(), X_train_dep.tolist(), y_train_dep.tolist(), X_test.tolist(), y_test.tolist()

def compare_lang_dependency(test_size, lang):
    ### split train and test sets for 1st and 2nd set up
    # 1st set up:  X_train_dep, y_train_dep, X_test_dep, y_test_dep
    # 2nd set up:  X_train_indep, y_train_indep, X_test_dep, y_test_dep
    X_train_indep, y_train_indep, X_train_dep, y_train_dep, X_test_dep, y_test_dep = lang_dependency_set(test_size = test_size, lang = lang)
    ### apply best params and run num_runs times to take the average of the results 
    # set number of runs
    num_runs = 5
    # get results on 1st set up
    results_dep = get_final_results(num_runs, best_params_representation,best_params_classification,classifier, X_train_dep, y_train_dep, X_test_dep, y_test_dep) 
    # get results on 2nd set up
    results_indep = get_final_results(num_runs, best_params_representation,best_params_classification,classifier, X_train_indep, y_train_indep, X_test_dep, y_test_dep) 
    # compare results of 1st and 2nd set up
    if results_dep[score] > results_indep[score]:
        dependency_result = 1  # dependent is better
    else:
        dependency_result = 0  # independent is better
    
    ### save the results 
    lang_dependency_results = {}
    lang_dependency_results[lang + "_dependent"] = results_dep
    lang_dependency_results[lang + "_independent"] = results_indep
    lang_dependency_results["dependency_result"] = dependency_result
    with io.open(results_path,'r+',encoding='utf8') as file:
        results_object = json.load(file)
        results_object[lang + "_dependency_result"] = lang_dependency_results
        file.seek(0)  # not sure if needed 
        json.dump(results_object, file)
    return comparison_result



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
min_df = list(np.arange(0.05, 2.05, 0.05))
min_df = [round(float(i),2) for i in min_df]
max_df = list(np.arange(0.75, 1.0, 0.01))
max_df = [round(float(i),2) for i in max_df]

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
n_estimators = [5, 10, 20, 30, 50]
n_estimators = [int(i) for i in n_estimators] # to make it JSON serializable
max_depth = [5, 8, 15, 25, 30]
max_depth = [int(i) for i in max_depth]

min_samples_split = [2, 5, 10, 15, 100]
min_samples_split = [int(i) for i in min_samples_split]
min_samples_leaf = [1, 2, 5, 10] 
min_samples_leaf = [int(i) for i in min_samples_leaf]

param_grid_rf = {'n_estimators':n_estimators, 'max_depth':max_depth,  
              'min_samples_split':min_samples_split, 
             'min_samples_leaf':min_samples_leaf}


#### Hyperparameter tuning Logistic Regression
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
C = [float(i) for i in C]
class_weight = ['balanced']
solver = ['liblinear', 'saga']
param_grid_lg = {"C":C, "penalty":penalty, "class_weight":class_weight, 'solver':solver}


#### Hyperparameter tuning Multinomial Naive Bayes
alpha = np.linspace(0.5, 1.5, 6)
alpha = [float(i) for i in alpha]
fit_prior = [True, False]
param_grid_mnb = {'alpha': alpha,'fit_prior': fit_prior}



#### Hyperparameter tuning Linear Support Vector Machine
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
C = [float(i) for i in C]
class_weight = ['balanced']
param_grid_lsvc  = {'C':C, 'penalty':penalty,"class_weight":class_weight}




######################################### ***** ADAPT THIS PART ***** ####################################################
#### test for one particular sklearn classification algo
# choose classifier
param_grid_classification = param_grid_lg # choose among param_grid_lg, param_grid_rf, param_grid_mnb, param_grid_lsvc
classifier = LogisticRegression  # choose respectively among LogisticRegression, RandomForestClassifier, MultinomialNB, LinearSVC 
num = 0
pth = "./models/model_TFIDF_LogisticRegression/model_1/"
model_path = lambda num : pth + "results_" + str(num) + ".json" # adapt path accordingly
results_path = model_path(num)
score = "auprc" # choose among 'auprc', 'auc', 'f1', 'accuracy', 'precision', 'recall'
##########################################################################################################################
# prepare framework for saving results
results_object={}
results_object['tune_param_representation'] = param_grid_tfidf
results_object['tune_param_classification'] = param_grid_classification
results_object['score'] = score

################## loading cleaned labeled data
# load cleaned labeled data
df_raw = pd.read_csv('./data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()

################## split and prepare data for text classification
'''
 training of 80% of all projects and
 evaluating of 20% of randomly chosen projects (independently of the language)
'''
X_train, X_test, y_train, y_test = train_test_split(df['project_details'], df['final_label'], test_size= 0.2, random_state=42)
#conversion
y_train = y_train.tolist()
X_train = X_train.tolist()
y_test = y_test.tolist()
X_test = X_test.tolist()
### (number of yes) == (number of no) in test set
X_test_1, y_test_1 = equal(X_test,y_test)
X_test = X_test_1
y_test = y_test_1


################################################# ***** RUN THIS PART ***** ###############################################
###### REMOVE FILE MANUALLY IF WANTED ######
# check if file already exists, if yes create new one
if os.path.exists(results_path):
    bn_list = list(map(path.basename,iglob(pth+"*.json")))
    num_list = []
    for bn in bn_list:
        num_list.extend([int(s) for s in bn.split() if s.isdigit()])
    max_num = max(num_list)
    num = max_num + 1
    results_path = model_path(num)

with io.open(results_path,'w+',encoding='utf8') as file:
    json.dump(results_object, file) 

################## get best parameter values along with the results 
best_params_representation, best_params_classification, best_results = get_best_combination_with_results(param_grid_tfidf, param_grid_classification, score, classifier,X_train, y_train, X_test, y_test)

################## run 5 times with best parameter values from and take the average 
num_runs = 5
final_results = get_final_results(num_runs, best_params_representation,best_params_classification,classifier, X_train, y_train, X_test, y_test) # apply best params and run num_runs times and take the average of the results as best result

################## save best parameter values and the results 
best_combination = {}
best_combination["best_params_representation"] = best_params_representation
best_combination["best_params_classification"] = best_params_classification
best_combination["best_results"] = final_results
with io.open(results_path,'r+',encoding='utf8') as file:
    results_object = json.load(file)
    results_object["best_combination"] = best_combination
    file.seek(0)  # not sure if needed 
    json.dump(results_object, file)

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
print("Final evaluation results: \n")
for metric in best_results:
    result = best_results[metric]
    print(metric + " : " + str(result) + "\n")

##################################### compare LANGUAGE DEPENDENCY with INDEPENDENCY #######################################
'''
Test if applying TFIDF on each language separately gives better results or independently of the language

Compare 1st set up with 2nd set up
- 1st set up: train on 80% german, evaluate on 20% german
- 2nd set up: train on 100% italian, 100% french, 100% english, 80% german all together at once. evaluate on *the same* 20% german
same for italian, french and english

'''
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


########## test language (in)dependency and save the results
## set test size
test_size = 0.2
languages = ['de','fr','it','en']
half = int(len(languages)/2) # 2 if len(languages) is 5
dep_count = 0
for lang in languages:
    dep_result = compare_lang_dependency(test_size, lang) # returns 1 (dependent is better) or 0 (independent is better)
    dep_count = dep_count + dep_result

## save the results 
with io.open(results_path,'r+',encoding='utf8') as file:
    results_object = json.load(file)
    results_object["overall_dependency_result"] = str(dep_count) + "/" + str(len(languages))
    file.seek(0)  # not sure if needed 
    json.dump(results_object, file)

## output the results
if dep_count > half:
    print("Dependency gives better results: " + str(dep_count) " out of " + str(len(languages)))
else:
    print("Dependency does not give better results: " + str(dep_count) " out of " + str(len(languages)))

