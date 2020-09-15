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



################# definitions
# The optimal cut-off would be where the true positive rate (tpr) is high
# and the false positive rate (fpr) is low,
# and tpr (- fpr) is zero or near to zero
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

def from_bin_to_vec(embeddings, vec_path):
    """ Produce from embeddings .bin file to .vec file and save it in vec_path.
        The .vec file is used for pretrainedVectors parameter for training the fastext classification model
    
    Parameters
    ----------
    embeddings: .bin file of the embeddings
    vec_path: path where the produced .vec file will be stored

    Returns
    -------
    returns nothing, but produces a .vec file of the embeddings and saves it in vec_path

    """
    f = embeddings
    # get all words from model
    words = f.get_words()
    with open(vec_path,'w') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass

def get_prediction_scores(trained_ft_model,X_test):
    """ Returns probability predictions of being labeled as positive on a set X_test.
        Fits for Fasttext.

    Parameters
    ----------
    trained_ft_model: trained supervised fasttext model
    X_test: set being predicted 

    Returns
    -------
    list of probabilities for being labeled positive on set X_test

    """
    ## generate class probabilites, where:
       # 1st element is probability for negative class,
       # 2nd element gives probability for positive class
    probs_ft = []
    for i in range(0,len(X_test)):
        # fasttext predict arguments: string, k=int
        # Given a string, get a list of labels (k) and a list of corresponding probabilities
        result = trained_ft_model.predict(X_test[i], k=2) # k=2 because we have 2 labels
        # result = ('__label__0', '__label__1'), array([0.972..., 0.027...])) 
        # first element is always the one with the higher probabiliy.
        # We want the first element always belongs to no and second to yes 
        if result[0][0] == '__label__0':
            probs_ft.append(result[1].tolist())
        elif result[0][0] == '__label__1':
            yes = result[1].tolist()[0]
            no = result[1].tolist()[1]
            probs = []
            probs.append(no)
            probs.append(yes)
            probs_ft.append(probs)
    ## take probabilites for being labeled positive
    y_scores_ft = [el[1] for el in probs_ft]
    return y_scores_ft


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
def sayhello(fname, sname):
    print('hello ' + fname + ' ' + sname)

params = {'fname':'Julia', 'sname':'Eigenmann'}
perform(sayhello, **params)
'''
def roc_model(tpr,fpr, thresholds):
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
    'tpr' : pd.Series(tpr, index = i), 
    '1-fpr': pd.Series(1-fpr, index=i), 
    'tf': pd.Series(tpr - (1-fpr), index=i), 
    'thresholds' : pd.Series(thresholds, index=i)})
    roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return roc

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


# see https://stackoverflow.com/questions/47895434/how-to-make-pipeline-for-multiple-dataframe-columns
def get_results(params_representation, params_classification, classifier):
    get_text_data = FunctionTransformer(lambda x: x[['title','project_details']], validate=False)
    get_numeric_data = FunctionTransformer(lambda x: x['CPV'], validate=False)
    vectorizer = TfidfVectorizer(sublinear_tf=True, **params_representation) # representation contains min_df, max_df

    pipe1 = Pipeline([('selector', get_text_data),('feature_extractor', vectorizer)])
    pipe2 = Pipeline([('selector', get_numeric_data)])
    
    union = FeatureUnion([('text_features', pipe1), ('numeric_features', pipe2)])
    X = union.fit_transform(df[['title','project_details','CPV']])
    
    '''
    random_state parameter:
    Use a new random number generator seeded by the given integer.
    Using an int will produce the same results across different calls.
    However, it may be worthwhile checking that your results are stable
    across a number of different distinct random seeds.
    Popular integer random seeds are 0 and 42.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, df['final_label'], test_size= 0.25, random_state=42)

    ## TODO: #### number of yes == number of no in test set

    ### apply hyperparameter and train model
    classification_model = perform(classifier, **params_classification) # e.g. classifier == LogisticRegression
    classification_model.fit(X_train, y_train)
    
    ### find the optimal classification threshold and predict class labels for on a set based on that threshold
    
    #generate class probabilites
    try: 
        probs = classification_model.predict_proba(X_test) # 2 elements will be returned in probs array,
        y_scores = probs[:,1] # 2nd element gives probability for positive class
    except:
        y_scores = classification_model.decision_function(X_test) # for svc function
    

    # Find optimal cutoff point
    # The optimal cut-off would be where the true positive rate (tpr) is high
    # and the false positive rate (fpr) is low,
    # and tpr (- fpr) is zero or near to zero
    # Plot a ROC of tpr vs 1-fpr
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores) 

    auc = metrics.auc(fpr, tpr)

    roc = roc_model(tpr,fpr, thresholds)

    # From the chart, the point where 'tpr' crosses '1-fpr' is the optimal cutoff point.
    # We want to find the optimal probability cutoff point

    # Find optimal probability threshold
    # Note: probs[:,1] will have the probability of being positive label

    opt_threshold = find_optimal_cutoff(y_test, y_scores)



    classification_model = fasttext.train_supervised(input=train_file, pretrainedVectors= vec_path, **params_classification)
    ### find and apply optimal (threshold) cutoff point
    # get scores, i.e. list of probabilities for being labeled positive on set X_test
    y_scores = get_prediction_scores(classification_model,X_test)
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
    results['accuracy'] = round(accuracy,5)
    results['precision'] = round(precision,5)
    results['recall'] = round(recall,5)
    results['auc'] = round(auc,5)
    results['auprc'] = round(auprc,5)
    results['f1'] = round(f1,5)
    return results

def get_combination_with_results(combination,all_keys, keys_representation, X_test, y_test):
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
    results = get_results(params_representation, params_classification, X_test, y_test) # returns dict of accuracy, precision, recall, auc, auprc, f1
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


def get_best_combination_with_results(param_grid_representation, param_grid_classification, score, X_test, y_test):
    keys_representation = list(param_grid_representation.keys())
    values_representation = list(param_grid_representation.values())
    keys_classification = list(param_grid_classification.keys())
    values_classification = list(param_grid_classification.values())
    all_values = values_representation + values_classification
    all_keys = keys_representation + keys_classification
    all_combinations = list(itertools.product(*all_values))
    print("A\n")
    num_available_cores = len(os.sched_getaffinity(0)) - 2
    print("B\n")
    pool = multiprocessing.Pool(processes=num_available_cores)
    print("C\n")
    f=partial(get_combination_with_results, all_keys=all_keys, keys_representation=keys_representation, X_test=X_test, y_test=y_test) 
    print("D\n")
    list_of_combination_results = pool.map(f, all_combinations) #returns list of [[params_representation_dict,params_classification_dict], results_dict] 
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

def get_final_results(num_runs, params_wordembeddings, params_classification,X_test, y_test):
    metrics = ["accuracy","precision","recall","auc","auprc","f1"]
    final_results = dict.fromkeys(metrics, [])
    for n in range(num_runs):
        results = get_results(str(n), params_wordembeddings, params_classification,X_test, y_test)
        for metric in metrics:
            result = results[metric]
            final_results[metric].append(result)
    for metric in metrics:
        l = final_results[metric]
        final_results[metric] = sum(l)/len(l)
    return final_results  



################## loading data
df = pd.read_csv('./input/merged_file_cleaned.csv', sep='\t', encoding = 'utf-8')
df= df[['project_title', 'project_details','CPV','final_label']].copy() # columns are ['final_label', 'project_details','CPV','project_title']

################## split and prepare data for text classification
# Validation Set approach : take 75% of the data as the training set and 25 % as the test set. X is a dataframe with the input variable
# K fold cross-validation approach as well?
length_to_split = int(len(df) * 0.75)

X = df['project_details']
y = df['final_labels']

## Splitting the X and y into train and test datasets
X_train, X_test = X[:length_to_split], X[length_to_split:]
y_train, y_test = y[:length_to_split], y[length_to_split:]
#conversion
y_train = y_train.tolist()
X_train = X_train.tolist()
y_test = y_test.tolist()
X_test = X_test.tolist()

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

y_test_old = y_test
X_test_old = X_test
y_test = y_test_new
X_test = X_test_new


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

min_df = list(np.arange(1, 200, 1))
min_df = [float(i/1000) for i in min_df]  # to make it JSON serializable
max_df = list(np.arange(750, 1000, 1)) 
max_df = [float(i/1000) for i in max_df]  # to make it JSON serializable
param_grid_tfidf= {"min_df":min_df,"max_df":max_df}

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



#### test for one particular sklearn classification algo

param_grid_classification = param_grid_lg

score = "auprc" 
results_path = "./model_TFIDF_sklearn/model_1/results.json"
results_object={}
results_object['tune_param_tfidf'] = param_grid_tfidf
results_object['tune_param_classification'] = param_grid_classification
results_object['score'] = score
if os.path.exists(results_path):
    os.remove(results_path)
with io.open(results_path,'w+',encoding='utf8') as file: # syntax error after if statement whyy??
    json.dump(results_object, file) # indendation error why????
best_params_tfidf, best_params_classification, best_results = get_best_combination_with_results(param_grid_tfidf, param_grid_classification, score, X_test, y_test)



