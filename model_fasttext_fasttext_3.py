from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import os, os.path
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
from sklearn.model_selection import KFold
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


def get_train_test_sets(train_indicies=None, test_indicies=None,test_size= 0.2, random_state=0):
    if train_indicies is None:
        X_train, X_test, y_train, y_test = train_test_split(df['project_details'], df['final_label'], test_size= test_size, random_state=random_state)
    else:
        X_train= []
        X_test = []
        y_train = []
        y_test = []
        for i in train_indicies:
            X_train.append(df[i])
            y_train.append(df['final_label'][i])
        for j in test_indicies:
            X_test.append(df[j])
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
    ### make train and test set compatible for fastText
    # ...
    ## train_set.txt shape:
    # __label__0 "some text..."  (pre-processed project details)
    # __label__1 "some text..."
    # __label__0 "some text..."
    train_file = "./data/fasttext/train_set.txt"  
    test_file = "./data/fasttext/test_set.txt"  ## test_set.txt actually not needed later
    with io.open(train_file,'w',encoding='utf8') as f:
        for i in range(0,len(y_train)):
            f.write("__label__" + str(y_train[i]) + " " + X_train[i] + "\n")
    
    with io.open(test_file,'w',encoding='utf8') as f:
        for i in range(0,len(y_test)):
            f.write("__label__" + str(y_test[i]) + " " + X_test[i] + "\n")
    
    return X_train, X_test, y_train, y_test, train_file



def get_results(params_wordembeddings, params_classification,name=0,train_indicies=None,test_indicies=None, random_state = 0, report = False):       
    _, X_test, _, y_test, train_file = get_train_test_sets(train_indicies, test_indicies,test_size= 0.2, random_state=random_state)
    model_name = "comb_" + str(name)
    # bin_path = "word_vectors/fasttext/" + model_name + ".bin" 
    vec_path = "word_vectors/fasttext/" + model_name + ".vec" 
    print("before unsup\n")
    #try:
    print("params_wordembeddings: " + str(params_wordembeddings))
    embeddings = fasttext.train_unsupervised(input="./data/fasttext/data.txt", **params_wordembeddings) 
    #except RuntimeError:
    #        print("RuntimeError occurred (probably due to high learning rate)\n")
    #        return None
    print("after unsup\n")
    # embeddings.save_model(bin_path)
    # embeddings = load_model(bin_path)
    ### convert from fasttext embeddings (would be saved as .bin) to .vec,
    ### in order to use the embeddings .vec file as pretrainedVectors for fasttext text classification
    from_bin_to_vec(embeddings, vec_path)
    # dimension of embeddings has to fit with dimension of look-up table (embeddings) in classification model
    params_classification["dim"] = embeddings.get_dimension()
    print("before sup\n")
    classification_model = fasttext.train_supervised(input=train_file, pretrainedVectors= vec_path, **params_classification)
    print("after sup\n")
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
    results['accuracy'] = round(float(accuracy),5)
    results['precision'] = round(float(precision),5)
    results['recall'] = round(float(recall),5)
    results['auc'] = round(float(auc),5)
    results['auprc'] = round(float(auprc),5)
    results['f1'] = round(float(f1),5)
    if report == True:
        results['report'] = classification_report(y_test, y_pred)
    print(str(results))
    return results

def get_combination_with_results(combination,all_keys, keys_wordembeddings):
    params_wordembeddings = {}
    params_classification = {}
    d1={}
    d2={}
    d3={}
    d4={}
    name = multiprocessing.current_process().name
    for a, b in zip(all_keys, combination):
        if len(params_wordembeddings) != len(keys_wordembeddings):
            params_wordembeddings[a] = b
        else:
            params_classification[a] = b
    results = get_results(params_wordembeddings, params_classification, name=name) # returns dict of accuracy, precision, recall, auc, auprc
    d1['params_wordembeddings'] = params_wordembeddings
    d2['params_classification'] = params_classification
    d3['results'] = results
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
    return [[params_wordembeddings,params_classification], results] 


def get_best_combination_with_results(param_grid_wordembeddings, param_grid_classification, score):
    keys_wordembeddings = list(param_grid_wordembeddings.keys())
    values_wordembeddings = list(param_grid_wordembeddings.values())
    keys_classification = list(param_grid_classification.keys())
    values_classification = list(param_grid_classification.values())
    all_values = values_wordembeddings + values_classification
    all_keys = keys_wordembeddings + keys_classification
    all_combinations = list(itertools.product(*all_values))
    print("A\n")
    num_available_cores = len(os.sched_getaffinity(0)) - 12
    print("B\n")
    pool = multiprocessing.Pool(processes=num_available_cores)
    print("C\n")
    f=partial(get_combination_with_results, all_keys=all_keys, keys_wordembeddings=keys_wordembeddings) 
    print("D\n")
    list_of_combination_results = pool.map(f, all_combinations) #returns list of [[params_wordembeddings_dict,params_classification_dict], results_dict] 
    list_of_combination_results = [item for item in list_of_combination_results if item[1] is not None]     
    max_score_value = max(results[score] for combination,results in list_of_combination_results)
    max_comb_results = [[combination,results] for combination,results in list_of_combination_results if results[score] == max_score_value] # list of [[params_representation_dict,params_classification_dict], results] 
    print("Length of max_comb_results :" + str(len(max_comb_results)))
    best_results = max_comb_results[0][1].copy() 
    best_params_wordembeddings = max_comb_results[0][0][0].copy()
    best_params_classification = max_comb_results[0][0][1].copy()
    return best_params_wordembeddings, best_params_classification, best_results

def get_averaged_results(params_wordembeddings, params_classification,num_runs=5,train_indicies = None, test_indicies = None, report= False):
    metrics = ["accuracy","precision","recall","auc","auprc","f1"]
    betw_results = {}
    final_results = {}
    random_state = 10
    for n in range(num_runs):
        if (report == True) and (n < (num_runs-1)):
            report = False
        else:
            report = True
        results = get_results(params_wordembeddings, params_classification,name = n,random_state = random_state+n ,report=report)
        print("results run " + str(n) + ": " + str(results))
        for m in metrics:
            betw_results.setdefault(m,[]).append(results[m])
        print("between results : " + str(betw_results))
    for m in metrics:
        m_list = betw_results[m]
        final_results[m] = round(float(sum(m_list)/len(m_list)),5)
    final_results['report'] = results['report']
    print(str(final_results))
    return final_results

def lang_dependency_set(test_size, lang):
    dep_indicies = range(len(lang_indicies[lang]))
    k = len(lang_indicies[lang]) * test_size
    dep_test_indicies = random.sample(dep_indicies, int(k)) # each time when called --> different set 
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

def compare_lang_dependency(test_size, lang, report=False):
    ### split train and test set indicies for 1st and 2nd set up *** each time when called --> different set
    train_indep_indicies, train_dep_indicies, test_indicies = lang_dependency_set(test_size = test_size, lang = lang)
    ### apply best params and run num_runs times to take the average of the results 
    # set number of runs
    # get results on 1st set up
    results_dep = get_averaged_results(best_params_wordembeddings,best_params_classification,train_indicies= train_dep_indicies, test_indicies=test_indicies, report= report) 
    # get results on 2nd set up
    results_indep = get_averaged_results(best_params_wordembeddings,best_params_classification, train_indicies=train_indep_indicies, test_indicies= test_indicies, report= report) 
    # compare results of 1st and 2nd set up
    if results_dep[score] > results_indep[score]:
        dep_result = 1  # dependent is better
    else:
        dep_result = 0  # independent is better
    return dep_result, results_dep, results_indep



################## Apply Fastext and fine tune the parameters for better results
'''
We consider two main steps for better results:
- tuning parameters for text representation (word embeddings)
- tuning parameters for text classification
The optimal classification threshold will also be applied
'''
####### tuning parameters for fasttext WORD EMBEDDINGS
'''
    input             # training file path (required)
    model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
    lr                # learning rate [0.05]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [5]
    minn              # min length of char ngram [3]
    maxn              # max length of char ngram [6]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [ns]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    verbose           # verbose [2]
'''
####### tuning parameters for fasttext WORD EMBEDDINGS
## Tutorial for word representations suggests to tune especially following parameters:
# - dim = [100,300]
# - minn and maxn
# - epoch (if dataset is massive, better < 5)
# - lr = [0.01,1.0] 
model=['skipgram']
loss=['hs']
dim = [int(i) for i in [50,100,200,300]] 
minn = [int(i) for i in [2,3,4]]
maxn = [int(i) for i in [5,6,7]]
epoch = [int(i) for i in [1,2,3,5,6]]
lr = [round(float(i),3) for i in list(np.arange(0.01, 0.1, 0.02))]  # Avoid RuntimeError: NaN encountered (due to too high lr)
param_grid_wordembeddings= {"model":model,"loss":loss,"dim":dim, "minn":minn, "maxn":maxn, "epoch":epoch, "lr":lr}

####### tuning parameters for fasttext CLASSIFICATION
'''
    input             # training file path (required)
    lr                # learning rate [0.1]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [1]
    minCountLabel     # minimal number of label occurences [1]
    minn              # min length of char ngram [0]
    maxn              # max length of char ngram [0]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [softmax]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    label             # label prefix ['__label__']
    verbose           # verbose [2]
    pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
'''
####### tuning parameters for fasttext CLASSIFICATION
## Tutorial for text classification suggests to tune especially following parameters:
# - epoch [5,50]
# - lr = [0.01,1.0] 
# - wordNgrams [1,5]
# And pre-processing the data (already done in another step)

# Dimension of look-up table in classification model has to be equal to dimension of embeddings,
# get_best_parameter_value_with_results function handles this issue
loss = ['hs']
epoch = [int(i) for i in list(np.arange(5, 50, 11))] 
lr = [round(float(i),3) for i in list(np.arange(0.01, 0.1, 0.02))] # Avoid RuntimeError: NaN encountered (due to too high lr)
wordNgrams = [int(i) for i in [1,2,3,4,5]]
param_grid_classification= {"loss": loss,"epoch":epoch, "lr":lr, "wordNgrams":wordNgrams}

################### load and prepare some DATA ########################################################################################

# load cleaned labeled data
df_raw = pd.read_csv('./data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()

# load language indicies
file = open('./data/lang_indicies.json','r',encoding='utf8')
lang_indicies = json.load(file)
file.close()

#### prepare data for fasttext text representation 
## make data compatible for fasttext 
## data.txt shape:
# __label__0 "some text..."  (pre-processed project details)
# __label__1 "some text..."
# __label__0 "some text..."

data_file = "./data/fasttext/data.txt"
with io.open(data_file,'w',encoding='utf8') as f:
    for i in range(0,len(df['final_label'])):
        f.write("__label__" + str(df['final_label'][i]) + " " + df['project_details'][i] + "\n")


######################################### ***** ADAPT THIS PART ***** ####################################################
#### test for one particular sklearn classification algo
# choose classifier
num = 0
pth = "./models/model_FastText/model_1/"
model_path = lambda num : pth + "results_" + str(num) + ".json" # adapt path accordingly
results_path = model_path(num)
score = "auprc" # choose among 'auprc', 'auc', 'f1', 'accuracy', 'precision', 'recall'
################################################# ***** RUN THIS PART ***** ###############################################
###### EVENTUALLY REMOVE FILE MANUALLY ######
# check if file already exists, if yes create new one
# prepare framework for saving results
results_object={}
results_object['tune_param_wordembeddings'] = param_grid_wordembeddings
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
#X_train, X_test, y_train, y_test, train_file = get_train_test_sets()
best_params_wordembeddings, best_params_classification, best_results = get_best_combination_with_results(param_grid_wordembeddings, param_grid_classification, score)

############## OR load the (saved) best results
with io.open(results_path,'r+',encoding='utf8') as file:
    results_object = json.load(file)

score_value = 0.0
best_comb_name = ""
for name,_ in results_object.items():
    if 'comb' not in name:
        continue
    v = results_object[name]['results'][score]
    if v > score_value:
        score_value = v
        best_comb_name = name

best_params_wordembeddings = results_object[best_comb_name]["params_wordembeddings"]
best_params_classification = results_object[best_comb_name]["params_classification"]
best_results = results_object[best_comb_name]["results"]

################## run 5 times with best parameter values and take the average 
averaged_results = get_averaged_results(best_params_representation,best_params_classification) 

################## save best parameter values and the results 
best_combination = {}
best_combination["best_params_representation"] = best_params_representation
best_combination["best_params_classification"] = best_params_classification
best_combination["best_results"] =  best_results
best_combination["best_average_results"] = averaged_results

file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object["best_combination"] = best_combination
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()

################## OUTPUT the best parameter values and the results

print("Best word embeddings parameter values according to " + score + ": \n")
for param in best_params_wordembeddings:
    best_value = best_params_wordembeddings[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")
print("Best classification parameter values according to " + score + ": \n")
for param in best_params_classification:
    best_value = best_params_classification[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")
print("Final (averaged) evaluation results: \n")
for metric in best_results:
    result = best_results[metric]
    print(metric + " : " + str(result) + "\n")
print("report : " + str(best_results["report"]) + "\n")

##################################### compare LANGUAGE DEPENDENCY with LANGUAGE INDEPENDENCY #######################################
'''
Test if performing on each language separately gives better results or independently of the language

Compare 1st set up with 2nd set up
- 1st set up: train on 80% german, evaluate on 20% german
- 2nd set up: train on 100% italian, 100% french, 100% english, 80% german all together at once. evaluate on *the same* 20% german
same for italian, french and english

'''


########## test language (in)dependency and save the results
metrics = ["accuracy","precision","recall","auc","auprc","f1"]
betw_results_dep = {}
betw_results_indep = {}
final_results = {}
test_size = 0.2
languages = ['de','fr','it','en']
half = int(len(languages)/2) # 2 if len(languages) is 5
dep_count = 0
num_runs = 5
report = False
for lang in languages:
    for i in range(num_runs):
        if i == (num_runs-1):
            report = True
        dep_result, results_dep, results_indep = compare_lang_dependency(test_size, lang, report=report) # dep_result returns 1 (dependent is better) or 0 (independent is better)
        for m in metrics:
            betw_results_dep.setdefault(m,[]).append(results_dep[m])
            betw_results_indep.setdefault(m,[]).append(results_indep[m])
    for m in metrics:
        m_list_dep = betw_results_dep[m]
        final_results_dep[m] = round(float(sum(m_list_dep)/len(m_list_dep)),5)
        m_list_indep = betw_results_indep[m]
        final_results_indep[m] = round(float(sum(m_list_indep)/len(m_list_indep)),5)
    final_results_dep['report'] = results_dep['report']
    final_results_indep['report'] = results_indep['report']
        ### save the results 
    lang_dependency_results = {}
    lang_dependency_results[lang + "_dependent"] = results_dep
    lang_dependency_results[lang + "_independent"] = results_indep
    lang_dependency_results["dependency_result"] = dep_result/num_runs
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object[lang + "_dependency_result"] = lang_dependency_results
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    dep_count = dep_count + dep_result

overall_dependency_result = dep_result/len(languages)
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




