import re
import pandas as pd
import os, os.path
import numpy as np
import json
import re
#import seaborn as sns # used for plot interactive graph. 
# https://stackoverflow.com/questions/3453188/matplotlib-display-plot-on-a-remote-machine
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import re
import fasttext
import io
from functools import partial
import itertools
import multiprocessing
import random
from glob import iglob
from os import path
from filelock import FileLock
from sklearn.metrics import classification_report
from numpy import argmax
from numpy import sqrt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report
import argparse

################# exceptions
class Error(Exception):
    # Base class for other exceptions    
    pass

class NotuningError(Error):
    # Raised when choosing 'fold1results' runmode even though there was no tuning process (with 'fold1') at all.
    pass

class ReferenceError(Error):
    # Raised when reference argument refers to a non existent results file 
    pass

################# definitions
def get_best_combination_name(results_object):
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
    return best_comb_name

model_path = lambda num : pth + "results_" + str(num) + ".json" # adapt path accordingly
def adapt_resultspath(pth, pos=0):
    # pos=0 to access most recent already existing file. 
    # pos=1 to  access new file that is not created yet
    # If no file exists, point to first file path that is not created yet (/results_0) .
    num = 0
    res_path = model_path(num)
    if os.path.exists(res_path):
        bn_list = list(map(path.basename,iglob(pth+"*.json")))
        num_list = []
        for bn in bn_list:
            num_list.extend(int(i) for i in re.findall('\d+', bn))
        max_num = max(num_list)
        new_num = max_num + pos
        res_path = model_path(new_num)
    return res_path


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

def get_train_test_sets(train_indicies=None, test_indicies=None,test_size= 0.2, random_state=0, testing=None):
    if train_indicies is None and testing is None:
        X_train, X_test, y_train, y_test = train_test_split(df['project_details'], df['final_label'], test_size= test_size, random_state=random_state)
    elif train_indicies is not None or "Ines" in testing: # for languages or for "Ines"
        print("In get train test sets.")
        print("train_indicies length :" + str(len(train_indicies)))
        print("test_indicies length :" + str(len(test_indicies)))
        X_train= []
        X_test = []
        y_train = []
        y_test = []
        for i in train_indicies:
            X_train.append(df['project_details'][i])
            y_train.append(df['final_label'][i])
        for j in test_indicies:
            X_test.append(df['project_details'][j])
            y_test.append(df['final_label'][j])
    elif testing=="Julia":
        X_train= []
        X_test = []
        y_train = []
        y_test = []
        for i in range(0,df.shape[0]):
            X_train.append(df2['project_details'][i])
            y_train.append(df2['final_label'][i])
        for j in range(df.shape[0],df2.shape[0]):
            X_test.append(df2['project_details'][j])
            y_test.append(df2['final_label'][j])            
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
    if "Ines" not in testing:
        X_test_1, y_test_1 = equal(X_test,y_test)
        X_test = X_test_1.copy()
        y_test = y_test_1.copy()
    ### make train and test set compatible for fastText
    # ...
    ## train_set.txt shape:
    # __label__0 "some text..."  (pre-processed project details)
    # __label__1 "some text..."
    # __label__0 "some text..."
    train_file = "../data/fasttext/train_set.txt"  
    test_file = "../data/fasttext/test_set.txt"  ## test_set.txt actually not needed later
    with io.open(train_file,'w',encoding='utf8') as f:
        for i in range(0,len(y_train)):
            f.write("__label__" + str(y_train[i]) + " " + X_train[i] + "\n")
    with io.open(test_file,'w',encoding='utf8') as f:
        for i in range(0,len(y_test)):
            f.write("__label__" + str(y_test[i]) + " " + X_test[i] + "\n")
    return X_train, X_test, y_train, y_test, train_file, y_train.count(1), y_train.count(0), y_test.count(1), y_test.count(0) 

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
    """
    predictions = []
    for s in prediction_scores:
        if s >= threshold:
            predictions.append(1)
        elif s < threshold:
            predictions.append(0)
    return predictions



def get_results(params_wordembeddings, params_classification,name=0,train_indicies=None,test_indicies=None, valid_indicies=None, test_size=0.2, random_state = 0, report = False, curve= False, save_results=True, repeat=False, testing= None):       
    print("in get_results, testing: " + testing)
    if not repeat:
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        for _,res in results_object.items():
            if 'params_wordembeddings' not in res:
                continue
            if res['params_wordembeddings'] == params_wordembeddings and res['params_classification'] == params_classification and "Ines" not in testing:
                print("wordembeddings and classification parameters already exist in the current results_file\n")
                return res['results']
    if "Ines" in testing:
        _, X_valid, _, y_valid, _, _, _, y_valid_1, y_valid_0 = get_train_test_sets(train_indicies = train_indicies, test_indicies = valid_indicies,test_size= test_size, random_state=random_state, testing=testing)               
    _, X_test, _, y_test, train_file, y_train_1, y_train_0, y_test_1, y_test_0 = get_train_test_sets(train_indicies, test_indicies,test_size= test_size, random_state=random_state, testing=testing)
    model_name = "comb_" + str(name)
    # bin_path = "word_vectors/fasttext/" + model_name + ".bin" 
    vec_path = "../data/fasttext/wordvectors_" + model_name + ".vec" 
    print("before unsup\n")
    #try:
    print("params_wordembeddings: " + str(params_wordembeddings))
    embeddings = fasttext.train_unsupervised(input="../data/fasttext/data.txt", **params_wordembeddings) 
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
    # train 
    classification_model = fasttext.train_supervised(input=train_file, pretrainedVectors= vec_path, **params_classification)
    if "Ines" in testing:
        trainedmodelfiles[classifier + "_Ines"] = 'trained_' + classifier + '_model_Ines.bin'
        classification_model.save_model('trained_' + classifier + '_model_Ines.bin')
        # get scores, i.e. list of probabilities for being labeled positive on set X_valid
        y_scores_valid = get_prediction_scores(classification_model,X_valid)
        # find optimal probability threshold in pr-curve and roc_curve
        # Threshold same for validation as for testing
        best_prc_threshold, best_fscore = find_best_prc_threshold(y_valid, y_scores_valid)
        best_roc_threshold, best_gmean = find_best_roc_threshold(y_valid, y_scores_valid)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred_valid = get_predictions(best_prc_threshold, y_scores_valid)
        y_roc_pred_valid = get_predictions(best_roc_threshold, y_scores_valid)
        # test
        # get scores, i.e. list of probabilities for being labeled positive on set X_test
        y_scores= get_prediction_scores(classification_model,X_test)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred = get_predictions(best_prc_threshold, y_scores)
        y_roc_pred = get_predictions(best_roc_threshold, y_scores)  
    else:    
        if "Julia" in testing:
            trainedmodelfiles[classifier + "_Julia"] = 'trained_' + classifier + '_model_Julia.bin'
            classification_model.save_model('trained_' + classifier + '_model_Julia.bin')
        else: 
            trainedmodelfiles[classifier] = 'trained_' + classifier + '_model.bin'
            classification_model.save_model('trained_' + classifier + '_model.bin')
        # test
        # get scores, i.e. list of probabilities for being labeled positive on set X_test
        y_scores= get_prediction_scores(classification_model,X_test)
        # find optimal probability threshold in pr-curve and roc_curve
        # Threshold same for validation as for testing
        best_prc_threshold, best_fscore = find_best_prc_threshold(y_test, y_scores)
        best_roc_threshold, best_gmean = find_best_roc_threshold(y_test, y_scores)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred = get_predictions(best_prc_threshold, y_scores)
        y_roc_pred = get_predictions(best_roc_threshold, y_scores)  
    if "Ines" in testing:
        ################## Evaluation
        ## based on threshold with best fscore in precision-recall-curve
        accuracy_prc_valid = accuracy_score(y_valid, y_prc_pred_valid)
        precision_prc_valid = precision_score(y_valid, y_prc_pred_valid)
        recall_prc_valid = recall_score(y_valid, y_prc_pred_valid)
        f1_prc_valid = f1_score(y_valid, y_prc_pred_valid)
        gmean_prc_valid = geometric_mean_score(y_valid, y_prc_pred_valid)
        tn_prc_valid, fp_prc_valid, fn_prc_valid, tp_prc_valid = confusion_matrix(y_valid, y_prc_pred_valid).ravel()
        ## based on threshold with best gmean in fpr-tpr-curve (or roc-curve)
        accuracy_roc_valid = accuracy_score(y_valid, y_roc_pred_valid)
        precision_roc_valid = precision_score(y_valid, y_roc_pred_valid)
        recall_roc_valid = recall_score(y_valid, y_roc_pred_valid)
        f1_roc_valid = f1_score(y_valid, y_roc_pred_valid)
        gmean_roc_valid = geometric_mean_score(y_valid, y_roc_pred_valid)
        tn_roc_valid, fp_roc_valid, fn_roc_valid, tp_roc_valid = confusion_matrix(y_valid, y_roc_pred_valid).ravel()    
        auc_valid = roc_auc_score(y_valid, y_scores_valid)
        auprc_valid = average_precision_score(y_valid, y_scores_valid)
    ################## Evaluation/Testing
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
        roc_curve = roc_curve(y_test, y_scores, pos_label=1)
        precision_recall_curve = precision_recall_curve(y_test, y_scores, pos_label=1)
    auc = roc_auc_score(y_test, y_scores)
    auprc = average_precision_score(y_test, y_scores)
    results = {}
    results['y_train_1'] = y_train_1
    results['y_train_0'] = y_train_0
    if "Ines" in testing:
        results['y_valid_1'] = y_valid_1
        results['y_valid_0'] = y_valid_0
    results['y_test_1'] = y_test_1
    results['y_test_0'] = y_test_0
    if "Ines" in testing:
        results['accuracy_prc_valid'] = round(float(accuracy_prc_valid),5)
        results['precision_prc_valid'] = round(float(precision_prc_valid),5)
        results['recall_prc_valid'] = round(float(recall_prc_valid),5)
        results['f1_prc_valid'] = round(float(f1_prc_valid),5)
        results['gmean_prc_valid'] = round(float(gmean_prc_valid),5)
        results['accuracy_roc_valid'] = round(float(accuracy_roc_valid),5)
        results['precision_roc_valid'] = round(float(precision_roc_valid),5)
        results['recall_roc_valid'] = round(float(recall_roc_valid),5)
        results['f1_roc_valid'] = round(float(f1_roc_valid),5)
        results['gmean_roc_valid'] = round(float(gmean_roc_valid),5)
        results['tn_prc_valid'] = int(tn_prc_valid)
        results['fp_prc_valid'] = int(fp_prc_valid)
        results['fn_prc_valid'] = int(fn_prc_valid)
        results['tp_prc_valid'] = int(tp_prc_valid)
        results['tn_roc_valid'] = int(tn_roc_valid)
        results['fp_roc_valid'] = int(fp_roc_valid)
        results['fn_roc_valid'] = int(fn_roc_valid)
        results['tp_roc_valid'] = int(tp_roc_valid)
        results['auc_valid'] = round(float(auc_valid),5)
        results['auprc_valid'] = round(float(auprc_valid),5)
        if report == True:
            results['report_prc_valid'] = classification_report(y_valid, y_prc_pred_valid)
            results['report_roc_valid'] = classification_report(y_valid, y_roc_pred_valid) 
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
        results['report_prc'] = classification_report(y_test,y_prc_pred)
        results['report_roc'] = classification_report(y_test, y_roc_pred)
    print(str(results))
    # save results
    if save_results == True and repeat == False:
        d1={}
        d2={}
        d3={}
        d4={}
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
    return results


def get_combination_with_results(combination,all_keys, keys_wordembeddings, test_size=0.2):
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
    results = get_results(params_wordembeddings, params_classification, name=name, test_size=test_size) # returns dict of accuracy, precision, recall, auc, auprc
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


def get_best_combination_with_results(param_grid_wordembeddings, param_grid_classification, test_size=0.2):
    keys_wordembeddings = list(param_grid_wordembeddings.keys())
    values_wordembeddings = list(param_grid_wordembeddings.values())
    keys_classification = list(param_grid_classification.keys())
    values_classification = list(param_grid_classification.values())
    all_values = values_wordembeddings + values_classification
    all_keys = keys_wordembeddings + keys_classification
    all_combinations = list(itertools.product(*all_values))
    num_available_cores = len(os.sched_getaffinity(0))
    num_cores = num_available_cores - 10
    pool = multiprocessing.Pool(processes=num_cores)
    f=partial(get_combination_with_results, all_keys=all_keys, keys_wordembeddings=keys_wordembeddings, test_size=test_size) 
    print("D\n")
    _ = pool.imap_unordered(f, all_combinations) #returns list of [[params_wordembeddings_dict,params_classification_dict], results_dict]     
    '''
    list_of_combination_results = pool.map(f, all_combinations) #returns list of [[params_wordembeddings_dict,params_classification_dict], results_dict] 
    list_of_combination_results = [item for item in list_of_combination_results if item[1] is not None]     
    max_score_value = max(results[score] for combination,results in list_of_combination_results)
    max_comb_results = [[combination,results] for combination,results in list_of_combination_results if results[score] == max_score_value] # list of [[params_wordembeddings,params_classification], results] 
    print("Length of max_comb_results :" + str(len(max_comb_results)))
    best_results = max_comb_results[0][1].copy() 
    best_params_wordembeddings = max_comb_results[0][0][0].copy()
    best_params_classification = max_comb_results[0][0][1].copy()
    '''
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    best_comb_name = get_best_combination_name(results_object)
    best_params_wordembeddings = results_object[best_comb_name]["params_wordembeddings"]
    best_params_classification = results_object[best_comb_name]["params_classification"]
    best_results = results_object[best_comb_name]["results"]
    return best_params_wordembeddings, best_params_classification, best_results

def get_averaged_results(params_wordembeddings, params_classification,num_runs=5,train_indicies = None, test_indicies = None, test_size=0.2, report= False, curve= False, saveas='best'):
    betw_results = {}
    final_results = {}
    random_state = 10
    average_results = {}   # before multi
    for n in range(num_runs):
        if n < (num_runs-1):
            r = False
        elif (report == True) and (n == num_runs-1):
            r = True
        else: # (report == False) and (n == num_runs-1):
            r = False
        #if any(isinstance(el, list) for el in train_indicies) == True: # if list of lists
        if train_indicies is not None:
            tr_i = train_indicies[n]
            te_i = test_indicies[n]
        else:
            tr_i = train_indicies
            te_i = test_indicies
        results = get_results(params_wordembeddings, params_classification,name = n,train_indicies=tr_i,test_indicies=te_i, test_size = test_size,random_state = random_state+n ,report=r, curve=curve, repeat=True)
        average_results["results_" + str(n)] = results  # before 'best_results'
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object["average_"+ saveas +"_results"] = average_results  # before "multiple_best_results"
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
        for m in results:
            betw_results.setdefault(m,[]).append(results[m])
        #print("between results : " + str(betw_results))
    for m in results:
        a = betw_results[m]
        if 'report' in m:
            final_results[m] = results[m]
        elif 'threshold' in m:
            continue # we don't need the average thresholds
        elif 'curve' in m: # e.g. a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
            b = list(map(list, zip(*a))) # [[[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]]]
            c = [list(map(list, zip(*i))) for i in b]  # [[[1, 7], [2, 8]], [[3, 9], [4, 10]], [[5, 11], [6, 12]]]
            d = [[round(float(sum(subsubc)/len(subsubc)),5) for subsubc in subc] for subc in c] # subc = [[1, 7], [2, 8]], subsubc = [1, 7]
            final_results[m] = d
        else:
            final_results[m] = round(float(sum(a)/len(a)),5)
    return final_results

def lang_dependency_set(lang, test_size=0.2):
    dep_indicies = range(len(lang_indicies[lang]))
    k = len(lang_indicies[lang]) * test_size
    dep_test_indicies = random.sample(dep_indicies, int(k)) # each time when called --> different set 
    test_indicies = []
    train_dep_indicies = []    
    for i in range(len(lang_indicies[lang])):
        df_index = lang_indicies[lang][i]
        if i in dep_test_indicies:
            test_indicies.append(df_index)
        else:
            train_dep_indicies.append(df_index)
     # avoid problem:
     #  While using new_list = my_list, any modifications to new_list changes my_list everytime. 
     # --> use list.copy()
    train_indep_indicies = train_dep_indicies.copy()
    for l in lang_indicies:
        if l == lang:
            continue
        else:
            train_indep_indicies.extend(lang_indicies[l])   
    return train_indep_indicies, train_dep_indicies, test_indicies  

def compare_lang_dependency(lang,params_wordembeddings, params_classification, num_runs=5, test_size=0.2): 
    ### split train and test set indicies for 1st and 2nd set up
    train_indep_indicies_list = []
    train_dep_indicies_list = []
    test_indicies_list = []
    for i in range(num_runs):
        train_indep_indicies, train_dep_indicies, test_indicies = lang_dependency_set(lang = lang, test_size = test_size)
        train_indep_indicies_list.append(train_indep_indicies)
        train_dep_indicies_list.append(train_dep_indicies)
        test_indicies_list.append(test_indicies)
    # get results on 1st set up
    results_dep = get_averaged_results(params_wordembeddings,params_classification,num_runs=num_runs,train_indicies=train_dep_indicies_list, test_indicies=test_indicies_list, test_size=test_size, report=True, saveas=lang+"dep") 
    # get results on 2nd set up
    results_indep = get_averaged_results(params_wordembeddings,params_classification, num_runs=num_runs, train_indicies=train_indep_indicies_list, test_indicies=test_indicies_list, test_size=test_size, report= True, saveas=lang+"indep") 
    # compare results of 1st and 2nd set up
    print('Classification results for the ' + languages[lang] + ' language based on a language-DEPENDENT classifiction methodology :\n' + str(results_dep) + '\n\n')
    print('Classification results for the ' + languages[lang] + ' language based on a language-INDEPENDENT classifiction methodology :\n' + str(results_indep) + '\n\n')
    if results_dep[score] > results_indep[score]:
        dependency_result = 1  # dependent is better
        print('For the ' + languages[lang] + ' language, the language-DEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
    else:
        dependency_result = 0  # independent is better
        print('For the ' + languages[lang] + ' language, the language-INDEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
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
df_raw = pd.read_csv('../data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# load cleaned labeled data test set 
df_raw_test = pd.read_csv('../data/cleaned_labeled_projects_testset.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()
# Create a new dataframe
df_test = df_raw_test[['final_label', 'project_details','CPV','project_title']].copy()
# For testing the models
df2 = df.append(df_test, ignore_index=True)
# load language indicies
file = open('../data/lang_indicies.json','r',encoding='utf8')
lang_indicies = json.load(file)
file.close()

languages = {'de': 'German','fr':'French','it':'Italian','en':'English'}
trainedmodelfiles = {}

#### prepare data for fasttext text representation 
## make data compatible for fasttext 
## data.txt shape:
# __label__0 "some text..."  (pre-processed project details)
# __label__1 "some text..."
# __label__0 "some text..."

data_file = "../data/fasttext/data.txt"
pth = "../data/fasttext/"
if not os.path.isdir(pth):
    os.makedirs(pth)

with io.open(data_file,'w',encoding='utf8') as f:
    for i in range(0,len(df['final_label'])):
        f.write("__label__" + str(df['final_label'][i]) + " " + df['project_details'][i] + "\n")



######################################### ***** SET USER INPUT ARGUMENTS ***** ####################################################
# get and handle user input arguments 
parser = argparse.ArgumentParser()
parser.add_argument('runmode', type=str, choices= ['fold1','fold1results', 'fold2', 'fold2results', 'runmodel', 'fold2runmodel', 'testmodelJulia', 'testmodelInes'])
parser.add_argument('--dimU', type=int, default=100)
parser.add_argument('--minnU', type=int, default= 3)
parser.add_argument('--maxnU', type=int, default= 6)
parser.add_argument('--epochU', type=int, default=5)
parser.add_argument('--lrU', type=float, default= 0.05)
parser.add_argument('--epochS', type=int, default=5)
parser.add_argument('--lrS', type=float, default=0.1)
parser.add_argument('--wordNgramsS', type=int, default=1)
parser.add_argument('--metric', type=str, choices=['accuracy_prc','precision_prc', 'recall_prc', 'f1_prc', 'gmean_prc', 'accuracy_roc', 'precision_roc', 'recall_roc', 'f1_roc', 'gmean_roc', 'auc', 'auprc'], default='auprc', )
parser.add_argument('--reference', type=int, default= -1)
args = parser.parse_args()


score = args.metric
classifier = "fasttext"

# Set path
pth = "../results/model_" + classifier + "_" + score + "/" 

# Precise model reference
if args.reference == -1 and args.runmode == 'fold1': ## -1 is default
    ## create new results file
    results_path = adapt_resultspath(pth, pos=1)
elif args.reference == -1 and args.runmode != 'fold1': ## -1 is default
    ## refer to most recently created existing results file
    results_path = adapt_resultspath(pth, pos=0)
else:
    results_path = model_path(args.reference)
    if not os.path.exists(results_path):
        parser.error("You have chosen for the reference parameter a number (int) that refers to a non existent results file: " + results_path + "\n" + "Choose a for the reference parameter a number (int) that refers to an existent results file.\n")


################################################# ***** RUNNING PART ***** ###############################################

if args.runmode == 'fold1':   # fine-tuning procedure
    results_object={}
    results_object['classifier'] = 'fasttext'
    results_object['tune_param_wordembeddings'] = param_grid_wordembeddings
    results_object['tune_param_classification'] = param_grid_classification
    results_object['score'] = score
    file = open(results_path, "w",encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    # get best parameter values along with the results 
    '''
    training of 80% of all projects and
    evaluating of 20% of randomly chosen projects (independently of the language)
    '''
    #X_train, X_test, y_train, y_test, train_file = get_train_test_sets()
    best_params_wordembeddings, best_params_classification, best_results = get_best_combination_with_results(param_grid_wordembeddings, param_grid_classification, score)
    best_results_averaged = get_averaged_results(best_params_wordembeddings,best_params_classification) 
    # save the results
    best_combination = {}
    best_combination["best_params_wordembeddings"] = best_params_wordembeddings
    best_combination["best_params_classification"] = best_params_classification
    best_combination["best_results"] =  best_results
    best_combination["best_results_averaged"] = best_results_averaged
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close() 
    results_object["best_combination"] = best_combination
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    ################## OUTPUT the best results
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
    print("Averaged classification results: \n")
    for metric in best_results_averaged:
        result = best_results_averaged[metric]
        print(metric + " : " + str(result) + "\n")
elif args.runmode == 'fold1results': # return best fine-tuning step results
    try:
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
    except FileNotFoundError:
        print("Following path doesn't exist: " + results_path + ". Launch first 'fold1' for a while and then try again with 'fold1results'.")
    try:
        best_params_wordembeddings = results_object['best_combination']["params_wordembeddings"]
        best_params_classification = results_object['best_combination']["params_classification"]
        best_results_averaged = results_object["best_combination"]["best_results_averaged"]
    except KeyError:
        best_comb_name = get_best_combination_name(results_object)
        try: 
            if best_comb_name == "":
                raise NotuningError
        except NotuningError:
            print("There was no tuning process. Run with 'fold1' runmode for a while then try again with 'fold1results'.")
        best_params_wordembeddings = results_object[best_comb_name]["params_wordembeddings"]
        best_params_classification = results_object[best_comb_name]["params_classification"]
        best_results = results_object[best_comb_name]["results"]
    # run 5 times with best parameter values and take the average 
    best_results_averaged = get_averaged_results(best_params_wordembeddings,best_params_classification) 
    # save the results
    best_combination = {}
    best_combination["best_params_wordembeddings"] = best_params_wordembeddings
    best_combination["best_params_classification"] = best_params_classification
    best_combination["best_results"] =  best_results
    best_combination["best_results_averaged"] = best_results_averaged
    results_object["best_combination"] = best_combination
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    ################## OUTPUT the best results
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
    print("Averaged classification results: \n")
    for metric in best_results_averaged:
        result = best_results_averaged[metric]
        print(metric + " : " + str(result) + "\n")
elif args.runmode == 'fold2':
    try:
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
    except FileNotFoundError:
        print("Following path doesn't exist: " + results_path + ". Launch first 'fold1' for a while and then try again with 'fold2'.")
    try:
        best_params_wordembeddings = results_object['best_combination']["params_wordembeddings"]
        best_params_classification = results_object['best_combination']["params_classification"]
        best_results_averaged = results_object["best_combination"]["best_results_averaged"]    
    except KeyError:
        best_comb_name = get_best_combination_name(results_object)
        try: 
            if best_comb_name == "":
                raise NotuningError
        except NotuningError:
            print("There was no tuning process. Run with 'fold1' runmode for a while then try again with 'fold1results'.")
        best_params_wordembeddings = results_object[best_comb_name]["params_wordembeddings"]
        best_params_classification = results_object[best_comb_name]["params_classification"]
        best_results = results_object[best_comb_name]["results"]
    # run 5 times with best parameter values and take the average 
    best_results_averaged = get_averaged_results(best_params_wordembeddings,best_params_classification) 
    # save the results
    best_combination = {}
    best_combination["best_params_wordembeddings"] = best_params_wordembeddings
    best_combination["best_params_classification"] = best_params_classification
    best_combination["best_results"] =  best_results
    best_combination["best_results_averaged"] = best_results_averaged
    results_object["best_combination"] = best_combination
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    '''
    Compare 1st set up with 2nd set up
    - 1st set up: train on 80% german, evaluate on 20% german
    - 2nd set up: train on 100% italian, 100% french, 100% english, 80% german all together at once. evaluate on *the same* 20% german
    same for italian, french and english
    ''' 
    ########## test language (in)dependency and save the results
    for key in languages:
        dep_result = compare_lang_dependency(key, best_params_wordembeddings, best_params_classification) # returns 1 (dependent is better) or 0 (independent is better)   
        ## results already saved in 'compare_lang_dependency' function
        ## OUTPUT the results
        if dep_result == 1:
            print('For the ' + languages[key] + ' language, the language-DEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
        else: 
            print('For the ' + languages[key] + ' language, the language-INDEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
elif args.runmode == 'fold2results':
    try:
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
    except FileNotFoundError:
        print("Following path doesn't exist: " + results_path + ". Launch first 'fold2' and let the process reach the end.")
    for key in languages:
        try: 
            results_dep = results_object[key + "_dependency_result"][key + "_dependent"]
            results_indep = results_object[key + "_dependency_result"][key + "_independent"]
            dep_result = results_object[key + "_dependency_result"]["dependency_result"]
        except KeyError:
            print("The 'fold2' process for specific model with results path " + results_path + " did not reach the end. Launch with 'fold2' again and let the process reach the end.")
        print('Classification results for the ' + languages[key] + ' language based on a language-DEPENDENT classifiction methodology :\n' + str(results_dep) + '\n\n')
        print('Classification results for the ' + languages[key] + ' language based on a language-INDEPENDENT classifiction methodology :\n' + str(results_indep) + '\n\n')
        if dep_result == 1:
            print('For the ' + languages[key] + ' language, the language-DEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
        else: 
            print('For the ' + languages[key] + ' language, the language-INDEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
elif args.runmode == 'runmodel':
    if os.path.isfile(results_path):
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object["runmodel"] = {}
        results_object["runmodel"]["classifier"] = "fasttext"
        results_object["runmodel"]["score"] = score
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    else:
        results_object={}
        results_object["runmodel"] = {}
        results_object["runmodel"]["classifier"] = "fasttext"
        results_object["runmodel"]["score"] = score
        if not os.path.isdir(pth):
            os.makedirs(pth)
        file = open(results_path, "w",encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    # define user params wordembeddings
    user_params_wordembeddings = {}
    user_params_wordembeddings['model'] = 'skipgram'
    user_params_wordembeddings['loss'] = 'hs'
    user_params_wordembeddings['dim'] = args.dimU
    user_params_wordembeddings['minn'] = args.minnU
    user_params_wordembeddings['maxn'] = args.maxnU
    user_params_wordembeddings['epoch'] = args.epochU
    user_params_wordembeddings['lr'] = args.lrU
    # define user params classification
    user_params_classification = {} 
    user_params_classification['loss'] = 'hs'
    user_params_classification['epoch'] = args.epochS
    user_params_classification['lr'] = args.lrS
    user_params_classification['wordNgrams'] = args.wordNgramsS 
    # run 5 times with best parameter values and take the average 
    runmodel_results_averaged = get_averaged_results(user_params_wordembeddings,user_params_classification) 
    # save the results
    user_combination = {}
    user_combination["user_params_wordembeddings"] = user_params_wordembeddings
    user_combination["user_params_classification"] = user_params_classification
    user_combination["runmodel_results_averaged"] = runmodel_results_averaged
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object["runmodel"]["user_combination"] = user_combination 
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    ################## OUTPUT the user results
    print("User word embeddings parameter values according to " + score + ": \n")
    for param in user_params_wordembeddings:
        user_value = user_params_wordembeddings[param]
        print(param + " : " + str(user_value) + "\n")
    print("\n")
    print("User classification parameter values according to " + score + ": \n")
    for param in user_params_classification:
        user_value = user_params_classification[param]
        print(param + " : " + str(user_value) + "\n")
    print("\n")
    print("Averaged classification results: \n")
    for metric in runmodel_results_averaged:
        result = runmodel_results_averaged[metric]
        print(metric + " : " + str(result) + "\n")
elif args.runmode == 'fold2runmodel':
    if os.path.isfile(results_path):
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object["runmodel"] = {}
        results_object["runmodel"]["classifier"] = "fasttext"
        results_object["runmodel"]["score"] = score
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    else:
        results_object={}
        results_object["runmodel"] = {}
        results_object["runmodel"]["classifier"] = "fasttext"
        results_object["runmodel"]["score"] = score
        if not os.path.isdir(pth):
            os.makedirs(pth)
        file = open(results_path, "w",encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    # define user params wordembeddings
    user_params_wordembeddings = {}
    user_params_wordembeddings['model'] = 'skipgram'
    user_params_wordembeddings['loss'] = 'hs'
    user_params_wordembeddings['dim'] = args.dimU
    user_params_wordembeddings['minn'] = args.minnU
    user_params_wordembeddings['maxn'] = args.maxnU
    user_params_wordembeddings['epoch'] = args.epochU
    user_params_wordembeddings['lr'] = args.lrU
    # define user params classification
    user_params_classification = {} 
    user_params_classification['loss'] = 'hs'
    user_params_classification['epoch'] = args.epochS
    user_params_classification['lr'] = args.lrS
    user_params_classification['wordNgrams'] = args.wordNgramsS 
    # save the results
    user_combination = {}
    user_combination["user_params_wordembeddings"] = user_params_wordembeddings
    user_combination["user_params_classification"] = user_params_classification
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object["runmodel"]["user_combination"] = user_combination 
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
    '''
    Compare 1st set up with 2nd set up
    - 1st set up: train on 80% german, evaluate on 20% german
    - 2nd set up: train on 100% italian, 100% french, 100% english, 80% german all together at once. evaluate on *the same* 20% german
    same for italian, french and english
    ''' 
    ########## test language (in)dependency and save the results
    for key in languages:
        dep_result = compare_lang_dependency(key, user_params_wordembeddings, user_params_classification) # returns 1 (dependent is better) or 0 (independent is better)   
        ## results already saved in 'compare_lang_dependency' function
        ## OUTPUT the results
        if dep_result == 1:
            print('For the ' + languages[key] + ' language, the language-DEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
        else: 
            print('For the ' + languages[key] + ' language, the language-INDEPENDENT classification methodology achieves better classification results according to the ' + score + ' metric\n')
elif args.runmode == 'testmodelJulia':
    if os.path.isfile(results_path):
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object["testmodel"] = {}
        results_object["testmodel"]["classifier"] = classifier
        results_object["testmodel"]["score"] = score
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    else:
        results_object={}
        results_object["testmodel"] = {}
        results_object["testmodel"]["classifier"] = classifier
        results_object["testmodel"]["score"] = score
        if not os.path.isdir(pth):
            os.makedirs(pth)
        file = open(results_path, "w",encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    # define user params wordembeddings
    best_params_wordembeddings1 = {}
    best_params_wordembeddings1['model'] = 'skipgram'
    best_params_wordembeddings1['loss'] = 'hs'
    best_params_wordembeddings1['dim'] = 50
    best_params_wordembeddings1['minn'] = 2
    best_params_wordembeddings1['maxn'] = 6
    best_params_wordembeddings1['epoch'] = 2
    best_params_wordembeddings1['lr'] = 0.07
    # define user params classification
    best_params_classification1 = {} 
    best_params_classification1['loss'] = 'hs'
    best_params_classification1['epoch'] = 38
    best_params_classification1['lr'] = 0.09
    best_params_classification1['wordNgrams'] = 2    
    testmodel_results = get_results(best_params_wordembeddings1, best_params_classification1,testing="Julia")
    # save the results  
    best_combination1 = {}
    best_combination1["best_params_wordembeddings1"] = best_params_wordembeddings1
    best_combination1["best_params_classification1"] = best_params_classification1
    best_combination1["testmodel_results"] = testmodel_results       
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object["testmodel"]["best_combination1"] = best_combination1 
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
elif args.runmode == 'testmodelInes':
    # Ines' sample distribution
    trainInes_indicies= []   # 3988 positive and 3988 negative samples
    validInes_indicies = [] # 498 positive and 498 negative samples
    testInes_indicies = []  # 498 positive and 498 negative samples
    num_train_1 = 0   
    num_train_0 = 0    
    num_valid_1 = 0    
    num_valid_0 = 0     
    num_test_1 = 0    
    num_test_0 = 0     
    for i in range(0,df.shape[0]):
        if (df['final_label'][i] == 1) and num_train_1 < 3988:
            trainInes_indicies.append(i)
            num_train_1 = num_train_1 + 1
        elif (df['final_label'][i] == 0) and num_train_0 < 3988:
            trainInes_indicies.append(i)
            num_train_0 = num_train_0 + 1
        elif (df['final_label'][i] == 1) and num_valid_1 < 498:
            validInes_indicies.append(i)
            num_valid_1 = num_valid_1 + 1
        elif (df['final_label'][i] == 0) and num_valid_0 < 498:
            validInes_indicies.append(i)
            num_valid_0 = num_valid_0 + 1
        elif (df['final_label'][i] == 1) and num_test_1 < 498:
            testInes_indicies.append(i)
            num_test_1 = num_test_1 + 1
        elif (df['final_label'][i] == 0) and num_test_0 < 498:
            testInes_indicies.append(i)
            num_test_0 = num_test_0 + 1
    print("trainIneslen :" + str(len(trainInes_indicies)))
    print("validIneslen :" + str(len(validInes_indicies)))
    print("testIneslen :" + str(len(testInes_indicies)))
    if os.path.isfile(results_path):
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object[args.runmode] = {}
        results_object[args.runmode]["classifier"] = classifier
        results_object[args.runmode]["score"] = score
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    else:
        results_object={}
        results_object[args.runmode] = {}
        results_object[args.runmode]["classifier"] = classifier
        results_object[args.runmode]["score"] = score
        if not os.path.isdir(pth):
            os.makedirs(pth)
        file = open(results_path, "w",encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
    # define user params_representation (tfidf) and params_classification
    # add best hyperparameter values
    # define user params wordembeddings
    best_params_wordembeddings1 = {}
    best_params_wordembeddings1['model'] = 'skipgram'
    best_params_wordembeddings1['loss'] = 'hs'
    best_params_wordembeddings1['dim'] = 50
    best_params_wordembeddings1['minn'] = 2
    best_params_wordembeddings1['maxn'] = 6
    best_params_wordembeddings1['epoch'] = 2
    best_params_wordembeddings1['lr'] = 0.07
    # define user params classification
    best_params_classification1 = {} 
    best_params_classification1['loss'] = 'hs'
    best_params_classification1['epoch'] = 38
    best_params_classification1['lr'] = 0.09
    best_params_classification1['wordNgrams'] = 2    
    #validmodel_results = get_results(best_params_wordembeddings1, best_params_classification1, train_indicies=trainInes_indicies, test_indicies= validInes_indicies, testing="InesValid")
    testmodel_results = get_results(best_params_wordembeddings1, best_params_classification1, train_indicies=trainInes_indicies, test_indicies= testInes_indicies, valid_indicies=validInes_indicies, testing="Ines")
    # save the results  
    best_combination1 = {}
    best_combination1["best_params_wordembeddings1"] = best_params_wordembeddings1
    best_combination1["best_params_classification1"] = best_params_classification1
    #best_combination1["validmodel_results"] = validmodel_results  
    best_combination1[args.runmode + "_results"] = testmodel_results     
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    results_object[args.runmode]["best_combination1"] = best_combination1 
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()





 






'''
FastText:
"best_params_wordembeddings": {'dim':50, 'minn':2, 'maxn':5, 'epoch':2, 'lr':0.01},
"best_params_classification": {'epoch':49, 'lr':0.09, 'wordNgrams':1, 'dim':50},
"best_results": {... "auprc": 0.96487}}

comparing with best results from previous run/results_file:
"best_params_wordembeddings": {'dim':50, 'minn':2, 'maxn':6, 'epoch':2, 'lr':0.07},
"best_params_classification": {'epoch':38, 'lr':0.09, 'wordNgrams':2, 'dim':50},

test manually also (these combinations don't appear in results_file yet):
result_0 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':6, 'epoch':2, 'lr':0.07}, {'loss':'hs', 'epoch':38, 'lr':0.09, 'wordNgrams':2, 'dim':50})
result_1 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':6, 'epoch':2, 'lr':0.01}, {'loss':'hs','epoch':38, 'lr':0.09, 'wordNgrams':2, 'dim':50})
result_2 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':5, 'epoch':2, 'lr':0.07}, {'loss':'hs','epoch':38, 'lr':0.09, 'wordNgrams':2, 'dim':50})
result_3 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':6, 'epoch':2, 'lr':0.07}, {'loss':'hs','epoch':38, 'lr':0.09, 'wordNgrams':1, 'dim':50})
result_4 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':6, 'epoch':2, 'lr':0.01}, {'loss':'hs','epoch':38, 'lr':0.09, 'wordNgrams':1, 'dim':50})
result_5 = get_results({'model':'skipgram', 'loss':'hs','dim':50, 'minn':2, 'maxn':5, 'epoch':2, 'lr':0.07}, {'loss':'hs','epoch':38, 'lr':0.09, 'wordNgrams':1, 'dim':50})

Do they achieve better results?
'''

