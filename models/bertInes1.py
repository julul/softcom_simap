from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import io
import json
import itertools
from sklearn import metrics
#from multiprocessing import cpu_count
import numpy as np
import os, os.path
from glob import iglob
from os import path
from sklearn.model_selection import train_test_split
import re
import random
from numpy import argmax
from numpy import sqrt
from sklearn.metrics import precision_recall_curve
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
from sklearn.utils.extmath import softmax
import argparse
import torch


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
######################### BERT tuning without process or thread parallelisation with pool
################# excpetions
class Error(Exception):
    # Base class for other exceptions    
    pass

class NotuningError(Error):
    # Raised when choosing 'fold1results' runmode even though there was no tuning process (with 'fold1') at all.
    pass

class ReferenceError(Error):
    # Raised when reference argument refers to a non existent results file 
    pass



################# some definitions
'''
#https://stackoverflow.com/questions/10239760/interrupt-pause-running-python-program-in-pdb/39478157#39478157
def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)
'''
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

def get_train_test_sets(train_indicies=None, test_indicies=None,test_size=0.1, random_state=0, testing= None):
    print("inside train test sets\n")
    if train_indicies is None and testing is None:
        X_train, X_test, y_train, y_test = train_test_split(df['project_details'], df['final_label'], test_size= test_size, random_state=random_state)
    elif train_indicies is not None or "Ines" in testing:
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
    elif testing == "Julia":
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
    return X_train, X_test, y_train, y_test, y_train.count(1), y_train.count(0), y_test.count(1), y_test.count(0)


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
    for s in prediction_scores:
        if s >= threshold:
            predictions.append(1)
        elif s < threshold:
            predictions.append(0)
    return predictions


def get_results(classification_model_args,args_combination, train_indicies = None, test_indicies = None, valid_indicies=None, test_size=0.1, random_state = 0, report = False, curve=False, save_results=True, repeat=False, testing=None):
    if repeat == False:
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        for _,res in results_object.items():
            if 'args_combination' not in res:
                continue
            if res['args_combination'] == args_combination and "Ines" not in testing:
                print("tuning parameters already exist in the current results_file\n")
                return res['results']
    print("before train test sets\n")
    if "Ines" in testing:
        _, X_valid, _, y_valid, _, _, y_valid_1, y_valid_0 = get_train_test_sets(train_indicies = train_indicies, test_indicies = valid_indicies,test_size= test_size, random_state=random_state, testing=testing)            
    X_train, X_test, y_train, y_test, y_train_1, y_train_0, y_test_1, y_test_0 = get_train_test_sets(train_indicies = train_indicies, test_indicies = test_indicies,test_size= test_size, random_state=random_state, testing=testing)    
    # Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
    # The first column is the text with type str, and the second column is the label with type int.
    train_df = pd.DataFrame([[a,b] for a,b in zip(X_train, y_train)])
    test_df = pd.DataFrame([[a,b] for a,b in zip(X_test, y_test)])
    if "Ines" in testing:
        valid_df = pd.DataFrame([[a,b] for a,b in zip(X_valid, y_valid)])
    # set some additional (non-tuning) args parameters
    args_combination['eval_batch_size'] = args_combination['train_batch_size']
    args_combination['overwrite_output_dir'] = True
    args_combination['reprocess_input_data'] = True # default True
    args_combination['use_cached_eval_features'] = False # default False
    '''
    Evaluation during training uses cached features.
    Setting this to False will cause features to be recomputed at every evaluation step.
    '''
    args_combination['cache_dir'] = "cache_dir_Ines1"  # default "cache_dir"
    ################## Train the model
    if "Ines" in testing:
        trainedmodeldir[classifier + "_Ines"] = "trained_" + classifier + '_model_Ines1/'
        if not os.path.isdir("trained_" + classifier + '_model_Ines1/'):
            os.makedirs("trained_" + classifier + '_model_Ines1/')
        args_combination['output_dir'] = "trained_" + classifier + '_model_Ines1/'
    elif "Julia" in testing:
        trainedmodeldir[classifier + "_Julia"] = "trained_" + classifier + '_model_Julia1/'
        if not os.path.isdir("trained_" + classifier + '_model_Julia1/'):
            os.makedirs("trained_" + classifier + '_model_Julia1/')
        args_combination['output_dir'] = "trained_" + classifier + '_model_Julia1/'
    else:
        trainedmodeldir[classifier] = "trained_" + classifier + '_model1/'
        if not os.path.isdir("trained_" + classifier + '_model1/'):
            os.makedirs("trained_" + classifier + '_model1/')
        args_combination['output_dir'] = "trained_" + classifier + '_model1/'  
    if "Ines" in testing:
        '''
        eval_model
            Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        '''
        args_combination['evaluate_during_training'] = True
        model = ClassificationModel(**classification_model_args, args = args_combination)
        # evaluate during training
        model.train_model(train_df, eval_df=valid_df)
        # test
        result, model_outputs ,_ = model.eval_model(eval_df= test_df)
        print(str(result) + "\n")
        print(str(model_outputs) + "\n")
        '''
        https://github.com/ThilinaRajapakse/simpletransformers/issues/30
        https://www.reddit.com/r/LanguageTechnology/comments/d8befe/understanding_bert_prediction_output_for/
        model_outputs:
        List of log probability predictions for all samples, with the predicted probability (by applying the sigmoid function) for each label.
        1st position negative and 2nd position positive.
        model_outputs = [[prob_for_0 prob_for_1],[prob_for_0 prob_for_1], ...]
        '''
        '''
        ## make model_outputs to scores
        probs = softmax(model_outputs.astype(np.double))
        y_scores_valid = probs[:,1] # 2nd element gives probability for positive class   
        # find optimal probability threshold in pr-curve and roc_curve
        # threshold same for validing as for testing
        best_prc_threshold, best_fscore = find_best_prc_threshold(y_valid, y_scores_valid)
        best_roc_threshold, best_gmean = find_best_roc_threshold(y_valid, y_scores_valid)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred_valid = get_predictions(best_prc_threshold, y_scores_valid)
        y_roc_pred_valid = get_predictions(best_roc_threshold, y_scores_valid)
        # test
        result, model_outputs ,_ = model.eval_model(eval_df= test_df)
        print(str(result) + "\n")
        print(str(model_outputs) + "\n")
        '''
        ## make model_outputs to scores
        probs = softmax(model_outputs.astype(np.double))
        y_scores = probs[:,1] # 2nd element gives probability for positive class  
        # find optimal probability threshold in pr-curve and roc_curve
        # threshold same for validing as for testing
        best_prc_threshold, best_fscore = find_best_prc_threshold(y_test, y_scores)
        best_roc_threshold, best_gmean = find_best_roc_threshold(y_test, y_scores)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred = get_predictions(best_prc_threshold, y_scores)
        y_roc_pred = get_predictions(best_roc_threshold, y_scores)
    else:
        model = ClassificationModel(**classification_model_args, args = args_combination)
        model.train_model(train_df)
        # test
        result, model_outputs ,_ = model.eval_model(eval_df= test_df)
        print(str(result) + "\n")
        print(str(model_outputs) + "\n")
        ## make model_outputs to scores
        probs = softmax(model_outputs.astype(np.double))
        y_scores = probs[:,1] # 2nd element gives probability for positive class   
        # find optimal probability threshold in pr-curve and roc_curve
        best_prc_threshold, best_fscore = find_best_prc_threshold(y_test, y_scores)
        best_roc_threshold, best_gmean = find_best_roc_threshold(y_test, y_scores)
        # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
        y_prc_pred = get_predictions(best_prc_threshold, y_scores)
        y_roc_pred = get_predictions(best_roc_threshold, y_scores)
    '''
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
        auc_valid = metrics.roc_auc_score(y_valid, y_scores_valid)
        auprc_valid = metrics.average_precision_score(y_valid, y_scores_valid)
    '''
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
        roc_curve = metrics.roc_curve(y_test, y_scores, pos_label=1) # fpr, tpr, thresholds
        precision_recall_curve = metrics.precision_recall_curve(y_test, y_scores, pos_label=1) # precision, recall, thresholds
        # create tpr-fnr-curve?
    auc = metrics.roc_auc_score(y_test, y_scores)
    # or auc = auc(fpr, tpr)
    auprc = metrics.average_precision_score(y_test, y_scores)
    # or auprc = auc(recall, precision)
    # create auc = auc(fnr,tpr)
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
    if save_results == True and repeat == False:
        d1={}
        d3={}
        d4={}
        d1['args_combination'] = args_combination
        d3['results'] = results
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
        comb_nr = "comb_" + str(number_of_combinations+1)
        d4[comb_nr] = {}
        d4[comb_nr].update(d1)
        d4[comb_nr].update(d3)
        results_object.update(d4)
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()    
    return results

def get_combination_with_results(combination, combination_keys, classification_model_args, test_size=0.1):
    print('B')
    args_combination = {}
    d1={}
    d3={}
    d4={}
    #name = multiprocessing.current_process().name
    for a, b in zip(combination_keys,combination):
        args_combination[a] = b
    results = get_results(classification_model_args,args_combination, test_size=test_size) # returns dict of accuracy, precision, recall, auc, auprc ...
    d1['args_combination'] = args_combination
    d3['results'] = results
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
    comb_nr = "comb_" + str(number_of_combinations+1)
    d4[comb_nr] = {}
    d4[comb_nr].update(d1)
    d4[comb_nr].update(d3)
    results_object.update(d4)
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()    
    print(results)
    return [args_combination, results] 


def get_best_combination_with_results(classification_model_args, modelargs_tuning_grid):    
    modelargs_tuning_values = list(modelargs_tuning_grid.values())
    combination_keys = list(modelargs_tuning_grid.keys())
    all_combinations = list(itertools.product(*modelargs_tuning_values)) 
    list_of_combination_results = []
    all_combinations_shuffled = random.sample(all_combinations, len(all_combinations))
    for combination in all_combinations_shuffled:
        combination_result = get_combination_with_results(combination = combination, combination_keys = combination_keys, classification_model_args=classification_model_args)
        list_of_combination_results.append(combination_result)
    '''
    max_score_value = max(results[score] for combination,results in list_of_combination_results)
    max_comb_results = [[combination,results] for combination,results in list_of_combination_results if results[score] == max_score_value] # list of [[params_representation_dict,params_classification_dict], results] 
    print("Length of max_comb_results :" + str(len(max_comb_results)))
    best_results = max_comb_results[0][1].copy() 
    best_combination = max_comb_results[0][0].copy()
    print(best_results)
    '''
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    best_comb_name = get_best_combination_name(results_object)
    best_combination = results_object[best_comb_name]["args_combination"]
    best_results = results_object[best_comb_name]["results"]
    return best_combination, best_results  
    

def get_averaged_results(classification_model_args, params, num_runs=5, train_indicies=None, test_indicies=None, test_size=0.1, report=False, curve=False, saveas= 'best'):
    betw_results = {}
    final_results = {}
    random_state = 10
    average_results = {} # before multiple_best_results
    for n in range(num_runs):
        if n < (num_runs-1):
            r = False
        elif (report == True) and (n == num_runs-1):
            r = True
        else: # (report == False) and (n == num_runs-1):
            r = False
        if train_indicies is not None: # if list of lists
            tr_i = train_indicies[n]
            te_i = test_indicies[n]
        else:
            tr_i = train_indicies
            te_i = test_indicies
        print("before get results\n")
        results = get_results(classification_model_args,params, train_indicies=tr_i,test_indicies= te_i, test_size=test_size,random_state = random_state+n ,report=r, curve=curve, repeat=True)
        average_results["results_" + str(n)] = results  # before 'best_results'
        file = open(results_path,'r',encoding='utf8')
        results_object = json.load(file)
        file.close()
        results_object["average_"+ saveas +"_results"] = average_results
        file = open(results_path,'w+',encoding='utf8')
        file.write(json.dumps(results_object))
        file.close()
        for m in results:
            betw_results.setdefault(m,[]).append(results[m])
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
    #print(str(final_results))
    return final_results  


def lang_dependency_set(lang, test_size=0.1):
    dep_indicies = range(len(lang_indicies[lang]))
    k = len(lang_indicies[lang]) * test_size
    dep_test_indicies = random.sample(dep_indicies, int(k))
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

def compare_lang_dependency(lang, params, num_runs=5,test_size=0.1):
    ### split train and test set indicies for 1st and 2nd set up    
    train_indep_indicies_list = []
    train_dep_indicies_list = []
    test_indicies_list = []
    for i in range(num_runs):
        train_indep_indicies, train_dep_indicies, test_indicies = lang_dependency_set(lang = lang, test_size = test_size)
        train_indep_indicies_list.append(train_indep_indicies)
        train_dep_indicies_list.append(train_dep_indicies)
        test_indicies_list.append(test_indicies)
    ### apply best params and run num_runs times to take the average of the results 
    # get results on 1st set up
    results_dep = get_averaged_results(classification_model_args, params,num_runs=num_runs, train_indicies=train_dep_indicies_list, test_indicies=test_indicies_list, test_size=test_size, report= True, saveas=lang+"dep")
    #results_dep = get_results(best_params_representation, best_params_classification,classifier,train_indicies=train_dep_indicies,test_indicies= test_indicies,test_size=test_size, report=True, curve=True, repeat=True)
    # get results on 2nd set up
    results_indep = get_averaged_results(classification_model_args, params,num_runs=num_runs, train_indicies=train_indep_indicies_list, test_indicies=test_indicies_list, test_size=test_size, report= True, saveas=lang+"indep")  
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


################## Create a ClassificationModel BERT
'''
For the purposes of fine-tuning, the authors recommend choosing from the following values (from Appendix A.3 of the BERT paper):

Batch size: 16, 32
Learning rate (Adam): 5e-5, 3e-5, 2e-5
Number of epochs: 2, 3, 4


class ClassificationModel:
    def __init__(
        self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):

        
        Initializes a ClassificationModel model.
        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a 
               Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to
               assign to each label for loss calculation.
 ---------> args (optional): Default args will be used if this parameter is not   
|              provided. If provided, it should be a dict containing the args that should
|              be changed in the default args.
|           use_cuda (optional): Use GPU if available. Setting to False will force
|              model to use CPU only.
|           cuda_device (optional): Specific GPU that should be used.
|               Will use the first available GPU by default.
|           **kwargs (optional): For providing proxies, force_download, resume_download, 
|              cache_dir and other options specific to the 'from_pretrained'
|              implementation where this will be supplied.
|
v
class ModelArgs: (default values for 'args')
    adam_epsilon: float = 1e-8  # tune adam_epsilon?
    learning_rate: float = 4e-5
    eval_batch_size: int = 8
    num_train_epochs: int = 1
    train_batch_size: int = 8
    max_seq_length: int = 128
    ...
    process_count: cpu_count() - 2 if cpu_count() > 2 else 1
        Number of cpu cores (processes) to use when converting examples to features.
        Default is (number of cores - 2) or 1 if (number of cores <= 2)
    use_multiprocessing: True
      (If True, multiprocessing will be used when converting data into features.
       Disabling can reduce memory usage, but may substantially slow down processing.)

'''
# define tuning parameters and values BERT
batch_sizes = [int(i) for i in [8,16,32]]
learning_rates = [5e-5, 4e-5, 3e-5, 2e-5]
epoch_numbers = [int(i) for i in [1,2,3,4]]
# For any BERT model, the maximum sequence length after tokenization is 512
max_seq_length = [512]  # default 128
#process_count= [cpu_count() - 2 if cpu_count() > 2 else 1] # default
process_count= [os.cpu_count() - 2 if os.cpu_count() > 2 else 1]
cuda_available = torch.cuda.is_available()
if cuda_available:
    use_multiprocessing = [False] # default True
else:
    use_multiprocessing = [True]

modelargs_tuning_grid = {}
modelargs_tuning_grid['learning_rate'] = learning_rates
modelargs_tuning_grid['train_batch_size'] = batch_sizes
modelargs_tuning_grid['num_train_epochs'] = epoch_numbers
modelargs_tuning_grid['max_seq_length'] = max_seq_length
modelargs_tuning_grid['process_count'] = process_count
modelargs_tuning_grid['use_multiprocessing'] = use_multiprocessing

model_type = 'distilbert'
model_name = 'distilbert-base-multilingual-cased'
#use_cuda = False # default True
classification_model_args = {}
classification_model_args['model_type'] = model_type
classification_model_args['model_name'] = model_name
classification_model_args['use_cuda'] = cuda_available

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

trainedmodeldir = {}

score = "auprc"
classifier = "bert"

pth = "../results/model_" + classifier + "_" + score + "/"
 ## refer to most recently created existing results file
results_path = adapt_resultspath(pth, pos=0)
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
    results_object["testmodelInes"] = {}
    results_object["testmodelInes"]["classifier"] = classifier
    results_object["testmodelInes"]["score"] = score
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
else:
    results_object={}
    results_object["testmodelInes"] = {}
    results_object["testmodelInes"]["classifier"] = classifier
    results_object["testmodelInes"]["score"] = score
    if not os.path.isdir(pth):
        os.makedirs(pth)
    file = open(results_path, "w",encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()
# add best hyperparameter values
best_params1 = {}
best_params1['train_batch_size'] = 32
best_params1['learning_rate'] = 0.00005
best_params1['num_train_epochs'] = 4
best_params1['max_seq_length'] = 512
testmodel_results = get_results(classification_model_args,best_params1, train_indicies=trainInes_indicies, test_indicies= testInes_indicies, valid_indicies=validInes_indicies, testing="Ines")

'''
args_combination= best_params1
train_indicies=trainInes_indicies
test_indicies= testInes_indicies
valid_indicies=validInes_indicies
testing="Ines"

report = False
curve=False
save_results=True
repeat=False
test_size=0.1
random_state = 0

#def get_results(classification_model_args,args_combination, train_indicies = None, test_indicies = None, valid_indicies=None, test_size=0.1, random_state = 0, report = False, curve=False, save_results=True, repeat=False, testing=None):
if repeat == False:
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    for _,res in results_object.items():
        if 'args_combination' not in res:
            continue
        if res['args_combination'] == args_combination and "Ines" not in testing:
            print("tuning parameters already exist in the current results_file\n")
            # return res['results']
print("before train test sets\n")
if "Ines" in testing:
    _, X_valid, _, y_valid, _, _, y_valid_1, y_valid_0 = get_train_test_sets(train_indicies = train_indicies, test_indicies = valid_indicies,test_size= test_size, random_state=random_state, testing=testing)            
X_train, X_test, y_train, y_test, y_train_1, y_train_0, y_test_1, y_test_0 = get_train_test_sets(train_indicies = train_indicies, test_indicies = test_indicies,test_size= test_size, random_state=random_state, testing=testing)    
# Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
# The first column is the text with type str, and the second column is the label with type int.
train_df = pd.DataFrame([[a,b] for a,b in zip(X_train, y_train)])
test_df = pd.DataFrame([[a,b] for a,b in zip(X_test, y_test)])
if "Ines" in testing:
    valid_df = pd.DataFrame([[a,b] for a,b in zip(X_valid, y_valid)])
# set some additional (non-tuning) args parameters
args_combination['eval_batch_size'] = args_combination['train_batch_size']
args_combination['overwrite_output_dir'] = True
args_combination['reprocess_input_data'] = True
################## Train the model
if "Ines" in testing:
    trainedmodeldir[classifier + "_Ines"] = "trained_" + classifier + '_model_Ines/'
    if not os.path.isdir("trained_" + classifier + '_model_Ines/'):
        os.makedirs("trained_" + classifier + '_model_Ines/')
    args_combination['output_dir'] = "trained_" + classifier + '_model_Ines/'
elif "Julia" in testing:
    trainedmodeldir[classifier + "_Julia"] = "trained_" + classifier + '_model_Julia/'
    if not os.path.isdir("trained_" + classifier + '_model_Julia/'):
        os.makedirs("trained_" + classifier + '_model_Julia/')
    args_combination['output_dir'] = "trained_" + classifier + '_model_Julia/'
else:
    trainedmodeldir[classifier] = "trained_" + classifier + '_model/'
    if not os.path.isdir("trained_" + classifier + '_model/'):
        os.makedirs("trained_" + classifier + '_model/')
    args_combination['output_dir'] = "trained_" + classifier + '_model/'  
model = ClassificationModel(**classification_model_args, args = args_combination)
model.train_model(train_df)
if "Ines" in testing:
    
    eval_model
        Returns:
        result: Dictionary containing evaluation results.
        model_outputs: List of model outputs for each row in eval_df
        wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
    
    # validate 
    result, model_outputs ,_ = model.eval_model(eval_df= valid_df)
    print(str(result) + "\n")
    print(str(model_outputs) + "\n")
    
    https://github.com/ThilinaRajapakse/simpletransformers/issues/30
    https://www.reddit.com/r/LanguageTechnology/comments/d8befe/understanding_bert_prediction_output_for/
    model_outputs:
    List of log probability predictions for all samples, with the predicted probability (by applying the sigmoid function) for each label.
    1st position negative and 2nd position positive.
    model_outputs = [[prob_for_0 prob_for_1],[prob_for_0 prob_for_1], ...]
    
    ## make model_outputs to scores
    probs = softmax(model_outputs.astype(np.double))
    y_scores_valid = probs[:,1] # 2nd element gives probability for positive class   
    # find optimal probability threshold in pr-curve and roc_curve
    # threshold same for validing as for testing
    best_prc_threshold, best_fscore = find_best_prc_threshold(y_valid, y_scores_valid)
    best_roc_threshold, best_gmean = find_best_roc_threshold(y_valid, y_scores_valid)
    # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
    y_prc_pred_valid = get_predictions(best_prc_threshold, y_scores_valid)
    y_roc_pred_valid = get_predictions(best_roc_threshold, y_scores_valid)
    # test
    result, model_outputs ,_ = model.eval_model(eval_df= test_df)
    print(str(result) + "\n")
    print(str(model_outputs) + "\n")
    ## make model_outputs to scores
    probs = softmax(model_outputs.astype(np.double))
    y_scores = probs[:,1] # 2nd element gives probability for positive class   
    # apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
    y_prc_pred = get_predictions(best_prc_threshold, y_scores)
    y_roc_pred = get_predictions(best_roc_threshold, y_scores)
else:
    # test
    result, model_outputs ,_ = model.eval_model(eval_df= test_df)
    print(str(result) + "\n")
    print(str(model_outputs) + "\n")
    ## make model_outputs to scores
    probs = softmax(model_outputs.astype(np.double))
    y_scores = probs[:,1] # 2nd element gives probability for positive class   
    # find optimal probability threshold in pr-curve and roc_curve
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
    auc_valid = metrics.roc_auc_score(y_valid, y_scores_valid)
    auprc_valid = metrics.average_precision_score(y_valid, y_scores_valid)
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
    roc_curve = metrics.roc_curve(y_test, y_scores, pos_label=1) # fpr, tpr, thresholds
    precision_recall_curve = metrics.precision_recall_curve(y_test, y_scores, pos_label=1) # precision, recall, thresholds
    # create tpr-fnr-curve?
auc = metrics.roc_auc_score(y_test, y_scores)
# or auc = auc(fpr, tpr)
auprc = metrics.average_precision_score(y_test, y_scores)
# or auprc = auc(recall, precision)
# create auc = auc(fnr,tpr)
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
if save_results == True and repeat == False:
    d1={}
    d3={}
    d4={}
    d1['args_combination'] = args_combination
    d3['results'] = results
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
    comb_nr = "comb_" + str(number_of_combinations+1)
    d4[comb_nr] = {}
    d4[comb_nr].update(d1)
    d4[comb_nr].update(d3)
    results_object.update(d4)
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()    
testmodel_results =  results


# save the results  
best_combination1 = {}
best_combination1["best_params1"] = best_params1
best_combination1["testmodelInes" + "_results"] = testmodel_results       
file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object["testmodel"]["best_combination1"] = best_combination1 
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()

for i in range(0,1000000000):
    print(i)

'''


def get_results(classification_model_args,args_combination, train_indicies = None, test_indicies = None, valid_indicies=None, test_size=0.1, random_state = 0, report = False, curve=False, save_results=True, repeat=False, testing=None):


test_size=0.1
random_state = 0
testing="Ines"

_, X_valid, _, y_valid, _, _, y_valid_1, y_valid_0 = get_train_test_sets(train_indicies = trainInes_indicies, test_indicies = validInes_indicies,test_size= test_size, random_state=random_state, testing=testing)            
X_train, X_test, y_train, y_test, y_train_1, y_train_0, y_test_1, y_test_0 = get_train_test_sets(train_indicies = trainInes_indicies, test_indicies = testInes_indicies,test_size= test_size, random_state=random_state, testing=testing)    
train_df = pd.DataFrame([[a,b] for a,b in zip(X_train, y_train)])
test_df = pd.DataFrame([[a,b] for a,b in zip(X_test, y_test)])
valid_df = pd.DataFrame([[a,b] for a,b in zip(X_valid, y_valid)])
args_combination['eval_batch_size'] = args_combination['train_batch_size']
args_combination['overwrite_output_dir'] = True
args_combination['reprocess_input_data'] = True # default True
args_combination['use_cached_eval_features'] = False # default False
args_combination['output_dir'] = "trained_" + classifier + '_model_Ines1/'
args_combination['evaluate_during_training'] = True
model = ClassificationModel(**classification_model_args, args = args_combination)
# test
result, model_outputs ,_ = model.eval_model(eval_df= test_df)
print(str(result) + "\n")
print(str(model_outputs) + "\n")
'''
https://github.com/ThilinaRajapakse/simpletransformers/issues/30
https://www.reddit.com/r/LanguageTechnology/comments/d8befe/understanding_bert_prediction_output_for/
model_outputs:
List of log probability predictions for all samples, with the predicted probability (by applying the sigmoid function) for each label.
1st position negative and 2nd position positive.
model_outputs = [[prob_for_0 prob_for_1],[prob_for_0 prob_for_1], ...]
'''
'''
## make model_outputs to scores
probs = softmax(model_outputs.astype(np.double))
y_scores_valid = probs[:,1] # 2nd element gives probability for positive class   
# find optimal probability threshold in pr-curve and roc_curve
# threshold same for validing as for testing
best_prc_threshold, best_fscore = find_best_prc_threshold(y_valid, y_scores_valid)
best_roc_threshold, best_gmean = find_best_roc_threshold(y_valid, y_scores_valid)
# apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
y_prc_pred_valid = get_predictions(best_prc_threshold, y_scores_valid)
y_roc_pred_valid = get_predictions(best_roc_threshold, y_scores_valid)
# test
result, model_outputs ,_ = model.eval_model(eval_df= test_df)
print(str(result) + "\n")
print(str(model_outputs) + "\n")
'''
## make model_outputs to scores
probs = softmax(model_outputs.astype(np.double))
y_scores = probs[:,1] # 2nd element gives probability for positive class  
# find optimal probability threshold in pr-curve and roc_curve
# threshold same for validing as for testing
best_prc_threshold, best_fscore = find_best_prc_threshold(y_test, y_scores)
best_roc_threshold, best_gmean = find_best_roc_threshold(y_test, y_scores)
# apply optimal pr-curve and roc_curve threshold to the prediction probability and get label predictions
y_prc_pred = get_predictions(best_prc_threshold, y_scores)
y_roc_pred = get_predictions(best_roc_threshold, y_scores)
################## Evaluation/Testing
# ## based on threshold with best fscore in precision-recall-curve
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
    roc_curve = metrics.roc_curve(y_test, y_scores, pos_label=1) # fpr, tpr, thresholds
    precision_recall_curve = metrics.precision_recall_curve(y_test, y_scores, pos_label=1) # precision, recall, thresholds
    # create tpr-fnr-curve?
auc = metrics.roc_auc_score(y_test, y_scores)
# or auc = auc(fpr, tpr)
auprc = metrics.average_precision_score(y_test, y_scores)
# or auprc = auc(recall, precision)
# create auc = auc(fnr,tpr)
results = {}
results['y_train_1'] = y_train_1
results['y_train_0'] = y_train_0
if "Ines" in testing:
    results['y_valid_1'] = y_valid_1
    results['y_valid_0'] = y_valid_0
results['y_test_1'] = y_test_1
results['y_test_0'] = y_test_0
'''
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
'''
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
if save_results == True and repeat == False:
    d1={}
    d3={}
    d4={}
    d1['args_combination'] = args_combination
    d3['results'] = results
    file = open(results_path,'r',encoding='utf8')
    results_object = json.load(file)
    file.close()
    number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
    comb_nr = "comb_" + str(number_of_combinations+1)
    d4[comb_nr] = {}
    d4[comb_nr].update(d1)
    d4[comb_nr].update(d3)
    results_object.update(d4)
    file = open(results_path,'w+',encoding='utf8')
    file.write(json.dumps(results_object))
    file.close()    


auprc = 0.982462091360261
auroc = 0.9824579442267061
eval_loss = 0.2402059633168392
fn = 26
fp = 35
mcc = 0.8776533760121471
tn = 463
tp = 472