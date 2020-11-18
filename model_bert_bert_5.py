# ref: https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04

from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn
import io
import json
import multiprocessing
from functools import partial
import itertools
import time
from random import randint
from multiprocessing.pool import ThreadPool
import numpy
import os, os.path
from glob import iglob
from os import path
from sklearn.model_selection import train_test_split
import re

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
######################### BERT tuning without process or thread parallelisation with pool

################# some definitions
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
    
    return X_train, X_test, y_train, y_test



def get_results(classification_model_args,args_combination, train_indicies = None, test_indicies = None, random_state = 0, report = False):
    X_train, X_test, y_train, y_test = get_train_test_sets(train_indicies, test_indicies,test_size= 0.2, random_state=random_state)    
    # Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
    # The first column is the text with type str, and the second column is the label with type int.
    train_df = pd.DataFrame([[a,b] for a,b in zip(X_train, y_train)])
    eval_df = pd.DataFrame([[a,b] for a,b in zip(X_test, y_test)])
    # set some additional (non-tuning) args parameters
    args_combination['eval_batch_size'] = args_combination['train_batch_size']
    args_combination['overwrite_output_dir'] = True
    model = ClassificationModel(**classification_model_args, args = args_combination) 
    ################## Train the model
    model.train_model(train_df)
    ################## Evaluate the model
    metrics = {}
    metrics["accuracy"] = sklearn.metrics.accuracy_score
    metrics["precision"] = sklearn.metrics.precision_score
    metrics["recall"] = sklearn.metrics.recall_score
    metrics["auc"] = sklearn.metrics.roc_auc_score
    metrics["auprc"] = sklearn.metrics.average_precision_score
    metrics["f1"] = sklearn.metrics.f1_score
    '''
    eval_model
            Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
    '''
    result, _ ,_ = model.eval_model(eval_df= eval_df, **metrics)
    for key,value in result.items():
        if isinstance(value,(float,numpy.float64)):
            result[key] = round(float(value),5) # convert to float in case it is of type numpy.float64 to be json compatible
        elif isinstance(value,numpy.int64):
            result[key] = int(value) # convert to int in case it is of type numpy.int64 to be json compatible
    return result


def get_combination_with_results(combination, combination_keys, classification_model_args):
    print('B')
    args_combination = {}
    d1={}
    d3={}
    d4={}
    #name = multiprocessing.current_process().name
    for a, b in zip(combination_keys,combination):
        args_combination[a] = b
    results = get_results(classification_model_args,args_combination) # returns dict of accuracy, precision, recall, auc, auprc 
    print('B2')
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


def get_best_combination_with_results(classification_model_args, modelargs_tuning_grid, score):    
    print('A')
    modelargs_tuning_values = list(modelargs_tuning_grid.values())
    combination_keys = list(modelargs_tuning_grid.keys())
    all_combinations = list(itertools.product(*modelargs_tuning_values)) 
    '''
    num_available_cores = len(os.sched_getaffinity(0))
    # pool = multiprocessing.Pool(processes=num_available_cores)   
    pool = ThreadPool(processes=num_available_cores)
    print('A1')
    f=partial(get_combination_with_results, combination_keys = combination_keys, classification_model_args=classification_model_args, train_df=train_df, eval_df=eval_df) 
    print('A2')
    list_of_combination_results = pool.map(f, all_combinations) 
    pool.close()
    pool.join()
    print('A3')
    '''
    list_of_combination_results = []
    for combination in all_combinations:
        combination_result = get_combination_with_results(combination = combination, combination_keys = combination_keys, classification_model_args=classification_model_args)
        list_of_combination_results.append(combination_result)
    max_score_value = max(results[score] for combination,results in list_of_combination_results)
    max_comb_results = [[combination,results] for combination,results in list_of_combination_results if results[score] == max_score_value] # list of [[params_representation_dict,params_classification_dict], results] 
    print("Length of max_comb_results :" + str(len(max_comb_results)))
    best_results = max_comb_results[0][1].copy() 
    best_combination = max_comb_results[0][0].copy()
    print(best_results)
    return best_combination, best_results  
    

def get_averaged_results(num_runs, classification_model_args, params, train_indicies=None, test_indicies=None):
    metrics = ["accuracy","precision","recall","auc","auprc","f1"]
    betw_results = {}
    final_results = {}
    random_state = 10
    for n in range(num_runs):
        if n == (num_runs-1):
            report = True
        else:
            report = False
        results = get_results(classification_model_args,params,random_state = random_state+n ,report=report)
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
    dep_test_indicies = random.sample(dep_indicies, int(k))
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

def compare_lang_dependency(test_size, lang):
    ### split train and test set indicies for 1st and 2nd set up
    train_indep_indicies, train_dep_indicies, test_indicies = lang_dependency_set(test_size = test_size, lang = lang)
    ### apply best params and run num_runs times to take the average of the results 
    # set number of runs
    num_runs = 5
    # get results on 1st set up
    results_dep = get_results(num_runs, classification_model_args, best_params, train_dep_indicies, test_indicies, report= True)
    # get results on 2nd set up
    results_indep = get_results(num_runs, classification_model_args, best_params, train_indep_indicies, test_indicies, report= True) 
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

        """
        Initializes a ClassificationModel model.
        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
     -----> args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"


class ModelArgs: (default values for 'args')
    adam_epsilon: float = 1e-8  # tune adam_epsilon?
    learning_rate: float = 4e-5
    eval_batch_size: int = 8
    num_train_epochs: int = 1
    train_batch_size: int = 8
    max_seq_length: int = 128
    ...
'''
# define tuning parameters and values BERT
batch_sizes = [8,16,32]
batch_sizes = [int(i) for i in batch_sizes]
learning_rates = [5e-5, 4e-5, 3e-5, 2e-5]
epoch_numbers = [1,2,3,4]
epoch_numbers = [int(i) for i in epoch_numbers]
modelargs_tuning_grid = {}
modelargs_tuning_grid['learning_rate'] = learning_rates
modelargs_tuning_grid['train_batch_size'] = batch_sizes
modelargs_tuning_grid['num_train_epochs'] = epoch_numbers

model_type = 'distilbert'
model_name = 'distilbert-base-multilingual-cased'
use_cuda = False
classification_model_args = {}
classification_model_args['model_type'] = model_type
classification_model_args['model_name'] = model_name
classification_model_args['use_cuda'] = use_cuda

################### load and prepare some DATA ########################################################################################
# load cleaned labeled data
df_raw = pd.read_csv('./data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()

# load language indicies
file = open('./data/lang_indicies.json','r',encoding='utf8')
lang_indicies = json.load(file)
file.close()

######################################### ***** ADAPT THIS PART ***** ####################################################
num = 0
pth = "./models/model_Bert/model_1/"
model_path = lambda num : pth + "results_" + str(num) + ".json" # adapt path accordingly
results_path = model_path(num)
score = "auprc"  # choose "auprc","auc", "recall", "precision", "accuracy" or "f1", depending which score the evaluation of the best combination has to be based on
################################################# ***** RUN THIS PART ***** ###############################################
# save results
results_object={}
results_object["tune_params"] = modelargs_tuning_grid
results_object["score"] = score
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
#X_train, X_test, y_train, y_test = get_train_test_sets()   
# Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
# The first column is the text with type str, and the second column is the label with type int.
#train_df = pd.DataFrame([[a,b] for a,b in zip(X_train, y_train)])
#eval_df = pd.DataFrame([[a,b] for a,b in zip(X_test, y_test)])

# RUN
best_params, best_results = get_best_combination_with_results(classification_model_args=classification_model_args, modelargs_tuning_grid=modelargs_tuning_grid, score=score)

## check max_len --> length of longest tokenized sentence
## check adam properties tuning
## check https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
##      for setting max_len and speeding up training time with multiprocessing in conversion from example to feature
### change 'label' to 'final_label'

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

best_params = results_object[best_comb_name]["args_combination"]
best_results = results_object[best_comb_name]["results"]

################## run 5 times with best parameter values and take the average 
num_runs = 5
averaged_results = get_averaged_results(num_runs, classification_model_args, best_params) # apply best params and run num_runs times and take the average of the results as best result

################## save best parameter values and the results 

best_combination = {}
best_combination["best_params"] = best_params
best_combination["best_results"] = best_results
best_combination["best_averaged_results"] = averaged_results

file = open(results_path,'r',encoding='utf8')
results_object = json.load(file)
file.close()
results_object["best_combination"] = best_combination
file = open(results_path,'w+',encoding='utf8')
file.write(json.dumps(results_object))
file.close()

################## OUTPUT the best parameter values and the results
print("Best parameter values according to " + score + ": \n")
for param in best_params:
    best_value = best_params[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")

print("Final evaluation results after running " + str(num_runs) + " times and taking the average: \n")
for metric in final_results:
    result = best_results[metric]
    print(metric + " : " + str(result) + "\n")



