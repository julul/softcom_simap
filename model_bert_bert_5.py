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

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
############### BERT tuning without process or thread parallelisation with pool


### Load data 

df = pd.read_csv('./input/merged_file_all.csv', sep='\t', encoding = 'utf-8')
df= df[['labels', 'project_details']].copy()
################## prepare data for text classification
# Validation Set approach : take 90% of the data as the training set and 10 % as the test set. X is a dataframe with  the input variable
# K fold cross-validation approach as well?
length_to_split = int(len(df) * 0.90)

X = df['project_details']
y = df['labels']

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

##################  Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.

train_data = [[a,b] for a,b in zip(X_train, y_train)]
train_df = pd.DataFrame(train_data)

eval_data = [[a,b] for a,b in zip(X_test, y_test)]
eval_df = pd.DataFrame(eval_data)


################# some definitions

def get_results(classification_model_args,args_combination, train_df, eval_df):
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


def get_combination_with_results(combination, combination_keys, classification_model_args,train_df, eval_df):
    print('B')
    args_combination = {}
    d1={}
    d3={}
    d4={}
    #name = multiprocessing.current_process().name
    for a, b in zip(combination_keys,combination):
        args_combination[a] = b
    results = get_results(classification_model_args,args_combination, train_df, eval_df) # returns dict of accuracy, precision, recall, auc, auprc 
    print('B2')
    d1['args_combination'] = args_combination
    d3['results'] = results
    with io.open(results_path,'r+',encoding='utf8') as file:
        results_object = json.load(file)
        number_of_combinations = len([key for key, value in results_object.items() if 'comb' in key.lower()])
        comb_nr = "comb_" + str(number_of_combinations+1)
        d4[comb_nr] = {}
        d4[comb_nr].update(d1)
        d4[comb_nr].update(d3)
        results_object.update(d4)
        file.seek(0)  # not sure if needed 
        json.dump(results_object, file)
    print(results)
    return [args_combination, results] 


def get_best_combination_with_results(classification_model_args, modelargs_tuning_grid, score, train_df, eval_df):    
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
        combination_result = get_combination_with_results(combination = combination, combination_keys = combination_keys, classification_model_args=classification_model_args, train_df=train_df, eval_df=eval_df)
        list_of_combination_results.append(combination_result)
    accuracy_scores = [results["accuracy"] for combination,results in list_of_combination_results]
    precision_scores = [results["precision"] for combination,results in list_of_combination_results]
    recall_scores = [results["recall"] for combination,results in list_of_combination_results]
    auc_scores = [results["auc"] for combination,results in list_of_combination_results] # auc scores
    auprc_scores = [results["auprc"] for combination,results in list_of_combination_results] # auprc scores 
    f1_scores = [results["f1"] for combination,results in list_of_combination_results] 
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
    best_combination = list_of_combination_results[index_max][0] 
    return best_combination, best_results  
    
def get_final_results(num_runs, classification_model_args, best_params, train_df, eval_df):
    metrics = ["accuracy","precision","recall","auc","auprc","f1"]
    final_results = dict.fromkeys(metrics, [])
    for n in range(num_runs):
        results = get_results(classification_model_args,best_params, train_df, eval_df)
        for metric in metrics:
            result = results[metric]
            final_results[metric].append(result)
    for metric in metrics:
        l = final_results[metric]
        final_results[metric] = sum(l)/len(l)
    return final_results
    

################## Create a ClassificationModel
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
# define tuning parameters and values


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

score = "auprc"  # choose "auprc","auc", "recall", "precision", "accuracy" or "f1", depending which score the evaluation of the best combination has to be based on

# save results
results_path = "./model_bert_bert/model_1/results.json"
results_object={}
results_object["tune_params"] = modelargs_tuning_grid
results_object["score"] = score
with io.open(results_path,'w+',encoding='utf8') as file: 
    json.dump(results_object, file) 
# returns {param1:value1, param2:value2, ...}, {"auprc":float1, "auc":float2, ... }
best_params, best_results = get_best_combination_with_results(classification_model_args=classification_model_args, modelargs_tuning_grid=modelargs_tuning_grid, score=score, train_df = train_df, eval_df = eval_df)


## check max_len --> length of longest tokenized sentence
## check adam properties tuning
## check https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
##      for setting max_len and speeding up training time with multiprocessing in conversion from example to feature
### change 'label' to 'final_label'

best_combination = {}
best_combination["best_params"] = best_params
num_runs = 5
final_results = get_final_results(num_runs, classification_model_args, best_params, train_df, eval_df) # apply best params and run num_runs times and take the average of the results as best result
best_combination["best_results"] = final_results
with io.open(results_path,'r+',encoding='utf8') as file:
    results_object = json.load(file)
    results_object["best_combination"] = best_combination
    file.seek(0)  # not sure if needed 
    json.dump(results_object, file)

print("Best parameter values according to " + score + ": \n")
for param in best_params:
    best_value = best_params[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")

print("Final evaluation results after running " + str(num_runs) + " times and taking the average: \n")
for metric in final_results:
    result = best_results[metric]
    print(metric + " : " + str(result) + "\n")



