# Benchmark for evaluating text classification models on multilingual textual documents
Compare machine learning algorithms that classify multilingual textual documents from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

For each specific model we proceed in two folds: First, we fine-tune the hyperparameters of the specific model. Second, with the fine-tuned hyperparameters, we compute the classification results of the specific model based on two different language-related classification techniques. 

## Execution
```
$ cd models
```

Then, either

```
$ python3 tfidf_models.py <runmode> <classifier> --metric --reference --hyperparameters
```
or

```
$ python3 fasttext_model.py <runmode> --metric --reference --hyperparameters
```
or
```
$ python3 bert_model.py <runmode> --metric --reference --hyperparameters
```

### Parameters
* **`<runmode>`** (string, *mandatory*): Run mode. `'fold1'`, `'fold1results'`, `'fold2'`, `'fold2results'` and `'runmodel'` are supported. Choose `'fold1'` for running the hyperparameters fine-tuning step and choose `'fold1results'` for returning the model's best tuning results. Choose `'fold2'` for running the multilingual-related step and choose `'fold2results'` for returning its result. Runmode `'fold1'` can take long; On unix shell press `Ctrl + z` to pause the `'fold1'` process and press `fg` to continue the same `'fold1'` process. To stop the `'fold1'` process press `Ctrl + c`, or `Ctrl + d` or `Ctrl + \`. The results of both folds, `'fold1'` and  `'fold2'`, for a specific model will be saved in the same file `'../results/model_<classifier>_<metric>/results_<number>'`. A specific model has a unique combination of `<classifier>` `<metric>` `<number>`. The model's best tuning results will be filtered and saved when finishing process `'fold1'`, when running `'fold1results'` or when running `'fold2'`. The complete results of the multilingual step will be saved  when the `'fold2'` process is finished, therefore the `'fold2'` process should not be interrupted to avoid incomplete results. When launching `'fold1'` a new specific model is created by default, and when launching `'fold2'` by default the process refers to the most recently created model. The process of `fold2` is initialized with the fine-tuned hyperparameters from process `'fold1'` for a specific model; If however there are no tuning results for a specific model, either because `'fold1'` has not been runned at all or not long enough before, then the process of `fold2` returns and error. Errors will also occur when running `'fold1results'`, without running (or not long enough) before in `'fold1'` mode for a specific model. And, errors will occur when launching `'fold2results'` without completing the process of `'fold2'` before for a specific model. The `'runmodel'` mode executes the model five times with the hyperparameters given by the `--hyperparameters` parameter and returns the average classification results.

* **`<classifier>`** (string, *mandatory*): Text classifier. `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'` are supported. Works only for executing the file `tfidf_models.py`. When executing the file `fasttext_model.py` we automatically run the FastText model and when executing the file `bert_model.py` we automatically run the Bert model

* **`--metric=<metric>`** (string, *optional*, defaults to `'auprc'`): Tuning metric. `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'` are supported. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 

* **`--reference=<number>`** (int, *optional*, defaults to -1): Reference to a specific model respectively results file (e.g. `'../results/model_<classifier>_<metric>/results_<number>'`). Useful for runmode `'fold1results'`, `'fold2'`, and `'fold2results'`. When launching `'fold1'`, a new specific model is created.  Set by default to -1, meaning that the process refers to the most recently created specific model. Choose a number that refers to an existing specific model (check in results directory). An error occurs if the specific model with that number doesn't exist.

* **`--hyperparameters=???`**


### Arguments


| runmode     | classifier             | --metric=metric  | --reference= number | --hyperparameters=      | 
| ----------  | ---------------------- |----------------- | ------------------- |-----------------------  |
| fold1       | LogisticRegression     | accuracy_prc     | -1 (default)        |                         |
| fold1results| RandomForestClassifier | precision_prc    |                     |                         |
| fold2       | MultinomialNB          | recall_prc       |                     |                         |
| fold2results| LinearSVC              | f1_prc           |                     |                         |
| runmodel    |                        | gmean_prc        |                     |                         |
|             |                        | accuracy_roc     |                     |                         |
|             |                        | precision_roc    |                     |                         |
|             |                        | recall_roc       |                     |                         |
|             |                        | f1_roc           |                     |                         |
|             |                        | gmean_roc        |                     |                         |
|             |                        | auc              |                     |                         |
|             |                        | auprc (default)  |                     |                         |

### Results

All results will be added to results folder as follows: `results/model_<classifier>_<metric>/results_<number>`.
At each new `fold1` execution, a new results file is created for a particular `<classifier>_<metric>` model combination.
The results file contains the classification results of the whole tuning procedure (fold1), i.e. the classification results of all metrics at each tuning step. The results file explicitly presents the best tuning results along with the corresponding hyperparameter values (after launching fold1results or fold2). The results file contains the classification results of all metrics of the whole fold2 procedure as well as the final "language-related" outcome (after launching fold2).



### Execution examples

Run first fold

```
$ cd models
$ python3 tfidf_models.py 'fold1' 'LogisticRegression'
```
After a while press  `Ctrl + c`, `Ctrl + d`or `Ctrl + \` to stop the process. You can also pause with `Ctrl + z` and then continue with `fg`.
Return the best tuning results of first fold along with the best hyperparameter values with following command.

```
$ python3 tfidf_models.py 'fold1results' 'LogisticRegression'
```
Run second fold with best hyperparameter values and let the process reach its end. At the end it will output the results.

```
$ python3 tfidf_models.py 'fold2' 'LogisticRegression'
```

Output the results of second fold.
```
$ python3 tfidf_models.py 'fold2results' 'LogisticRegression'
```

Same procedure with `fasttext_model.py` and `bert_model.py`.

```
$ python3 fasttext_model.py <runmode> 
```
```
$ python3 bert_model.py <runmode> 
```

Suppose we want to tune a model with another metric than with the default auprc metric
```
$ cd models
$ python3 tfidf_models.py 'fold1' 'LogisticRegression' --metric = 'auc'
```
Suppose we want to output the tuning results of a specific (existing) results file.
By default, the processes `fold1results`, `fold2` and `fold2results` point to the most current results file. The `fold1` creates a new results file.
Suppose the model `results/model_'LogisticRegression_auprc` contains `results_0`, `results_1`, `results_2` and we want to output the best tuning results of `results_1`. Then we would run:

```
$ python3 tfidf_models.py 'fold1results' 'LogisticRegression' --reference=1

```


(show best hyperparameter values)
(show table of classification results with best hyperparameter values)







