# Benchmark for evaluating text classification models on multilingual textual documents
Compare machine learning algorithms that classify multilingual textual documents from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

For each specific model we proceed in two folds: First, we fine-tune the hyperparameters of the specific model. Second, with the fine-tuned hyperparameters, we compute the classification results of the specific model based on two different language-related classification techniques. 
Execute fold1 and fold2 or execute a model with specified hyperparameter values

## Fold1 and Fold2 Execution
```
$ cd models
```

Then, either

```
$ python3 tfidf_models.py <classifier> <runmode> --metric --reference 
```
or

```
$ python3 fasttext_model.py <runmode> --metric --reference 
```
or
```
$ python3 bert_model.py <runmode> --metric --reference
```

## Model Execution
```
$ cd models
```

Then, either for Logistic Regression

```
$ python3 tfidf_models.py LogisticRegression runmodel --min_df --max_df --penalty --C --solver --metric --reference
```
or for Random Forest
```
$ python3 tfidf_models.py RandomForestClassifier  runmodel --min_df --max_df --n_estimators --max_depth --min_samples_split --min_samples_leaf --metric --reference
```
or for Multinomial Naive Bayes
```
$ python3 tfidf_models.py MultinomialNB runmodel --min_df --max_df --alpha --fit_prior --metric --reference
```
or for Linear SVC
```
$ python3 tfidf_models.py LinearSVC runmodel --min_df --max_df --penalty --C --metric --reference
```

or for FastText

```
$ python3 fasttext_model.py  runmodel --dimU --minnU --maxnU --epochU --lrU --epochS --lrS --wordNgramsS --metric --reference
```
or for Bert
```
$ python3 bert_model.py  runmodel --train_batch_size --learning_rate --num_train_epochs --max_seq_length --metric --reference
```


### Fold1 and Fold2 Arguments


| classifier             |runmode       | --metric=metric  | --reference= number | 
| ---------------------- |------------- |----------------- | ------------------- |
| LogisticRegression     | fold1        | accuracy_prc     | -1 (default)        |                        
| RandomForestClassifier | fold1results | precision_prc    |                     |                         
| MultinomialNB          | fold2        | recall_prc       |                     |                         
| LinearSVC              | fold2results | f1_prc           |                     |                        
|                        |              | gmean_prc        |                     |                      
|                        |              | accuracy_roc     |                     |                        
|                        |              | precision_roc    |                     |           
|                        |              | recall_roc       |                     |           
|                        |              | f1_roc           |                     |           
|                        |              | gmean_roc        |                     |           
|                        |              | auc              |                     |            
|                        |              | auprc (default)  |                     |


### Model Arguments


| classifier             |runmode       | [model hyperparameters]| --metric=metric  | --reference= number | 
| ---------------------- |------------- | ---------------------- | ---------------- | ------------------- |
| LogisticRegression     |runmodel      |                        | accuracy_prc     | -1 (default)        |                        
| RandomForestClassifier |              |                        | precision_prc    |                     |                         
| MultinomialNB          |              |                        | recall_prc       |                     |                         
| LinearSVC              |              |                        | f1_prc           |                     |                        
|                        |              |                        | gmean_prc        |                     |                      
|                        |              |                        | accuracy_roc     |                     |                        
|                        |              |                        | precision_roc    |                     |           
|                        |              |                        | recall_roc       |                     |           
|                        |              |                        | f1_roc           |                     |           
|                        |              |                        | gmean_roc        |                     |           
|                        |              |                        | auc              |                     |            
|                        |              |                        | auprc (default)  |                     |





### Fold1/Fold2 and Model Parameters

* **`<classifier>`** (string, *mandatory* for tfidf_models.py): Text classifier. `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'` are supported. Works only for executing the file `tfidf_models.py`. When executing the file `fasttext_model.py` we automatically run the FastText model and when executing the file `bert_model.py` we automatically run the Bert model

* **`--metric=<metric>`** (string, *optional*, defaults to `'auprc'`): Tuning metric. `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'` are supported. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 

* **`--reference=<number>`** (int, *optional*, defaults to -1): Reference to a specific model respectively results file (e.g. `'../results/model_<classifier>_<metric>/results_<number>'`). Useful for runmode `'fold1results'`, `'fold2'`, and `'fold2results'`. When launching `'fold1'`, a new specific model is created.  Set by default to -1, meaning that the process refers to the most recently created specific model. Choose a number that refers to an existing specific model (check in results directory). An error occurs if the specific model with that number doesn't exist.

* **`<runmode>`** (string, *mandatory*): Run mode. `'fold1'`, `'fold1results'`, `'fold2'`, `'fold2results'` and `'runmodel'` are supported. Choose `'fold1'` for running the hyperparameters fine-tuning step and choose `'fold1results'` for returning the model's best tuning results. Choose `'fold2'` for running the multilingual-related step and choose `'fold2results'` for returning its result. Runmode `'fold1'` can take long; On unix shell press `Ctrl + z` to pause the `'fold1'` process and press `fg` to continue the same `'fold1'` process. To stop the `'fold1'` process press `Ctrl + c`, or `Ctrl + d` or `Ctrl + \`. The results of both folds, `'fold1'` and  `'fold2'`, for a specific model will be saved in the same file `'../results/model_<classifier>_<metric>/results_<number>'`. A specific model has a unique combination of `<classifier>` `<metric>` `<number>`. The model's best tuning results will be filtered and saved when finishing process `'fold1'`, when running `'fold1results'` or when running `'fold2'`. The complete results of the multilingual step will be saved  when the `'fold2'` process is finished, therefore the `'fold2'` process should not be interrupted to avoid incomplete results. When launching `'fold1'` a new specific model is created by default, and when launching `'fold2'` by default the process refers to the most recently created model. The process of `fold2` is initialized with the fine-tuned hyperparameters from process `'fold1'` for a specific model; If however there are no tuning results for a specific model, either because `'fold1'` has not been runned at all or not long enough before, then the process of `fold2` returns and error. Errors will also occur when running `'fold1results'`, without running (or not long enough) before in `'fold1'` mode for a specific model. And, errors will occur when launching `'fold2results'` without completing the process of `'fold2'` before for a specific model. The `'runmodel'` mode executes the model five times with the hyperparameters given by the `--hyperparameters` parameter and returns the average classification results.

* **`[model hyperparameters]`**: Selected tuning hyperparameters for:
  * TF-IDF:
    * **`--max_df`** (float or int, *optional*, defaults to 1.0)
    * **`--min_df`** (float or int, *optional*, defaults to 1)
    check https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html for more details
 
  * Logistic Regression:
    * **`--penalty`** (string, *optional*, defaults to 'l2')
    * **`--C`** (float, *optional*, defaults to 1.0)
    * **`--solver`** (string, *optional*, defaults to 'lbfgs')
    check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details
  * Random Forest:
    * **`--n_estimators`** (int, *optional*, defaults to 100)
    * **`--max_depth`** (int, *optional*, defaults to None)
    * **`--min_samples_split`** (int or float, *optional*, defaults to 2)
    * **`--min_samples_leaf`** (int or float, *optional*, defaults to 1)
    check https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    for more details
  * Multinomial Naive Bayes:
    * **`--alpha`** (float, *optional*, defaults to 1.0)
    * **`--fit_prior`** (bool, *optional*, defaults to True)
    check https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details
  * Linear SVC:
    * **`--penalty`** (string, *optional*, defaults to 'l2')
    * **`--C`** (float, *optional*, defaults to 1.0)
    check https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html for more details
  * FastText:
    * **`--dimU`** (int, *optional*, defaults to 100)
    * **`--minnU`** (int, *optional*, defaults to 3)
    * **`--maxnU`** (int, *optional*, defaults to 6)
    * **`--epochU`** (int, *optional*, defaults to 5)
    * **`--lrU`** (float, *optional*, defaults to 0.05)
    * **`--epochS`** (int, *optional*, defaults to 5)
    * **`--lrS`** (float, *optional*, defaults to 0.1)
    * **`--wordNgramsS`** (int, *optional*, defaults to 1)
    Parameters ending with U are for *unsupervised* training (i.e. for word embeddings) and those ending with S are for *supervised* training (i.e. for classification). Check https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters for more details 
  * Bert:
    * **`--train_batch_size`** (int, *optional*, defaults to 8)
    * **`--learning_rate`** (float, *optional*, defaults to 4e-5)
    * **`--num_train_epochs`** (int, *optional*, defaults to 1)
    * **`--max_seq_length`** (int, *optional*, defaults to 128)
    Check https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model for more details. Notably the section *Configuring a Simple Transformers ModelPermalink*.



### Fold1/Fold2 and Model Results

All results will be added to results folder as follows: `results/model_<classifier>_<metric>/results_<number>`.
At each new `fold1` execution, a new results file is created for a particular `<classifier>_<metric>` model combination.
The results file contains the classification results of the whole tuning procedure (fold1), i.e. the classification results of all metrics at each tuning step. The results file explicitly presents the best tuning results along with the corresponding hyperparameter values (after launching fold1results or fold2). The results file contains the classification results of all metrics of the whole fold2 procedure as well as the final "language-related" outcome (after launching fold2).



### Fold1/ Fold2 and Model Execution examples

Go first to working directory

```
$ cd models
```

Run first fold

```
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
$ python3 tfidf_models.py 'fold1' 'LogisticRegression' --metric = 'auc'
```
Suppose we want to output the tuning results of a specific (existing) results file.
By default, the processes `fold1results`, `fold2` and `fold2results` point to the most current results file. The `fold1` creates a new results file.
Suppose the model `results/model_'LogisticRegression_auprc` contains `results_0`, `results_1`, `results_2` and we want to output the best tuning results of `results_1`. Then we would run:

```
$ python3 tfidf_models.py 'fold1results' 'LogisticRegression' --reference=1

```
Run models with our fine-tuned hyperparameter values. Other hyperparameter values can achieve better classification results.
Run Logistic Resgression model with best achieved hyperparameter values:
```
$ python tfidf_models.py 'LogisticRegression' 'runmodel' --max_df=0.9 --min_df=0.001 --penalty='l1' --C=10 --solver='liblinear'

```

Run Linear SVC model with best achieved hyperparameter values:

```
$ python tfidf_models.py 'LinearSVC' 'runmodel' --max_df=0.8 --min_df=0.01 --penalty='l2' --C=0.1
```


Run Random Forest model with best achieved hyperparameter values:

```
$ python tfidf_models.py 'RandomForestClassifier' 'runmodel' --max_df=0.95 --min_df=0.001 --n_estimators=50 --max_depth=30 --min_samples_split=10 --min_samples_leaf=2
```

Run Multinomial Naive Bayes model with best achieved hyperparamater values:
```
$ python tfidf_models.py 'MultinomialNB' 'runmodel' --max_df=0.85 --min_df=0.001 --alpha=0.5 --fit_prior=False

```


Run FastText model with best achieved hyperparameter values:

```

python fasttext_model.py 'runmodel' --dimU=50 --minnU=2 --maxnU=6 --epochU=2 --lrU=0.07 --epochS=38 --lrS=0.09 --wordNgramsS=2
```

Run Bert model with best achieved hyperparameter values:
```
python bert_model.py 'runmodel' --train_batch_size=32 --learning_rate=4 --num_train_epochs=4 --max_seq_length=512


(show table of classification results with best hyperparameter values)







