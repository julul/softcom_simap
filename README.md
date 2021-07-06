# softcom_simap
Compare machine learning algorithms that filter projects from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

For each specific model we proceed in two folds: First, we fine-tune the hyperparameters of the specific model. Second, with the fine-tuned hyperparameters, we compute the classification results of the specific model based on two different language-related classification techniques. 

# usage

*`python tfidf_models.py <runmode> <classifier> <metric> <number>`

*`python fasttext_model.py <runmode> <metric> <number>`

*`python bert_models.py <runmode> <metric> <number>`

# parameters
* **`runmode`** (string, *mandatory*): Run mode. `'fold1'`, `'fold1results'`, `'fold2'`, `'fold2results'` are supported. Choose `'fold1'` for running the hyperparameters fine-tuning step and choose `'fold1results'` for returning the model's best tuning results. Choose `'fold2'` for running the multilingual-related step and choose `'fold2results'` for returning its result. Runmode `'fold1'` can take long; On unix shell press `Ctrl + z` to pause the process and press `fg` to continue the process. To stop the process press `Ctrl + c`, or `Ctrl + d` or `Ctrl + \`. The results of both modes for a specific model will be saved in the same file `'../results/model_<classifier>_<metric>/results_<number>'`. A specific model has a unique combination of `<classifier>` `<metric>` `<number>`. The model's best tuning results will be filtered and saved when running `'fold1results'` or when running `'fold2'`. The complete results of the multilingual step will be saved  when the `'fold2'` process is finished, therefore the `'fold2'` process should not be interrupted to avoid incomplete results. When launching `'fold1'` a new specific model is created by default, and when launching `'fold2'` by default the process refers to the most recently created model. The process of `fold2` is initialized with the fine-tuned hyperparameters from process `'fold1'` for a specific model; If however there are no tuning results for a specific model, either because `'fold1'` has not been runned at all or not long enough before, then the process of `fold2` uses default hyperparameter values for that model. Errors will occur when running `'fold1results'`, without running (or not long enough) before in `'fold1'` mode for a specific model. Errors will occur when launching `'fold2results'` without completing the process of `'fold2'` before for a specific model. 

* **`classifier`** (string, *mandatory*): Available text classifier. `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'` are supported.

* **`metric`** (string, *mandatory*): Available metric. `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'` are supported. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 

* **`number`** (int, *optional*, defaults to -1): Reference to a specific model respectively results file (e.g. `'../results/model_<classifier>_<metric>/results_<number>'`) when running with runmode `'fold2'`. Set by default to -1, meaning that the process `'fold2'` refers to the most recently created specific model. Choose a number that refers to an existing specific model (check in results directory). An error occurs if the specific model with that number doesn't exist.



