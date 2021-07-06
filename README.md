# softcom_simap
Compare machine learning algorithms that filter projects from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

For each specific model we proceed in two folds: First, we fine-tune the hyperparameters of the specific model. Second, with the fine-tuned hyperparameters, we compute the classification results of the specific model based on two different language-related classification techniques. 

# usage

`python tfidf_models.py <runmode> <classifier> <metric> <number>`

# parameters
* **`runmode`** (string, *mandatory*): Run mode. Choose `'fold1'` for running the hyperparameters fine-tuning step and choose `'fold1results'` for returning the model's best tuning results. Choose `'fold2'` for running the multilingual-related step and choose `'fold2results'` for returning its result. Runmode `'fold1'` can take long; On unix shell press `Ctrl + z` to pause the process and press `fg` to continue the process. To stop the process press `Ctrl + c`, or `Ctrl + d` or `Ctrl + \`. The results of both modes for a specific model will be saved in the same file `'../results/model_TFIDF_<classifier>_<metric>/results_<number>'`. A specific model has a unique combination of `<classifier>` `<metric>` `<number>`. The model's best tuning results will be filtered and saved when running `'fold1results'` or when running `'fold2'`. The results of the multilingual step will be saved at the end of the `'fold2'` process, therefore the process with this mode should not be interrupted to get and save the results. When launching `'fold1'` by default a new specific model is created, and when launching `fold2` by default the process refers to the most recently created model.  Errors will occur when running `'fold1results'`, without running (or not long enough) before in `'fold1'` mode for a specific model. Errors will occur when launching `'fold2results'` without completing the process of `'fold2'` before. For a specific model

* **`classifier`** (string, *mandatory*): Available text classifier. Choose among `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'`.

* **`metric`** (string, *mandatory*): Available metric. Choose among `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'`. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 
* **`number`** (int, *optional*, defaults to -1): 



