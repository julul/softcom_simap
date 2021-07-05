# softcom_simap
Compare machine learning algorithms that filter projects from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

# usage

`python tfidf_models.py <runmode> <classifier> <metric>`

# arguments
-`<runmode>`: Run mode. Choose `'fold1'` for the hyperparameters fine-tuning step or choose `'fold2'` for the multilingual-related step.
Runmode `'fold1'` can take time; On unix shell press `Ctrl + z` to pause the process and press `fg` to continue the process. To stop the process press `Ctrl + c`, or `Ctrl + d` or `Ctrl + \`. The both modes' results of a specific model will be saved in the same file `'../results/model_TFIDF_<classifier>_<metric>'`. 
-`<classifier>`: Available text classifier. Choose among `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'`.
-`<metric>`: Available metric. Choose among `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'`. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 



