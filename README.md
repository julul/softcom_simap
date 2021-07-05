# softcom_simap
Compare machine learning algorithms that filter projects from simap.ch for Softcom Technologies SA.
With Bert, Fasttext, Logistic Regression, Linear SVC, Random Forest, and Multinomial Naive Bayes.

# usage

`python tfidf_models.py <> <classifier> <metric>`

# arguments
`<classifier>`: available text classifier. Choose among `'LogisticRegression'`, `'RandomForestClassifier'`, `'MultinomialNB'`, `'LinearSVC'`.
`<metric>`: available metric. Choose among `'accuracy_prc'`, `'precision_prc'`, `'recall_prc'`, `'f1_prc'`, `'gmean_prc'`, `'accuracy_roc'`, `'precision_roc'`, `'recall_roc'`, `'f1_roc'`, `'gmean_roc'`, `'auc'`, `'auprc'`. Metrics ending with `_prc` are based on the threshold with best f1-score in the pr-curve. Metrics ending with `_roc` are based on the threshold with the best gmean in roc-curve. 



