from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column in the label with type int.
train_data = [['Example sentence 1 for multilabel', [1, 1, 1, 1, 0, 1]]] + [['This thing is entirely different from the other thing. ', [0, 1, 1, 0, 0, 0]]]
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])


eval_data = [['Example sentence belonging to class 1', [1, 1, 1, 1, 1, 1]], ['This thing should be entirely different from the other thing. ', [0, 0, 0, 0, 0, 0]]]
eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

# Create a MultiLabelClassificationModel
use_cuda = False
model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=6, use_cuda=use_cuda, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5, 'silent': True})

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)

print('\nmodel_outputs from model.eval_model()')
print(model_outputs)

predictions, raw_outputs = model.predict(['Example sentence belonging to class 1', 'This thing should be entirely different from the other thing.'])

print('\npredictions from model.predict()')
print(predictions)

print('\nraw_outputs from model.predict()')
print(raw_outputs)
