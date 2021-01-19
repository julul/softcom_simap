import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# initialize list of lists 
data_dedep = [['accuracy', 0.77214, 0.77183, 0.77273, 0.77228, 0.77214, 0.71005],
['precision', 0.89647,0.89744,0.89525,0.89634,0.89647,0.8558],
['recall', 0.87438,0.87391,0.875,0.87445,0.87438,0.82717],
['f1', 0.92312,0.92467,0.92131,0.92298,0.92312,0.89125],
['auc',0.9279,0.92794,0.92794,0.92794,0.92794,0.8971],
['auprc',0.77214,0.95255,0.31636,0.47498,0.6503,0.64317]] 

data_deindep = [['accuracy', 0.77156, 0.77156, 0.87062, 0.92131, 0.9311, 0.77156],['precision', 0.89829,0.89878,0.86924,0.92131,0.93056,0.94094],['recall', 0.87438,0.89769,0.8725,0.92131,0.93173,0.2897],['f1', 0.92312,0.89823,0.87087,0.92131,0.93114,0.443],['auc',0.9279,0.89829,0.87062,0.92131,0.9311,0.63576],['auprc',0.77214,0.85798,0.82216,0.88815,0.90116,0.62774]] 


# Create the pandas DataFrame 
df_dedep = pd.DataFrame(data_dedep, columns = ['Metric', 'MNB', 'RF', 'LSVC', 'FT', 'LG', 'DB']) 
df_deindep = pd.DataFrame(data_deindep, columns = ['Metric', 'MNB', 'RF', 'LSVC', 'FT', 'LG', 'DB'])
fig = plt.figure()

ax = df_dedep.plot(x="Metric", y=['MNB', 'RF', 'LSVC', 'FT', 'LG', 'DB'],kind="bar", alpha= 0.3)
df_deindep.plot(ax=ax, x="Metric", y=['MNB', 'RF', 'LSVC', 'FT', 'LG', 'DB'], kind="bar", alpha= 0.3)
'''
sns.histplot(df_dedep, x="Metric", y=['MNB', 'RF', 'LSVC', 'FT', 'LG', 'DB'], element="step")
sns.histplot(penguins, x="flipper_length_mm", hue="species", element="step")
concatenated = pd.concat([set1.assign(dataset='set1'), set2.assign(dataset='set2')])
sns.scatterplot(x='Std', y='ATR', data=concatenated,
                hue='Asset Subclass', style='dataset')
'''

plt.savefig('./plots/barplot.png')
plt.show()

