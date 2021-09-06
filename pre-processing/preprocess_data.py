# https://stackoverflow.com/questions/28716241/controlling-the-threshold-in-logistic-regression-in-scikit-learn
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
import pandas as pd
import multiprocessing
from functools import partial
import preprocess_defs




########################### CLEANING STEP: Preprocess the project details #############################################
'''
We want project_details and project_title to be transformed into a single string of
meaningful words in low letters and without numbers in order to apply TF-IDF.
CPV is stays a number.
'''
################### sum up all meaningless words (stop words)
"""
languages = ["it","de","fr","en"]
stopwords_lists = dict.fromkeys(languages, [])
for l in languages:
    stopwords_path = "../data/stopwords/sw_" + l + "_cleaned.txt"
    f = open(stopwords_path,"r")
    text = f.read()
    words = ast.literal_eval(text) # convert string representation of list into list
    stopwords_lists[l] = words

meaningless_words = []
for _, sw in stopwords_lists.items():
    meaningless_words.extend(sw)

# drop duplicates
meaningless_words = list(dict.fromkeys(meaningless_words))


################### some definitions

def clean_text(text):
    # lower text
    text = text.lower()
    # unaccent the words to make it language independent
    text = unidecode.unidecode(text)
    # text should contain only alphabetic chars
    # any non-alphabetic chars like ',' or '\n' should be replaced by a space
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)
    # remove multiple spaces, words smaller than 3 letters, and meaningless words
    text = ' '.join([w for w in text.split() if len(w)>2 and w not in meaningless_words])
    return text


def preprocess_projects(partition_tuple, df):
    project_keys = list(df.keys())   # ['final_label', 'project_details','CPV','project_title']
    failed_extraction = []
    cleaned_projects = []
    start = partition_tuple[0]
    end = partition_tuple[1]
    for i in range(start,end):
        cleaned_project = dict.fromkeys(project_keys, []) # initialize as empty list
        cleaned_project['project_details'] = df['project_details'][i]
        # catch project whose extraction has failed
        if cleaned_project['project_details'] == '{}':
            failed_extraction.append(i)
            continue
        cleaned_project['final_label'] = df['final_label'][i] # 0 or 1       
        #cpv_raw = df['CPV'][i] # e.g. "['50000000', '70000000']"
        # convert string representation of list into list
        #cpv_list = ast.literal_eval(cpv_raw) # e.g ['50000000','70000000']
        #cpv = cpv_raw1[0] # e.g. '50000000'   
        #cleaned_project['CPV'] = cpv_list # e.g ['50000000','70000000']
        cleaned_project['CPV'] = df['CPV'][i] # e.g. "['50000000', '70000000']"
        cleaned_project['project_title'] = df['project_title'][i]
        # summarize those text (features) which need to be cleaned
        texts_keys = ['project_details', 'project_title']
        for t in range(len(texts_keys)):
            text = cleaned_project[texts_keys[t]]
            # assign cleaned text
            cleaned_project[texts_keys[t]] = clean_text(text)
        cleaned_projects.append(cleaned_project)
    return [cleaned_projects, failed_extraction]
"""

def get_partition_tuples(num_processes):
    partition = int(len(df)/num_processes)
    partitions = []
    for i in range(0,num_processes):
        partitions.append(i*partition)
    partitions.append(len(df))
    partition_tuples = []
    for i in range(0,len(partitions)-1):
        t = (partitions[i],partitions[i+1])
        partition_tuples.append(t)
    return partition_tuples
    
    


################### load unclean projects
# load labeled data
#df_raw = pd.read_csv('../data/labeled_projects.csv',sep='\t', encoding = 'utf-8')
df_raw = pd.read_csv('../data/labeled_projects_testset.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
df = df_raw[['final_label', 'project_details','CPV','project_title']].copy()



################### clean projects
# catch project whose extraction has failed
failed_extraction = []
# save each cleaned project into list
all_cleaned_projects = []

num_processes = 13

partition_tuples = get_partition_tuples(num_processes)

pool = multiprocessing.Pool(processes=num_processes)
f=partial(preprocess_defs.prepreprocess_projects, df=df) 

list_of_results = pool.map(f, partition_tuples) # returns list of [cleaned_projects, failed_extraction]
for c, f in list_of_results:
    all_cleaned_projects.extend(c) 
    failed_extraction.extend(f)

df_cleaned = pd.DataFrame(all_cleaned_projects) # columns are ['final_label', 'project_details','CPV','project_title']
df_failed = pd.DataFrame(failed_extraction)
################### save cleaned projects
df_cleaned.to_csv('../data/cleaned_labeled_projects_testset.csv', sep='\t', encoding = 'utf-8')
df_failed.to_csv('../data/failed_labeled_projects_testset.csv', sep='\t', encoding = 'utf-8')


#df1_raw = pd.read_csv('./data/cleaned_labeled_projects.csv',sep='\t', encoding = 'utf-8')
# Create a new dataframe
#df1 = df1_raw[['final_label', 'project_details','CPV','project_title']].copy()