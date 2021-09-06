import re
import ast
import unidecode


# For some reason multiprocessing.Pool does not always work with objects not defined in an imported module.
# https://bugs.python.org/issue25053
# https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror


########################### CLEANING STEP: Preprocess the project details #############################################
'''
We want project_details and project_title to be transformed into a single string of
meaningful words in low letters and without numbers in order to apply TF-IDF.
CPV is stays a number.
'''
################### sum up all meaningless words (stop words)

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