from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import numpy as np
import json
import re
import ast
from nltk.tokenize import word_tokenize
from scipy.stats import randint
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from langdetect import detect
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import ast
import re
import decimal
import pickle
from functools import reduce
from operator import or_
import unidecode
import glob, os

os.chdir('./projects_ID_NO')
json_files = [file for file in glob.glob('*.json')]
path = './projects_ID_NO/'
files = [path + f for f in json_files]
# go back
os.chdir('..')


######### TODO:
# adapt cleaning step

mistakes = ['\n']
actuals = [' ']
rep = {}

# cleaning mistakes 
for i in range(0,len(mistakes)) :
    rep[mistakes[i]] = actuals[i]

# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in rep.items()) 
#Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
pattern = re.compile("|".join(rep.keys()))


punctuations = {'«','»','!','(',')','-','[',']','{','}',';',':','\'','\"','\\',',','<','>','.','/','?','@','#','$','%','^','&','*','_','~', "''", '``', "'", "’", ' '}
punctuations_str = '''«»!()-[]{};:''"\,<>./?@#$%^&*_~''``'’'''
translator = str.maketrans(punctuations_str, ' '*len(punctuations_str))


with open(files[0], 'r') as f:
    data=f.read()
    # parse file
    j_data = json.loads(data)

main_keys = list(j_data.keys())

def stand(d):
    s_project = {}
    for k, v in d.items():
        v_new = []
        k_new = ''
        if isinstance(v, dict):
            v_new = stand(v)
        else:
        #print("{0} : {1}".format(k, v))
            if isinstance(v, list):
                for text in v:
                    # remove unwanted substrings
                    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
                    # remove punctuation from text and replace by space
                    text = str(text.translate(translator))
                    # lower text
                    text = text.lower()
                    # unaccent the words to make it language independent
                    text = unidecode.unidecode(text)
                    # remove multiple spaces
                    text = re.sub(' +', ' ', text)
                    # split into tokens
                    tokens = text.split()
                    # add to new v
                    v_new.extend(tokens)
            else:
                text = str(v)
                # remove unwanted substrings
                text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
                # remove punctuation from text and replace by space
                text = str(text.translate(translator))
                # lower text
                text = text.lower()
                # unaccent the words to make it language independent
                text = unidecode.unidecode(text)
                # remove multiple spaces
                text = re.sub(' +', ' ', text)
                # split into tokens
                tokens = text.split()
                # add to new v
                v_new.extend(tokens)
        if k not in main_keys:
            k = str(k)
            # remove unwanted substrings
            k = pattern.sub(lambda m: rep[re.escape(m.group(0))], k)
            # remove punctuation from text and replace by space
            k = str(k.translate(translator))
            # lower text
            k = k.lower()
            # unaccent the words to make it language independent
            k = unidecode.unidecode(k)
            # remove multiple spaces
            k = re.sub(' +', ' ', k)
            # add to new value
            k_new = k
        else:
            k_new = k
        s_project[k_new] = v_new
    return s_project

def isindict_part(text, d):
    for _, v in d.items():
        if isinstance(v, dict):
            if isindict_part(text,v):
                return True
        else:
            if isinstance(v, list):
                if any(text in word for word in v):
                    return True
            else:
                v = str(v)
                if text in v:
                    return True
    return False

def isindict_word(text, d):
    for _, v in d.items():
        if isinstance(v, dict):
            if isindict_word(text,v):
                return True
        else:
            if isinstance(v, list):
                if any(text == word for word in v):
                    return True
            else:
                v = str(v)
                if text in v:
                    return True
    return False

def delegation_criteria(cpv_list,sec3):
    delegation_words = ['soumissionnaire', 'anbieter', 'special','spezial'] # nothing comparable detected in english or italian
    if any('extern' in word for word in sec3):
        indices = []
        indices = [i for i, s in enumerate(sec3) if 'extern' in s]
        for index in indices:
            window_size = 2
            neighbors = []
            # check if word containing 'extern' has at least 2 neighbors on its left and its right.
            if index >=window_size and len(sec3)>=(index+window_size): 
                for i in range(1,window_size+1):
                    neighbors.append(sec3[index+i])
                    neighbors.append(sec3[index-i])
                for w in delegation_words:
                    # check if the word containing 'extern' has neighbor words containing strings of delegation_words
                    if any(w in word for word in neighbors):
                        return True
    if '79000000' in cpv_list: # contains 'recrutement' (french) which refers to delegation
        return True
    return False


        
yes_level1 = ['centre de service informatique','csi dfjp','ocsin','scrum master']
yes_level2 = ['developpement','entwicklung','development','sviluppo','software','logiciel', 'techn','tecn','internet','java','automatisation','application','website']
                # and endswith('app'), endswith('web')

yes_level3 = ['informati', 'system','service','dienstleistung','dienst','servizio','computer']
yes_level4 = ['seco','upu','weltpostverein','afs','bar']

no_level1_words = ['iam', 'omada','erp']
no_level1_parts = ['vulnerab', 'infrastru', 'easygov']  # added easygov

### TODO: possibly edit the sections
section1 = ['nom officiel et adresse du pouvoir adjudicateur',
            'nome ufficiale e indirizzo del committente',
            'offizieller name und adresse des auftraggebers',
            'official name and address of the contracting authority']

section2 = ['vocabulaire commun des marches publics',
            'vocabolario comune per gli appalti pubblici',
            'common procurement vocabulary',
            'gemeinschaftsvokabular']

section3 = ['description detaillee des taches',
            'descrizione dettagliata del progetto',
            'detaillierter produktebeschrieb',
            'detaillierter aufgabenbeschrieb',
            'detailed task description']



'''
### TEST
test_file1 = './projects_ID_NO/ID_195499_NO_1102961.json'  # should be labeled yes1 (yes-yes project)
test_file2 = './projects_ID_NO/ID_195499_NO_1102947.json'  # should be labeled yes1 (yes-yes project)
files1 = [files[0], test_file1, test_file2]
files = files1
'''

#################  Label project as 'yes1','yes2','yes3','no2' or 'no1'
'''
yes1 = 'very positive',
yes3 = 'rather positive',
yes2= 'between yes1 and yes3',
no1 = 'very negative',
no2 = 'rather negative'
'''

labels = []
for file in files:
    with open(file, 'r') as f:
        data=f.read()
        # parse file
        p = json.loads(data)
    project = stand(p) # standardize the project
    if 'label' in p:
        del p['label']
    keys = list(project['project_details'].keys())
    sec1 = []
    sec2 = []
    sec3 = []
    for key in keys:
        if any(text in key for text in section1):
            sec1 = project['project_details'][key]
        if any(text in key for text in section2):
            sec2 = project['project_details'][key]
        if any(text in key for text in section3):
            sec3 = project['project_details'][key]
    title = project['project_title']
    cpv_list = project['CPV']# list of cpv codes
    for w in yes_level1:
        if isindict_part(w, project['project_details']):
            p['label'] = 'yes1'
            break
    if 'label' in p: # label == yes1
        if p['label'] == 'yes1': # to be sure that label == yes1
            with open(file, 'w') as outfile:
                json.dump(p, outfile) # save labeled project
            labels.append(p['label'])
            continue # go to next project
    for w in yes_level2:
        if any(w in word for word in sec1) or any(w in word for word in sec2) or any(w in word for word in sec3) or any(w in word for word in title):
            p['label'] = 'yes2'
            break
    if 'label' in p:  # label == yes2
        if p['label'] == 'yes2': # to be sure that label == yes2
            for w in yes_level4:
                if any(w == word for word in sec1): 
                    p['label'] = 'yes1'
                    break
            if delegation_criteria(cpv_list,sec3): # check the criteria for 'delegation'
                p['label'] = 'yes1'
        if p['label'] == 'yes1':
            with open(file, 'w') as outfile:
                json.dump(p, outfile)
            labels.append(p['label'])
            continue
        else:
            for w in no_level1_words:
                if isindict_word(w, project['project_details']):
                    p['label'] = 'no2'
                    break
            if p['label'] == 'no2':
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])
                continue
            else:
                for w in no_level1_parts:
                    if isindict_part(w, project['project_details']):
                        p['label'] = 'no2'
                        break
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])
                continue
    if any(word.endswith('app') for word in sec1) or any(word.endswith('app') for word in sec2) or any(word.endswith('app') for word in sec3) or any(word.endswith('app') for word in title) or any(word.endswith('web') for word in sec1) or any(word.endswith('web') for word in sec2) or any(word.endswith('web') for word in sec3) or any(word.endswith('web') for word in title):
        p['label'] = 'yes2'
    if 'label' in p:  # label = yes2
        if p['label'] == 'yes2':
            for w in yes_level4:
                if any(w == word for word in sec1):
                    p['label'] = 'yes1'
                    break
            if delegation_criteria(cpv_list,sec3): # check the criteria for 'delegation'
                p['label'] = 'yes1'
        if p['label'] == 'yes1':
            with open(file, 'w') as outfile:
                json.dump(p, outfile)
            labels.append(p['label'])            
            continue
        else: # label = yes2
            for w in no_level1_words:
                if isindict_word(w, project['project_details']):
                    p['label'] = 'no2'
                    break
            if p['label'] == 'no2':
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])            
                continue
            else:# label = yes2
                for w in no_level1_parts:
                    if isindict_part(w, project['project_details']):
                        p['label'] = 'no2'
                        break
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])
                continue
    for w in yes_level3:
        if any(w in word for word in sec1) or any(w in word for word in sec2) or any(w in word for word in sec3) or any(w in word for word in title):
            p['label'] = 'yes3'
            break
    if 'label' in p:  # label = yes3
        if p['label'] == 'yes3': 
            for w in yes_level4:
                if any(w == word for word in sec1):
                    p['label'] = 'yes1'
                    break
            if delegation_criteria(cpv_list,sec3): # check the criteria for 'delegation'
                p['label'] = 'yes1'
        if p['label'] == 'yes1':
            with open(file, 'w') as outfile:
                json.dump(p, outfile)
            labels.append(p['label'])
            continue
        else:# label = yes3
            for w in no_level1_words:
                if isindict_word(w, project['project_details']):
                    p['label'] = 'no2'
                    break
            if p['label'] == 'no2':
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])
                continue
            else:# label = yes3
                for w in no_level1_parts:
                    if isindict_part(w, project['project_details']):
                        p['label'] = 'no2'
                        break
                with open(file, 'w') as outfile:
                    json.dump(p, outfile)
                labels.append(p['label'])
                continue
    else:
        p['label'] = 'no1'
    with open(file, 'w') as outfile:
        json.dump(p, outfile)
    labels.append(p['label'])
    continue

   
'''
    num_yes1= 0
    num_yes2 = 0
    num_yes3 = 0
    num_no1 = 0
    num_no2 = 0

'''

### sum up the multiple labels ('label') into two different labels ('final_label') only
### and save the final label into single project file
### as well as save all the labeled projects together into a single csv file
labeled_projects= []
for file in files:
    # read file
    with open(file, 'r') as f:
        data=f.read()
        # parse file
        j_data = json.loads(data)
    labeled_project = j_data
    if labeled_project['label'] in ['yes1','yes2']:
        labeled_project['final_label'] = 1
    else: # labeled_project['label'] in ['no1','no2','yes3']
        labeled_project['final_label'] = 0
    # save final label into single project file 
    with open(file, 'w') as outfile:
        json.dump(labeled_project, outfile)
    labeled_projects.append(labeled_project)

# Convert all labeled projects into dataframe and save it into single csv file
df = pd.DataFrame(labeled_projects)
df.to_csv('./input/merged_file.csv', sep='\t', encoding = 'utf-8')







