# https://stackoverflow.com/questions/28716241/controlling-the-threshold-in-logistic-regression-in-scikit-learn
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
import ast
import re
import unidecode



################### prepreocess the gathered lists of stopwords ##################

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
    # remove multiple spaces and single chars
    text = ' '.join([w for w in text.split() if len(w)>1])
    return text

################### clean all meaningless words
languages = ["it","de","fr","en"]
for l in languages:
    stopwords_path = "../data/stopwords/sw_" + l + ".txt"
    cleaned_stopwords_path = "../data/stopwords/sw_" + l + "_cleaned.txt"
    f = open(stopwords_path,"r")
    text = f.read()
    # convert string representation of list into list
    words = ast.literal_eval(text)
    # clean words
    cleaned_words = clean_text(' '.join(words)).split()
    # convert list back tp string representation of list
    text = str(cleaned_words)
    f = open(cleaned_stopwords_path, 'w')
    f.write(text)