# Program to measure the similarity between  
# two sentences using cosine similarity. 
import nltk
nltk.download('all')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
# X = input("Enter first string: ").lower() 
# Y = input("Enter second string: ").lower() 
X ="I love horror movies"  # soumissionnaires externes
Y ="Lights out is a horror movie" # recrutement
  
# tokenization 
X_list = word_tokenize(X)  
Y_list = word_tokenize(Y) 
  
# sw contains the list of stopwords 
sw = stopwords.words('english')  
l1 =[];l2 =[] 
  
# remove stop words from the string 
X_set = {w for w in X_list if not w in sw}  # {'love','horror','movies','I'}
Y_set = {w for w in Y_list if not w in sw}  # {'Lights','horror','movie'}
 
# form a set containing keywords of both strings (without duplicates)
rvector = X_set.union(Y_set)  # {'movies','movie','I','love','Lights','horror'}
for w in rvector: 
    if w in X_set: l1.append(1) # create a vector 
    else: l1.append(0) 
    if w in Y_set: l2.append(1) 
    else: l2.append(0) 

c = 0  
# cosine formula  
for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
cosine = c / float((sum(l1)*sum(l2))**0.5) 
print("similarity: ", cosine) 
# similarity:  0.2886751345948129