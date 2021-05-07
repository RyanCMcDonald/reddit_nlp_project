#!/usr/bin/env python
# coding: utf-8

# # <span style="color:Purple">Project 3 :  Web APIs & NLP</span> <img src="../resources/reddit_logo.png" width="110" height="110" />
# ---
# ## <span style="color:Orange">Preprocessing - Modeling</span>      
# 
# #### Ryan McDonald
# ---

# ### Notebook Contents:
# 
# - [Reading the Data](#intro)
# - [Overview of Count Vectorizer](#overview)
# - [Modeling](#modeling)
#     - [Production Model](#prod)
#         - [Extended Analysis](#analysis)
#     - [Model #2](#2)
#     - [Model #3](#3)
#     - [Model #4](#4)
#     - [Model #5](#5)
#     - [Model #6](#6)
# 

# 
# **Imports**

# In[10]:


# Baseline Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Models!!!S
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import export_text, DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC

# Metrics!!!
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# GUI!!!
import tkinter as tk
from tkinter import simpledialog


# <a id='intro'></a>
# 
# ## 1. Read the Data

# In[11]:


#None-Tokenized DataFrame
subs_notoken =pd.read_csv('../datasets/title_data')
subs_notoken


# In[12]:


subs_notoken['title'][3999]


# In[13]:


# Sentence Tokenized DataFrame
subs =pd.read_csv('../datasets/tokenized_df')
subs


# <a id='overview'></a>
# 
# ## 2. Quick Overview of Count Vectorizing

# In[14]:


# Changing subreddit labels for modeling!

my_dict = {
    'camping':0,
    'VanLife':1
}

subs['subreddit']= subs['subreddit'].map(my_dict)
subs.head()


# In[15]:


X = subs['title']
y = subs['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)


# ### A quick Preview of a CVEC 'densified' DataFrame before running a PipeLine
# **Starting with CountVectorizer (combines several preprocessing steps in one**

# In[16]:


# maintaining defualts values stopwords=None, to start.
# also defaults by stripping punctuation

cvec= CountVectorizer(min_df = 2, ngram_range=(1,1), stop_words='english' )
# a words much appear more than ONCE to be counted
cvec.fit(X_train)


# In[17]:


# Transform the corpus (data)

X_train_cv = cvec.transform(X_train)
X_test_cv = cvec.transform(X_test)


# **'densifying' and building a DataFrame from the sparse matrix**

# In[18]:


pd.DataFrame(X_train_cv.todense()).head()


# In[19]:


# Replacing column names with the words they represent

X_train_df = pd.DataFrame(X_train_cv.todense(), columns = cvec.get_feature_names())
X_train_df.head()


# **An interesting result! Considering this is a global forum, I would assume there may be several languages represented within the data. Curious to see how to runs in the model.  May need to conduct additional filtering, TBD**

# In[21]:


# lets investigate whether or not in remove 'stop words'

X_train_df = pd.DataFrame(X_train_cv.todense(),columns= cvec.get_feature_names())

X_train_df.sum().sort_values(ascending= False).head(15).plot(kind = 'barh')
plt.xlabel('Occurance (count)')  
plt.title("Top 15 Words in Training Data (w/o Stop Words)")

plt.rcParams.update({'font.size': 16})
plt.show()


# #### A reminder of the Baseline score...

# In[22]:


y.value_counts(normalize= True)


# <a id='modeling'></a>
# 
# ## 3. Modeling
# 
# <a id='prod'></a>
# ## Model #1 (Production Model) - CVEC/BNB
# 
# **Production model performed 1% better on testing data with the non-sent-tokenized data!**
# 

# **Basline Model Accuracy for non-Tokenized DataFrame:**

# In[23]:


subs_notoken['subreddit'].value_counts(normalize=True)


# In[24]:


# Will begin with a pipeline - CVEC transformer with BernoulliNB estimator
# Production model performed 1% better on testing data with the non-sent-tokenized data!

X = subs_notoken['title']
y = subs_notoken['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)


pipe= Pipeline([
    ('tf', TfidfVectorizer()),
    ('bnb', BernoulliNB())])
    


# In[25]:


# GridSearch through out hyperparameters!
# setting up parameter dictionary:

pipe_params= {'tf__stop_words':['english', None], # to use the 'stop_words' dict, or not!    
              'tf__ngram_range':[(1, 2), (2,2)],  # unigrams, and bigrams
              'tf__analyzer':['word'],            # defualt value to make 'grams' a 'word'
              'tf__min_df':[0, 5, 10]             # ignore words with less document frequency
}


# In[26]:


# Instatiating GridSearchCV

gs= GridSearchCV(pipe,
                param_grid=pipe_params,
                cv=8,                    # 8 fold cross validation
                verbose = 1)
gs.fit(X_train, y_train)


# **Pickling Model For Future Needs, if neccessary**

# In[27]:


with open('../pickles/prod_model.pkl', mode ='wb') as pickle_out:
    pickle.dump(pipe, pickle_out)


# <a id='analysis'></a>
# ### Production Model Analysis 

# In[28]:


gs.best_params_


# **The model, surprisingly decided to move forward with min_df= 0.  I was surprised we attained the results we did when including every word, regardless of term frequency.**

# In[29]:


# Score on Training and Testing Data

print(f'Training Accuracy Score is: {gs.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {gs.score(X_test, y_test)}')


# **Our model succesfully predicted 87.6% of the classes correctly!  Excellent!**
# 
# **Pros:**  
# 
# TFIDF worked very well in this model.  Similar to CountVectorizer (used in several other model iterations, below), TFIDF does a great job developing a count matrix from the text data in our titles.  This made it very straight forward to model.  Then within the TFIDF funcitonality is the IDF transformer.  This brings a 'weight' parameter into the information. The 'inverse document frequency' takes away the 'weight' of words that may occur many many times through all of the corpus and applies more weight to words occuring less frequently.  Once the BernoulliNB classifier was deployed onto the TFIDF transformed data, BNB quickly catergorized the binary features into the appropriate class and produced a great accuracy score.
# 
# **Downsides:**  
# 
# The model is pretty overfit!  This model wouldn't necessarily work in other types of analysis, but for this problem statement, we are more interested in the testing accuracy score.  Since this model produced the highest testing accuracy, we are going into production! We could infer that this model would perform well on other subreddit data, but perhaps not on other types of classification information, say weather statistics, or traffic light patterns. 
# 
# **This is just part of the answer.  Accuracy looks great (best of our models).  Let's take a look at the misclassification rate, via the Confusion Matrix!** 

# In[30]:


preds = gs.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
plot_confusion_matrix(gs, X_test, y_test, cmap='Blues');

plt.rcParams.update({'font.size': 14})
plt.title('     Confusion Matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[31]:


# Misclassification Rate
print(f'Misclassification Rate is: {((fp+fn)/ (tp + fp + tn + fn))}')


# **The model incorrectly predicted 12.4% of the observations**

# In[32]:


# Producing a dataframe with our PREDICTIONS to see which were misclassified!

preds = gs.predict(X_test)
predictions_dict = {
    # original, not cv - will be more readable
    'original text': X_test,
    # true label
    'actual': y_test,
    # predicted label
    'predictions': preds
}
# cast to df
pd.set_option("display.max_colwidth", None)
predictions_df = pd.DataFrame(predictions_dict)
predictions_df=predictions_df[predictions_df['actual'] != predictions_df['predictions']]
predictions_df.shape


# ### Which Titles Were Misclassified?

# In[33]:


predictions_df.head(10)
# Hard to tell by reading the titles why they were not classified correctly!


# Unfortunately, no clear deciding factor could be determined by reviewing our misclassified data. However, r/camping titles were predicted incorrectly 17% more often than Camping titles. These titles ranged form foreign languages, Emoticon-filled sentences, and random, more non-descript titles.  Further analysis may reveal more insights. 
# 

# In[35]:


predictions_df['actual'].value_counts(normalize=True)


# **For this model, with a .50 baseline score, the posts could have gone either way!.  But, seeing a bias towards Camping posts being misclassified was interesting. However, with the word 'camping' appearing more than any other (without stopwords) it makes sense.**

# ### User GUI For Testing Titles!
# **When the cell is ran, enter your title, and the appropriate subreddit to post in will display below the cell!**

# In[36]:


pipe3 = Pipeline([
    ('tf', TfidfVectorizer(stop_words='english',ngram_range=(1, 2),analyzer= 'word',min_df=0)),
    ('bnb', BernoulliNB())])

pipe3.fit(X_train, y_train)



import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Which SubReddit",
                                  prompt="What is your title?:")


# check it out
pipe3.predict([USER_INP])


# ### Which words were most important to the model?
# 
# **Based on the inverse document frequency of the word**
# 
# **Top IDF words were ran through the predicter GIU above to develop the 'predicted class' column**

# In[37]:


list =['VanLife','camping','VanLife','VanLife','VanLife','VanLife','VanLife','VanLife','VanLife','camping']
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1, 2), analyzer ='word', min_df=5)
 
tfidf.fit(X_train)

tfidf_dict = {
    'words' :tfidf.get_feature_names(),
    'idf':tfidf.idf_  
}
tfidif_dict = pd.DataFrame(tfidf_dict)
top_10_idf=tfidif_dict.sort_values(by='idf', ascending = False).head(10)
top_10_idf=top_10_idf.reset_index(drop= True)
top_10_idf['predicted class']= list
top_10_idf


# **80% of the top ten most 'important' words/phrases as determined by IDF are predicted to be associated with VanLife.  With these statistics, I would have expected the misclassified words to weigh heavier on the VanLife class.**

# ---

# <a id='2'></a>
# ## Additional Models have limited interpretations, however, some interesting finding are discussed below applicable models
# 
# ## Model #2 - CVEC/RFC
# 
# #### <span style="color:Red">**Warning!! This Model Takes over an HOUR to run!**</span>     
# 
# **Model is pickled below incase additional metrics are desired!**

# In[103]:


# # Will begin with a pipeline - CVEC transformer with Random Forest Classifier

# pipe= Pipeline([
#     ('cvec', CountVectorizer()),
#     ('rf', RandomForestClassifier()) # internally bagging! Bootstrap = True
# ])


# In[104]:


# # GridSearch through out hyperparameters!
# # setting up parameter dictionary:

# pipe_params= {'cvec__stop_words':['english', None],       # with and without stopwords
#     'cvec__max_features':[None, 2000, 3000, 4000, 5000],  # number of highest frequency features to use
#     'cvec__min_df':[2, 3],                                # ignore words at these low frequencies
#     'cvec__ngram_range':[(1, 1), (1, 2)],                 # unigrams and bigrams
#     'rf__n_estimators': [100, 200, 250],                  # # of trees in forest
#     'rf__max_depth':[None, 1, 2, 3, 4, 5, 10],            # max depth of 'tree'
#     'rf__criterion':['gini', 'entropy']                   # quality of split
              
# }


# In[106]:


# # Instatiating GridSearchCV

# gs= GridSearchCV(pipe,
#                 param_grid=pipe_params,
#                 cv=5,                    # 5 fold cross validation
#                 verbose = 1)
# gs.fit(X_train, y_train)


# In[107]:


# # the model# 2 above took 81 minutes to run!  Pickling it in case I need to come back to it later!

# with open('../pickles/model_2.pkl', mode ='wb') as pickle_out:
#     pickle.dump(pipe, pickle_out)


# **Below is the pickle open code if needed!**

# In[108]:


# with open('../pickles/model_2.pkl', mode= 'rb') as pickle_in:
#     pipe = pickle.load(pickle_in)    


# In[109]:


# # Best paramaters from our GridSearch!
# gs.best_params_


# In[110]:


# print(f'Training Accuracy Score is: {gs.score(X_train, y_train)}')
# print(f'Testing Accuracy Score is: {gs.score(X_test, y_test)}')


# **Training Accuracy Score is: 0.994**
# **Testing Accuracy Score is: 0.829**

# In[111]:


# preds = gs.predict(X_test)
# tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
# plot_confusion_matrix(gs, X_test, y_test, cmap='Greens');


# In[112]:


# # Misclassification Rate
# print(f'Misclassification Rate is: {((fp+fn)/ (tp + fp + tn + fn))}')


# **The model incorrectly predicted 17.1% of the observations**

# **Model appears to be very overfit.  Excellent accuracy on training data. Testing data is far off though.  Need to bring in more bias and/or limit features for next model.**

# ---

# <a id='3'></a>
# ## Model #3 - TFID/LogReg

# In[93]:


# WIll be Piplining TfidfVectorizer with Logistic Regression CV estimator
# But first some EDA

X = subs['title']
y = subs['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)


# In[94]:


# which words are most important to the model?
tfidf = TfidfVectorizer(min_df=2)   # word much appear in two different docs to count
tfidf.fit(X_train)

tfidf_dict = {
    'words' :tfidf.get_feature_names(),
    'idf':tfidf.idf_
}
tfidif_dict = pd.DataFrame(tfidf_dict)
tfidif_dict.sort_values(by='idf', ascending = False).head(10)


# **Tfidf deams 赤光 (akimitsu- manderin for 'bright', 'light' the most important word in the corpus!.  'Paola' is also a latin word by origin.  Interesting how the model pulled these words out**

# In[95]:


X_train_df[['赤光', 'subreddit']].groupby('subreddit').sum()

# this word appears only twice in 'camping' subreddit!


# In[98]:


# Pipline for tfidf and logistic regression CV

pipe2 = Pipeline([
    ('tf', TfidfVectorizer(stop_words='english',ngram_range=(1, 2), analyzer ='word', min_df=5)),
    ('lr', LogisticRegressionCV(solver = 'liblinear', penalty='l2', cv=5, random_state = 42))
])

pipe2.fit(X_train, y_train)

print(f'Training Accuracy Score is: {pipe2.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {pipe2.score(X_test, y_test)}')


# **Model appears to be overfit.  Excellent accuracy on training data. Testing data is far off though.  Need to bring in more bias and/or limit features for next model.**
# 
# **Overall, testing accuracy is close to our production model!**

# In[102]:


# Running a Prediction
pipe2.predict(['Look at my van!'])


# ---

# <a id='4'></a>
# ## Model #4- CVEC/DTC

# In[38]:


# Will be Piplining CountVectorizor transformer with DecisionTreeClassifier estimator

X = subs['title']
y = subs['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)


# In[63]:


pipe3 = Pipeline([
    ('cv', CountVectorizer(min_df=2, stop_words='english')),
    ('dtc', DecisionTreeClassifier(max_depth =40, random_state = 42, min_samples_leaf= 3))
])
pipe3.fit(X_train, y_train)

print(f'Training Accuracy Score is: {pipe3.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {pipe3.score(X_test, y_test)}')


# **Model 4 does not perform as well as the others. Will continue on to work with better performers.**

# <a id='5'></a>
# ## Model #5 - CVEC/ ADA Boost

# In[92]:


# Will begin with a pipeline - CVEC transformer with AdaBoost Classifier
# AdaBoost performed better than XGBoost!
# Utilizing tokenized dataset yielded better results!
X = subs['title']
y = subs['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)

pipe4= Pipeline([
    ('cvec', CountVectorizer()),
    ('abc', AdaBoostClassifier(random_state=42))
])


# In[74]:


# GridSearch through our hyperparameters!
# setting up parameter dictionary:

pipe4_params= {'cvec__stop_words':['english', None],    # with and without stopwords
    'cvec__max_features':[None, 2000, 3000],           # number of highest frequency features to use
    'cvec__min_df':[2, 3],                             # ignore words at these low frequencies
    'cvec__ngram_range':[(1, 1), (1, 2)],              # unigrams and bigrams                      
}


# In[75]:


# Instatiating GridSearchCV

gs2= GridSearchCV(pipe4,
                param_grid=pipe4_params,
                cv=5,                    # 5 fold cross validation
                verbose = 1)
gs2.fit(X_train, y_train)


# In[76]:


gs2.best_params_


# In[77]:


print(f'Training Accuracy Score is: {gs2.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {gs2.score(X_test, y_test)}')


# **Model #5 did NOT perform very well.  Will move on to other models**

# <a id='6'></a>
# ## Model #6 - CVEC/SVC

# In[84]:


# Will begin with a pipeline - CVEC transformer with LinearSVC estimator
# Utilizing non-tokenized dataset yielded better results!
X = subs_notoken['title']
y = subs_notoken['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)

pipe5= Pipeline([
    ('cvec', CountVectorizer()),
    ('svc', LinearSVC(random_state= 42, max_iter=5000, C=0.07 ))
])


# In[85]:


# GridSearch through out hyperparameters!
# setting up parameter dictionary:

pipe_params= {'cvec__stop_words':['english', None],
    'cvec__max_features':[None, 2000, 3000, 4000],
    'cvec__min_df':[2, 3],
    'cvec__ngram_range':[(1, 1), (1, 2)],
    }


# In[86]:


# Instatiating GridSearchCV

gs= GridSearchCV(pipe5,
                param_grid=pipe_params,
                cv=5)


# In[87]:


# Fitting GS to training data
gs.fit(X_train, y_train)


# In[88]:


# Best paramaters from our GridSearch!
gs.best_params_


# In[89]:


# Score on Training and Testing Data

print(f'Training Accuracy Score is: {gs.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {gs.score(X_test, y_test)}')


# **This is just part of the answer.  Accuracy looks decent at 85.7%, And, VERY CLOSE to our production model. However, our Production model outperforms by a bit, AND runs much faster.**
# 
# **Let's still take a look at the misclassification rate, via the Confusion Matrix!** 

# In[90]:


preds = gs.predict(X_test)
tn, fp, fn, tp =confusion_matrix(y_test, preds).ravel()
plot_confusion_matrix(gs, X_test, y_test, cmap='Greens');


# In[91]:


# Misclassification Rate
print(f'Misclassification Rate is: {((fp+fn)/ (tp + fp + tn + fn))}')


# **The model incorrectly predicted 14.4% of the observations**
# 
# ---
