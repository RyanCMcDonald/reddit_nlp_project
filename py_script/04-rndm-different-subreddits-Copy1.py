#!/usr/bin/env python
# coding: utf-8

# # <span style="color:Purple">Project 3 :  Web APIs & NLP</span> <img src="../resources/reddit_logo.png" width="110" height="110" />
# ---
# ## <span style="color:Orange">Random Two Subreddit Pulls For Production Model- r/politics and r/spaceX</span>      
# 
# #### Ryan McDonald, General Assembly <img src="../resources/GA.png" width="25" height="25" />
# ---

# **This model is for presentation use and does not have extended analysis**
# 
# **Two random subreddits were pulled based on top trending for the site**
# **Imports**

# In[28]:


import requests
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Metrics!!!
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# ## spacex Subreddit
# ### 1. PushShift Loop to Grab spacex Subreddit Content

# In[29]:


# developing the loop through fullsubreddit pages
# Default values for 'sort', 'sort_type', and 'size' are good!

url = 'https://api.pushshift.io/reddit/search/submission'

def grab_posts (subreddit, last_page =None):
    params = {
        'subreddit':subreddit,
        'before': 1614414483}    
# to ensure we pull up mostly the same posts each time, adding a 'before' param! 
# post pull up until Saturday, February 27, 2021 3:28:03 AM GMT-05:00

    if last_page != None:
        if len(last_page) > 0:
            params['before'] = last_page[-1]['created_utc'] # last posts created timestamp
        else:
            return []
    results = requests.get(url, params)
    
    return results.json()['data']


# In[30]:


def most_posts (subreddit, max_submissions = 1000):
    
    submissions = []         # new list of submissions
    last_page = None         # only limiting on # of submissions

    # loop incorporated from Alex Patry (textjuicer.com)    
    while last_page != [] and len(submissions) < max_submissions:
        last_page = grab_posts(subreddit, last_page)
        submissions += last_page
        time.sleep(1)        # need a 'lag time' between loops
    return submissions[:max_submissions]


# In[31]:


start_time = time.time()
limit_posts = most_posts('spacex')
print ('limit_posts took', time.time() - start_time, 'sec to run')


# In[32]:


len(limit_posts)


# ### 2. Build DataFrame of Relevant Information

# In[33]:


space = pd.DataFrame(limit_posts)


# In[34]:


space = space[['subreddit','title']]
space.head()


# ## politics Subreddit 
# ### 1. PushShift Loop to Grab politics Subreddit Content

# In[35]:


# developing the loop through fullsubreddit pages
# Default values for 'sort', 'sort_type', and 'size' are good!

url = 'https://api.pushshift.io/reddit/search/submission'

def grab_posts (subreddit, last_page =None):
    params = {
        'subreddit':subreddit,
        'before': 1614414483}    
# to ensure we pull up same posts each time, adding a 'before' param! 
# post pull up until Saturday, February 27, 2021 3:28:03 AM GMT-05:00
 
    if last_page != None:
        if len(last_page) > 0:
            params['before'] = last_page[-1]['created_utc'] # last posts created timestamp
        else:
            return []
    results = requests.get(url, params)
    
    return results.json()['data']


# In[36]:


def most_posts (subreddit, max_submissions = 1000):
    
    submissions = []         # new list of submissions
    last_page = None         # only limiting on # of submissions

    # loop incorporated from Alex Patry (textjuicer.com)      
    while last_page != [] and len(submissions) < max_submissions:
        last_page = grab_posts(subreddit, last_page)
        submissions += last_page
        time.sleep(1)        # need a 'lag time' between loops
    return submissions[:max_submissions]


# In[37]:


start_time = time.time()
limit_posts = most_posts('politics')
print ('limit_posts took', time.time() - start_time, ' sec to run')


# In[38]:


len(limit_posts)


# ### 2. Build DataFrame of Relevant Information

# In[39]:


politics = pd.DataFrame(limit_posts)


# In[40]:


politics = politics[['subreddit','title']]
politics.head()


# ## Combining DataFrames

# In[41]:


combined = [space, politics]
submissions = pd.concat(combined)


# In[42]:


submissions


# Although my submission pull function has a refernce timer on it (to pull same posts each time), there may still be changes in submissions based on up-voting, deletions, etc.  I will 'hash' out the save_to_csv to ensure data remains frozen for modeling.

# In[43]:


submissions.shape


# ### 3. Running new Data Through the Production Model!

# In[44]:


# Will begin with a pipeline - CVEC transformer with BernoulliNB estimator
# Production model performed 1% better on testing data with the non-sent-tokenized data!

X = submissions['title']
y = submissions['subreddit']

# Subreddit is close to normalized, but will stratify on 'y' as a best practice

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state=42)


pipe= Pipeline([
    ('tf', TfidfVectorizer()),
    ('bnb', BernoulliNB())])
    


# In[45]:


# GridSearch through out hyperparameters!
# setting up parameter dictionary:

pipe_params= {'tf__stop_words':['english', None],     
              'tf__ngram_range':[(1, 2), (2,2)],
              'tf__analyzer':['word'],
              'tf__min_df':[0, 5, 10]       
}


# In[46]:


# Instatiating GridSearchCV

gs= GridSearchCV(pipe,
                param_grid=pipe_params,
                cv=8,                    # 5 fold cross validation
                verbose = 1)
gs.fit(X_train, y_train)


# <a id='analysis'></a>
# ### Production Model Analysis 

# In[47]:


gs.best_params_


# In[48]:


# Score on Training and Testing Data

print(f'Training Accuracy Score is: {gs.score(X_train, y_train)}')
print(f'Testing Accuracy Score is: {gs.score(X_test, y_test)}')


# **Our model succesfully predicted 97.2% of the classes correctly!  Excellent!**
