#!/usr/bin/env python
# coding: utf-8

# # <span style="color:Purple">Project 3 :  Web APIs & NLP</span> <img src="../resources/reddit_logo.png" width="110" height="110" />
# ---
# ## <span style="color:Orange">EDA </span>      
# 
# #### Ryan McDonald

# 
# **Imports**

# In[1]:


import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ## 1. Read the Data

# In[2]:


subs =pd.read_csv('../datasets/submissions')
subs


# ### Baseline statistics

# In[6]:


# Showing 'object datatypes'
subs.dtypes


# In[4]:


# looks like some titles aren't unique!
subs.describe()


# In[5]:


# some posts are duplicated
subs['title'].value_counts(ascending = False).head(10)


# In[6]:


# no missing 'title' entries.
subs['title'].isnull().sum()


# In[7]:


# verifying equal distribution of data between each subreddit
subs['subreddit'].value_counts(normalize = True)


# In[8]:


# number of unique authors
subs['author'].nunique()


# In[37]:


subs['author'].value_counts(ascending = False)

# the most prolific author was deleted!.. drunkbackpacker???  Haha!


# ## 2. Preprocessing
# 

# **Going to start by breaking the titles down into single sentences for further processing.  If the models don't work as well as they could, further segmentation into smaller/larger groups of words may occur**
# 
#     - Will breakdown VanLife and camping titles seperately in order to preserve relationship.
#     - Full lists of titles to be seperated into sentences
#     - New DataFrame developed with sentence breakdown for further analysis

# In[10]:


# starting with VanLife Titles
vl_titles = list(subs['title'][0:4000])
len(vl_titles)


# In[11]:


# VanLife tokenized sentences
vl_sent = " ".join(vl_titles)
vl_stoken = sent_tokenize(vl_sent)
len(vl_stoken)


# In[12]:


# camping titles
vl_titles = list(subs['title'][4000:8000])
len(vl_titles)


# In[13]:


# camping tokenized sentences
cmp_sent = " ".join(vl_titles)
cmp_stoken = sent_tokenize(cmp_sent)
len(cmp_stoken)


# **Tokenizing into sentences appears to combine titles here and there, resulting in less unique datapoints. But, may in-turn create a better model**
# 
#     - Just to have it available, I'll preserve a list of the titles unaltered
#     
# **Building a DataFrame with tokenized titles**

# In[14]:


token_df = pd.DataFrame(
    {'subreddit':'camping',
    'title': cmp_stoken})
vl_df = pd.DataFrame(
    {'subreddit':'VanLife',
     'title':vl_stoken})

token_df = pd.concat([token_df, vl_df], ignore_index= True)
token_df.head()


# In[15]:


token_df.shape


# In[16]:


# Save to CSV!

token_df.to_csv('../datasets/tokenized_df', index= False )


# ### Baseline Score
# **With the amount of data pulled from Reddit, and the low baseline score, I would expect modeling to perform much better**

# In[17]:


token_df['subreddit'].value_counts(normalize= True)
# This will show baseline 'majority' case.  
# 'Guessing' VanLife each time would be correct 54.4% of the time!


# **I prefer to start off with the tokenized sentences because it skews the originally-normalized data.  Having this baseline score in VanLife's favor may product more interesting results down the line**

# #### Quick Sentiment Check! (curiousity strikes)
# 
# **Does either subreddit as a whole have a better sentiment analysis?**

# In[18]:


# Seeing as SIA takes 'length' into account when producing results,
# this is for 'entertainment' purposes only!

sia = SentimentIntensityAnalyzer()

vl_text= '-'.join(subs['title'][0:4000])
camp_text = '-'.join(subs['title'][4000:8000])


# In[19]:


sia.polarity_scores(vl_text)


# In[20]:


sia.polarity_scores(camp_text)


# **An interesting finding above! Both SubReddits have VERY similar sentiment polarity scores.  Assuming since they are related to each other, and typically involve optimistic people.  That was good to see!.  HOWEVER... to rule out any bias towards total word count, I'll break down the SIA below per title in the DataFrame**

# ### Applying SIA to the entire DataFrame!

# In[24]:


sentiment = subs['title'].apply(sia.polarity_scores)
sentiment_df = pd.DataFrame(sentiment.tolist())
sentiment_df.sort_values(by= ['compound'], ascending = False)


# **Average Sentiment Per Subreddit!**

# In[34]:


# Average VanLife Sentiment:
(sentiment_df.loc[0:4000]['compound'].mean())


# In[35]:


# Average Camping Sentiment:
sentiment_df.iloc[4000:8001]['compound'].mean()


# **It's not by much, but 'camping' subreddit mean sentiment is higher than 'VanLife' subereddit mean sentiment. Perhaps a few posts regarding broken vans or getting lost were written in the 'VanLife' subreddit!**

# In[ ]:




