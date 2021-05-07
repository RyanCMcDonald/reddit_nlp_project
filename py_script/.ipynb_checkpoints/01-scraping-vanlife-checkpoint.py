#!/usr/bin/env python
# coding: utf-8

# # <span style="color:Purple">Project 3 :  Web APIs & NLP</span> <img src="../resources/reddit_logo.png" width="110" height="110" />
# ---
# ## <span style="color:Orange">Subreddit Submission Pulls</span>      
# 
# #### Ryan McDonald
# ---

# 
# **Imports**

# In[1]:


import requests
import pandas as pd
import time


# ### 1. PushShift API Parameters Dictionary
# 
# | Parameter |                      Description                     |       Default       |                      Accepted Values                      |
# |:---------:|:----------------------------------------------------:|:-------------------:|:---------------------------------------------------------:|
# | q         | Search term.                                         | N/A                 | String / Quoted String for phrases                        |
# | ids       | Get specific comments via their ids                  | N/A                 | Comma-delimited base36 ids                                |
# | size      | Number of results to return                          | 25                  | Integer <= 100                                            |
# | fields    | One return specific fields (comma delimited)         | All Fields Returned | string or comma-delimited string                          |
# | sort      | Sort results in a specific order                     | "desc"              | "asc", "desc"                                             |
# | sort_type | Sort by a specific attribute                         | "created_utc"       | "score", "num_comments", "created_utc"                    |
# | aggs      | Return aggregation summary                           | N/A                 | ["author", "link_id", "created_utc", "subreddit"]         |
# | author    | Restrict to a specific author                        | N/A                 | String                                                    |
# | subreddit | Restrict to a specific subreddit                     | N/A                 | String                                                    |
# | after     | Return results after this date                       | N/A                 | Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days) |
# | before    | Return results before this date                      | N/A                 | Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days) |
# | frequency | Used with the aggs parameter when set to created_utc | N/A                 | "second", "minute", "hour", "day"                         |
# | metadata  | display metadata about the query                     | false               | "true", "false"                                           |

# ## VanLife Subreddit
# ### 1. PushShift Loop to Grab VanLife Subreddit Content

# In[2]:


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


# In[3]:


def most_posts (subreddit, max_submissions = 4000):
    
    submissions = []         # new list of submissions
    last_page = None         # only limiting on # of submissions

    # loop incorporated from Alex Patry (textjuicer.com)    
    while last_page != [] and len(submissions) < max_submissions:
        last_page = grab_posts(subreddit, last_page)
        submissions += last_page
        time.sleep(1)        # need a 'lag time' between loops
    return submissions[:max_submissions]


# In[4]:


start_time = time.time()
limit_posts = most_posts('VanLife')
print ('limit_posts took', time.time() - start_time, 'sec to run')


# In[5]:


len(limit_posts)


# ### 2. Build DataFrame of Relevant Information

# In[6]:


vl_df = pd.DataFrame(limit_posts)


# In[7]:


vl_df = vl_df[['subreddit', 'selftext', 'title','score','media_only', 'author']]
vl_df.head()


# ### 3. Save to CSV file

# Although my submission pull function has a refernce timer on it (to pull same posts each time), there may still be changes in submissions based on up-voting, deletions, etc.  I will 'hash' out the save_to_csv to ensure data remains frozen for modeling.

# In[8]:


# vl_df.to_csv('../datasets/vanlife_df', index = False)


# ## Camping Subreddit 
# ### 1. PushShift Loop to Grab Camping Subreddit Content

# In[9]:


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


# In[10]:


def most_posts (subreddit, max_submissions = 4000):
    
    submissions = []         # new list of submissions
    last_page = None         # only limiting on # of submissions

    # loop incorporated from Alex Patry (textjuicer.com)      
    while last_page != [] and len(submissions) < max_submissions:
        last_page = grab_posts(subreddit, last_page)
        submissions += last_page
        time.sleep(1)        # need a 'lag time' between loops
    return submissions[:max_submissions]


# In[11]:


start_time = time.time()
limit_posts = most_posts('camping')
print ('limit_posts took', time.time() - start_time, ' sec to run')


# In[12]:


len(limit_posts)


# ### 2. Build DataFrame of Relevant Information

# In[13]:


camp = pd.DataFrame(limit_posts)


# In[14]:


list(camp.columns)[:10]


# In[15]:


camp = camp[['subreddit', 'selftext', 'title','score','media_only', 'author']]
camp.head()


# ### 3. Save to CSV file

# Although my submission pull function has a refernce timer on it (to pull same posts each time), there may still be changes in submissions based on up-voting, deletions, etc.  I will 'hash' out the save_to_csv to ensure data remains frozen for modeling.

# In[16]:


# camp.to_csv('../datasets/camp_df', index = False)


# ## Combining DataFrames

# In[17]:


combined = [vl_df, camp]
submissions = pd.concat(combined)


# In[18]:


submissions


# Although my submission pull function has a refernce timer on it (to pull same posts each time), there may still be changes in submissions based on up-voting, deletions, etc.  I will 'hash' out the save_to_csv to ensure data remains frozen for modeling.

# In[19]:


submissions = submissions[['subreddit', 'title', 'author']]
submissions.shape


# In[20]:


# submissions.to_csv('../datasets/submissions', index = False)


# In[21]:


title_data = submissions[['subreddit', 'title']]
title_data.shape


# In[22]:


# title_data.to_csv('../datasets/title_data', index = False)

