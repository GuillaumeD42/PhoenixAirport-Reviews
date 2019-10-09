

# Librairies import

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import nltk
import string
import nlp_utilities as nlp
from nltk.text import Text
from matplotlib import pyplot
import pymysql
nltk.download()


# Data Import

server = "analyst-toolbelt.cn119w37trlg.eu-west-1.rds.amazonaws.com"
username = "*****"
password = "*****"

connection = pymysql.connect(host=server,
                             user=username,
                             password=password,
                             db='yelp',
                             charset='utf8')

QUERY = """
select * from review
where business_id = 'hW0Ne_HTHEAgGF1rAdmR-g';
"""

myreviews = pd.read_sql(QUERY, connection)

myreviews.head()


# Data Aalysus 
len(myreviews)

# Types of the columns

# In[12]:

myreviews.dtypes


# <center>Analysis of the Time Period</center>

# In[13]:

SQL1 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' ORDER BY a.date ASC"


# In[14]:

firstreview = pd.read_sql(SQL1, connection)


# In[15]:

firstreview[:1]


# As we can see, the first review is from 2007-02-11.

# In[16]:

SQL2 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' ORDER BY a.date DESC"


# In[17]:

lastreview = pd.read_sql(SQL2, connection)


# In[18]:

lastreview[:1]


# As we can see, the last review is from 2012-12-30.

# In[19]:

SQL3 = "SELECT a.stars, count(user_id) FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' GROUP BY a.stars"


# In[20]:

stars = pd.read_sql(SQL3, connection)


# In[21]:

stars


# In[23]:

stars.plot.bar(x='stars', rot=0, title='Ratings', figsize=(7,3), fontsize=12)


# <center> Let's check the time period of these ratings </center>

# In[23]:

SQL4 = "SELECT a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' AND a.stars = '5' ORDER BY 1 ASC"


# In[24]:

five_stars = pd.read_sql(SQL4, connection)


# In[25]:

len(five_stars)


# In[26]:

five_stars[:1]


# In[27]:

five_stars[9:10]


# As we can see (and I checked for every other ratings), the time lapse is not very relevant

# Let's look at the distribution of stars over time. There are no real trends, especially for the last 2 years of reviews. 

# In[31]:

get_ipython().magic('matplotlib inline')
myreviews.sort_values(by="date", ascending=True).plot(x="date", y="stars")


# <center><b> Reviews per user </b></center>

# In[28]:

SQL5 = "SELECT user_id, count(text) FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' GROUP BY user_id"


# In[29]:

review_per_user = pd.read_sql(SQL5, connection)


# In[30]:

review_per_user


# No user has posted twice

# In[39]:

SQL6 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' AND funny_votes > '5' ORDER BY funny_votes DESC"


# In[40]:

funny_votes = pd.read_sql(SQL6, connection)


# In[41]:

funny_votes


# In[45]:

funniest = funny_votes[0:1]


# In[46]:

funniest['text']


# In[48]:

def write_review_files(path, reviewlist):
    """ Takes a directory path to write files into, and a list of review text."""
    if not path.endswith("/"):
        path = path + "/"
    try:
        for i, row in enumerate(reviewlist):
            filename = "review" + str(i) + ".txt"
            with open(path + filename, "w") as handle:
                handle.write(row)
    except:
        print("Something wrong with the path or file list. Does the directory exist?")
    print("Wrote %s files to %s." % (len(reviewlist), path))


# In[49]:

write_review_files("data/funnyreview", funniest['text'])


# <u><b>Note</b></u>: This is also the file judged as the most useful review, it might be important to look at what is said
