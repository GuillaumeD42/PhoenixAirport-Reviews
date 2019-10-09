

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


# Data Aalysis 
len(myreviews)

# Types of the columns
myreviews.dtypes


# Analysis of the Time Period

SQL1 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' ORDER BY a.date ASC"

firstreview = pd.read_sql(SQL1, connection)
firstreview[:1]

SQL2 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' ORDER BY a.date DESC"

lastreview = pd.read_sql(SQL2, connection)

lastreview[:1]

SQL3 = "SELECT a.stars, count(user_id) FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' GROUP BY a.stars"

stars = pd.read_sql(SQL3, connection)

stars

stars.plot.bar(x='stars', rot=0, title='Ratings', figsize=(7,3), fontsize=12)


# Let's check the time period of these ratings 

SQL4 = "SELECT a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' AND a.stars = '5' ORDER BY 1 ASC"

five_stars = pd.read_sql(SQL4, connection)
len(five_stars)
five_stars[:1]
five_stars[9:10]

get_ipython().magic('matplotlib inline')
myreviews.sort_values(by="date", ascending=True).plot(x="date", y="stars")


# Reviews per user
SQL5 = "SELECT user_id, count(text) FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' GROUP BY user_id"

review_per_user = pd.read_sql(SQL5, connection)
review_per_user

SQL6 = "SELECT b.name, a.business_id, a.date, user_id, a.stars, useful_votes, funny_votes, cool_votes, text FROM review a INNER JOIN business b on a.business_id = b.business_id WHERE a.business_id = 'hW0Ne_HTHEAgGF1rAdmR-g' AND funny_votes > '5' ORDER BY funny_votes DESC"

funny_votes = pd.read_sql(SQL6, connection)
funny_votes
funniest = funny_votes[0:1]
funniest['text']


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

write_review_files("data/funnyreview", funniest['text'])


# Text Analysis - Dataset Cleaning

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


write_review_files("data/myreviews", mydata['text'])


def get_filenames(folder):
    """ Pass in a folder name, with or without the / at end.
    Returns a list of the files & paths inside it (no folders).
    """
    from os import listdir
    from os.path import isfile, join
    # because we want to return full paths, we need to make sure there is
    # a / at the end.
    # If this doesn't work on Windows, change the slash direction.
    if folder[-1:] != "/":
        folder = folder + "/"
    # this will return only the filenames, not folders inside the path
    # also filter out .DS_Store which is on Macs.
    return [folder + f for f in listdir(folder) if isfile(join(folder, f)) and f != ".DS_Store"]

filenames = get_filenames("data/myreviews")

def load_texts_as_string(filenames):
    """ Takes a list of filenames as arg.
    Returns a dictionary with filename as key, string as value.
    """
    from collections import defaultdict
    loaded_text = defaultdict(str)  # each value is a string, the text
    for filename in filenames:
        with open(filename, errors="ignore") as handle:
            loaded_text[filename] = handle.read()
    return loaded_text

texts = load_texts_as_string(filenames)

print("Searching the string 12:", re.search('[a-zA-Z]', '12'))
print("Searching the string time12", re.search('[a-zA-Z]', 'time12'))
print("Searching the string 3rd", re.search('[a-zA-Z]', '3rd'))


texts

tokenizedreviews = [nlp.tokenize_clean_stem(value) for key, value in texts.items()]
sourcefilenames = [key for key in texts.keys()]

mergedreviews = list(itertools.chain.from_iterable(tokenizedreviews))

mylist = ["'s", "n't"]

def remove_custom(wordlist, mylist):
    return [word for word in wordlist if word not in mylist]

clean_reviews = remove_custom(mergedreviews, mylist)

len(clean_reviews)

# Common Words in the DataSet

def print_counts(tokens, count=10):
    # Takes a list of words, counts, prints top "count" words.
    from collections import Counter
    mycounts = Counter(tokens)
    print("Word\tCount")
    for word,count in mycounts.most_common(count):
        print("%s\t%s" % (word,count))

print_counts(clean_reviews)

# Most Common Bigrams

bigram_measures = nltk.collocations.BigramAssocMeasures()

word_fd = nltk.FreqDist(clean_reviews)
bigram_fd = nltk.FreqDist(nltk.bigrams(clean_reviews))
finder = BigramCollocationFinder(word_fd, bigram_fd)
scored = finder.score_ngrams(bigram_measures.likelihood_ratio)
scored[0:50]


# Collocations

textobject = Text(clean_reviews)
textobject.collocations(25)


# Concordances

textobject.concordance("wifi")
textobject.concordance("hallway")
textobject.concordance("tsa")
textobject.concordance("airway")
textobject.concordance("rail")

# Similarities between the reviews (TF-IDF)

def tfidf(t, d, D):
    # term freq is the count of term as percent of the doc's words
    # d.count counts how many times t occurs in d.
    tf = float(d.count(t)) / len(d) 
    # Note this version doesn't use +1 in denominator as many do.
    idf = math.log( float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf * idf

def makeText_from_tokens(tokens):
    return nltk.Text(tokens)

def makeTextCollection(tokenslist):
    texts = [nltk.Text(doc) for doc in tokenslist]
    collection = nltk.TextCollection(texts)
    return collection, texts

def compute_tfidfs_by_doc(filenames):

    from collections import defaultdict  # not the textcollection!
    import nlp_utilities as nlp
    

    alltokens = []
    textslist = nlp.load_texts_as_string(filenames)
    for text in textslist.values():
        alltokens.append(nltk.word_tokenize(text))
    collection, textobjs = makeTextCollection(alltokens)
    
   
    stats = defaultdict(list) 
    
    for i, text in enumerate(textobjs):
        for word in text.vocab().keys():  
            
            tfidfscore = collection.tf_idf(word, text)
            tf = collection.tf(word, text) 
            count = text.count(word) 
            if tfidfscore > 0: 
                stats[filenames[i]].append({
                    "word": word,
                    "tfidf": tfidfscore,
                    "tf": tf,
                    "count": count
                })
    return stats

def top_n(list_of_dicts, field, n):
    """ Sorts dicts by a field's value and returns N top results. """
    return sorted(list_of_dicts, key=lambda x: x[field], reverse=True)[0:n]

def myfunction(folder,output_file):
    files = nlp.get_filenames(folder)
    mycoll = compute_tfidfs_by_doc(files)
    with open(output_file, "w") as handle:
        for key, values in mycoll.items():
            top_10 = top_n(values, 'tfidf', 10)   
            for result in top_10:
                result['word'] = str(result['word'])
                result['tfidf'] = str(result['tfidf'])
                result['count'] = str(result['count'])
                result['tf'] = str(result['tf'])
                handle.write(','.join([key, result['word'],result['tfidf'],result['count'],result['tf'], "\n"]))

myfunction("data/myreviews/","top10reviews")

mydataframe = pd.read_csv("top10reviews", names = ['Key','Word','TF-IDF','Count', 'TF' ,'NaN'])
mydataframe.head()
del mydataframe['NaN']


# Similarities between the reviews (K-Means)

texts = nlp.load_texts_as_string(filenames)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2, tokenizer=nlp.tokenize_clean_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts.values())
terms = tfidf_vectorizer.get_feature_names()

num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

def clean_filename(path):
    import os
    return os.path.basename(path).strip(".txt")

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for cluster in range(num_clusters):
    print("Cluster %d words:" % cluster, end='')
    
    for ind in order_centroids[cluster, :10]: 
        print(terms[ind], end=',')
    print()
    print()
    
    print("Cluster %d documents:" % cluster, end='')
    

    for item, filename in enumerate(list(texts.keys())):
        if clusters[item] == cluster:
            print(' %s,' % clean_filename(filename), end='')
    print() 
    print()
    
print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

# set up cluster names using a dict
cluster_names = {3: 'Flight', 
                 1: 'Food', 
                 2: 'Security', 
                 0: 'Harbor'}

dist = 1 - cosine_similarity(tfidf_matrix)

mds = MDS(n_components=2, dissimilarity="precomputed")
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]

def clean_filenames(paths):
    # Makes labels for dots that are just the root filename minus .txt
    import os
    return [os.path.basename(path).strip(".txt") for path in paths]

labels = clean_filenames(list(texts.keys()))

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=labels))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) 
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          
        which='both',     
        bottom='off',      
        top='off',         
        labelbottom='off')
    ax.tick_params(        axis= 'y',         
        which='both',      
        left='off',      
        top='off',         
        labelleft='off')
    
ax.legend(numpoints=1)  

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8) 

def get_years(labels):
    return [label.split('_')[-1:][0] for label in labels]

mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], get_years(labels)):
    ax.text(x, y, z, s, size=8)

plt.show()


# Clustering & Topic Modeling

tfidf = TfidfVectorizer(tokenizer=nlp.tokenize_clean_stem).fit_transform(texts.values())
vectors = tfidf.toarray()

dist = pdist(vectors, metric='cosine') 

link = linkage(dist, method="ward")

def make_dend(data, method='ward', labels=None, height=8):
    from pylab import rcParams
    dist = pdist(data, metric='cosine')
    link = linkage(dist, method=method)
    rcParams['figure.figsize'] = 6, height
    rcParams['axes.labelsize'] = 5
    if not labels:
        dend = dendrogram(link, orientation='right') #labels=names)
    else:
        # the label is actually the file + it's number 
        dend = dendrogram(link, orientation='right', labels=[str(i) + '_' + label for i, label in enumerate(labels)])
    return dist

def clean_filenames(paths):
    # Makes labels that are just the root filename minus .txt
    import os
    return [os.path.basename(path).strip(".txt") for path in paths]

dist = make_dend(vectors, method='complete', height=10, labels=clean_filenames(texts.keys()))

def make_heatmap_matrix(dist, method='complete'):
    """ Pass in the distance matrix; method options are complete or single """
    # Compute and plot first dendrogram.
    fig = plt.figure(figsize=(10,10))
    # x y width height (left, bottom, w, h)
    ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
    Y = linkage(dist, method=method)
    Z1 = dendrogram(Y, orientation='right')
    ax1.set_xticks([])  # suppress labels on height.
    # yticks are the number labels, let them live.

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.74,0.6,0.2])
    Z2 = dendrogram(Y)
    #ax2.set_xticks([])
    #ax2.set_yticks([])

    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = squareform(dist)
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)

make_heatmap_matrix(dist, method='complete')


# Sentiment Analysis

def get_vader_sentiment(sentence, analyzer):
    """ Pass in the setence to be analysed and the analyzer, e.g., sid.
    Returns a dictionary of the scores.
    """
    ss = analyzer.polarity_scores(sentence)
    return ss

sid = SentimentIntensityAnalyzer()

scores = []

for file in filenames:
    with open(file, encoding="utf8") as handle:
        text = handle.read()
        text = text.strip("\n")
        sentences = nltk.sent_tokenize(text) 
        for i, sent in enumerate(sentences):
            infos = {'Filename':file,'Text':sent,'Index':i}
            myscores = get_vader_sentiment(sent,sid).copy()
            mydict = {k : v for d in [infos, myscores] for k,v in d.items()}
            scores.append(mydict)

Sentiment_Reviews = pd.DataFrame.from_records(scores)

# Most positive reviews
Sentiment_Reviews.sort_values(by ='compound', ascending = False).head()

# Most negative reviews
Sentiment_Reviews.sort_values(by ='compound', ascending = True).head()

NEGWORDS = "data/sentiment_wordlists/negative-words.txt"
POSWORDS = "data/sentiment_wordlists/positive-words.txt"

def load_words(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        words = handle.readlines()
    words = [w.strip() for w in words if w[0] != ';']
    words = [word for word in words if word]  # get rid of empty string
    return words

negwords = load_words(NEGWORDS)
poswords = load_words(POSWORDS)

def get_overlap(list1, list2):
    """ If you have a list of words (tokens) and you want to get the overlap with a second list, 
    like polarity words.
    Returns the overlapping words and their counts as a tuple.
    """
    from collections import Counter
    list1_multiset = Counter(list1)
    list2_multiset = Counter(list2)
    overlap = list((list1_multiset & list2_multiset).elements())
    totals = []
    for word in overlap:
        totals.append((word, list1_multiset[word]))
    return totals

def get_sentiment_counts(text, filename, poswords=poswords, negwords=negwords):
    """ This takes a text, a filename, and polarity wordlists and counts for you.
    Returns a dictionary.
    """
    from collections import Counter
    count = dict()
    overlap_pos = get_overlap(text, poswords)
    overlap_neg = get_overlap(text, negwords)
    count = {
            "file": filename,
            "positive_total": int(sum(Counter(dict(overlap_pos)).values())),
            "positive_words": list(overlap_pos),
            "negative_total": int(sum(Counter(dict(overlap_neg)).values())),
            "negative_words": list(overlap_neg),
            "word_count": int(len(text)),
            "text": " ".join(text)
        }
    count['net_score'] = count['positive_total'] - count['negative_total']
    return count

myfields = ['file', 'positive_total', 'positive_words', 'negative_total', 'negative_words', 'net_score', 'word_count', 'text']

def write_sentiment_results(output_file, filenames, fields):
    """ Input args are the filename to write to, the files you are analying, and fields to write out. 
    The separator is a tab (\t).
    """
    import csv
    with open(output_file, 'w', errors='ignore') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for input_file in filenames:
            # here we do exactly what we did above, but for all of them.
            tokens = nlp.tokenize_text(input_file)
            tokens = nlp.clean_tokens(tokens)
            dictversion = get_sentiment_counts(tokens, input_file)
            writer.writerow(dictversion)

write_sentiment_results("reviews_airport.csv", filenames, fields=myfields)

reviews_df = pd.read_csv("reviews_airport.csv", sep="\t")

pos_df = reviews_df.sort_values(by='net_score', ascending = False)
pos_df[['file', 'positive_total', 'negative_total', 'net_score', 'word_count']]

neg_df = reviews_df.sort_values(by='net_score', ascending = True)
neg_df[['file', 'positive_total', 'negative_total', 'net_score', 'word_count']]

reviews_df[["positive_total", "negative_total"]].plot(kind="bar")
reviews_size = reviews_df.plot(x="word_count", y="net_score", kind="scatter")
plt.savefig('reviews.png')

reviews_df['net_score'].plot(kind="bar")
