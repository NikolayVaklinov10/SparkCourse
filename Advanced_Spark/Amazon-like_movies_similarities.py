#!/usr/bin/env python
# coding: utf-8

# **The goal of this application is to produce a program which can give recommendations to customers about which movie to watch next based on their previous tiltles they have watched**

# In[3]:


#get_ipython().run_cell_magic('html', '', '<img src="img/amazon.png",width="900",height="100">')


# In[8]:


import sys


# In[9]:


from pyspark import SparkConf, SparkContext


# In[10]:


from math import sqrt


# In[11]:


def loadMovieNames():
    movieNames = {}
    with open('ml-100k/u.item',encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# In[12]:


# Python 3 doesn't allow passing around unpacked tuples,
# so I explicitly extract the ratings now.
def makePairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))


# In[13]:


def filterDuplicates( userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


# In[14]:


def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
        
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    
    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))
        
    return (score, numPairs)


# In[15]:


conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext.getOrCreate(conf=conf)


# In[16]:


print("\nLoading movie names...")
nameDict = loadMovieNames()


# In[17]:


data = sc.textFile('ml-100k/u.data')


# In[18]:


# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l:(int(l[0]), (int(l[1]), float(l[2]))))


# In[19]:


# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)


# In[20]:


# At this point the RDD consists of userID => ((movieID,
# rating), (movieID, rating))


# In[21]:


# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)


# In[22]:


# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)


# In[23]:


# Now I have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()


# In[24]:


# We now have (movie1, movie2) => (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()


# In[25]:


# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")


# In[30]:


# Extract similarities for the movie we care about that are "good".
if (len(sys.argv) > 1):

    scoreThreshold = 0.97
    coOccurenceThreshold = 50

    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim:         (pairSim[0][0] == movieID or pairSim[0][1] == movieID)         and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))


# In[ ]:


#jupyter nbconvert --to python Amazon-like_movies_similarities.ipynb


# In[ ]:





# In[7]:





# In[ ]:




