#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating


# In[3]:


def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.item", encoding='ascii',errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# In[4]:


conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext.getOrCreate(conf=conf)
sc.setCheckpointDir('checkpoint')


# In[5]:


print("\nLoading movie names...")
nameDict = loadMovieNames()


# In[6]:


data = sc.textFile("ml-100k/u.data")


# In[7]:


ratings = data.map(lambda l:
l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()


# In[8]:


# Build the recommendation model using Altering Least Squares
print("\nTraining recommendation model...")
rank = 10


# In[9]:


# Lowered numIterations to ensure it works on lower-end systems
numIterations = 6
model = ALS.train(ratings, rank, numIterations)


# In[11]:


userID = int(sys.argv[1])

print("\nRatings for user ID " + str(userID) + ":")
userRatings = ratings.filter(lambda l: l[0] == userID)
for rating in userRatings.collect():
    print (nameDict[int(rating[1])] + ": " + str(rating[2]))

print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print (nameDict[int(recommendation[1])] +         " score " + str(recommendation[2]))


# In[ ]:




