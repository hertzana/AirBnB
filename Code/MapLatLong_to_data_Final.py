# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 03:45:41 2018

@author: hanzhu
"""

'''
This code is used to clean data extracted from Google Places API
'''

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import math


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from collections import Counter


os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB")

coords = pd.read_csv('latlong.csv', encoding = "ISO-8859-1")



coords['city_2'] = coords['places'].apply(lambda x: x.split("; ")[0])
coords['closest_city'] = coords['places'].apply(lambda x: x.split("; ")[-2])
coords['middle_places'] = coords['places'].apply(lambda x: x.split("; ")[1:-2])


wpt = nltk.WordPunctTokenizer()
english = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

f = lambda x : [r for r in wpt.tokenize((''.join(ch for ch in x if ch not in exclude)).lower()) if len(r) > 1]
place_types = pd.get_dummies(coords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)

ptrain = pd.concat([coords, place_types], axis=1)

############# from test set ####################################################
testcoords = pd.read_csv('latlong_test.csv', encoding = "ISO-8859-1")

testcoords['city_2'] = testcoords['places'].apply(lambda x: x.split("; ")[0])
testcoords['closest_city'] = testcoords['places'].apply(lambda x: x.split("; ")[-2])
testcoords['middle_places'] = testcoords['places'].apply(lambda x: x.split("; ")[1:-2])

f = lambda x : [r for r in wpt.tokenize((''.join(ch for ch in x if ch not in exclude)).lower()) if len(r) > 1]
place_types_test = pd.get_dummies(testcoords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)

ptest = pd.concat([testcoords, place_types_test], axis=1)
############# combine train and test sets ####################################################
place_types_all = pd.concat([ptrain, ptest])

place_types_all.to_csv('latlng_all.csv', index=False)

############# drop the following vars ####################################################
# locality, political, pointofinterest, establishment, neighborhood, sublocality', 
# sublocalitylevel1, 

# add premise and subpremise into one var

place_types_all = place_types_all.drop(['locality', 'political', 'pointofinterest', 
                                        'establishment', 'neighborhood', 'sublocality', 
                                        'sublocalitylevel1'], axis=1)

place_types_all['premise_all'] = place_types_all['premise'] + place_types_all['subpremise']

place_types_all = place_types_all.drop(['premise', 'subpremise'], axis=1)

# Export final cleaned Google Places API data
place_types_all.to_csv('latlng_edited.csv', index=False)

