# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:25:00 2018

@author: hanzhu
"""

import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt

import os
import datetime
from geopy.distance import vincenty


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB")




path = ""
train = pd.read_csv(path + "train.csv")
train['dataset'] = 'train'
test = pd.read_csv(path + "test.csv")
test['dataset'] = 'test'
data = pd.concat([train,test], axis = 0).reset_index()

# =============================================================================
# Word Processing on Description - Bag of Words
# =============================================================================
desc = data[['id', 'description']]

wpt = nltk.WordPunctTokenizer()
english = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def normalize_text(text):
    if pd.isnull(text):
        return ''
    else:
        # re.sub(r'[^a-zA-Z\s]', '', a, re.I|re.A) --> delete all punctuation and
        # non-character letters
        # strip = strip of trailing and leading spaces
        # lower = make all lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower().strip()
        # tokenize
        tokens = wpt.tokenize(text)
        # filter stopwords out of document
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                           if ((token not in stop_words) & (token in english))]
        # re-create document from filtered tokens
        text = ' '.join(filtered_tokens)
        return text

normalize_corpus = np.vectorize(normalize_text)

desc['normtext'] = [normalize_corpus(x) for x in desc['description']]

# Change normtext to a list of strings
tlist = []
for i in desc['normtext']:
    tlist.append(str(i))
    
allwords = ' '.join(tlist)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0., max_df=1., max_features=10000)
cv_matrix1 = cv.fit_transform(tlist)
cv_matrix1 = cv_matrix1.toarray()
headers = np.array(cv.get_feature_names())

# concat headers and matrix
words = pd.DataFrame(cv_matrix1, columns=headers)

# get the words that matter
view = ['view', 'panoramic']
walk = ['walk', 'walkable', 'walking']
positive = ['great', 'good']
pos_sent = ['inviting', 'colorful', 'gold', 'sensitive', 'desirable', 'plush', 'romantic', 'adorable', 'tastefully','tasteful',
            'cute', 'nicely', 'pleasure', 'favorite', 'quintessential', 'pretty', 'interesting', 'lovely', 'enjoy', 'enjoying']
very_pos = ['best', 'perfect', 'perfectly', 'amazing', 'super', 'wonderful', 'grand', 'awesome', 'excellent', 
            'fantastic', 'incredible', 'spectacular', 'fabulous', 'gem', 'authentic', 'rare', 'exciting', 
            'incredibly', 'rich', 'dream', 'magnificent', 'deluxe', 'scenic', 'picturesque', 'exceptional', 'paradise']
new = ['new', 'newly']
big = ['big', 'large', 'huge', 'giant', 'massive', 'sweeping', 'enormous', 'expansive', 'roomy']
near = ['near', 'close', 'nearby', 'proximity', 'adjacent']
minutes = ['minute', 'min']
quiet = ['quiet', 'peaceful', 'serene', 'tranquil', 'calm', 'peace']
beautiful = ['beautiful', 'beautifully', 'gorgeous', 'beauty']
city = ['city', 'urban', 'metropolitan']
comfortable = ['comfortable', 'comfy', 'comfortably', 'comfort']
cozy = ['cozy', 'coziness', 'homey', 'cosy']
central = ['central', 'center', 'centrally']
many = ['many', 'numerous', 'ample', 'plenty']
shopping = ['shopping', 'shop', 'store', 'plaza', 'bakery']
porch = ['porch', 'patio', 'deck']
entire = ['entire', 'whole']
couch = ['couch', 'sofa']
historic = ['historic', 'history', 'historical']
outdoors = ['outdoors', 'mountain', 'outdoor', 'lush', 'trail', 'nature', 'glen', 'forest', 'valley', 'apple', 'oak',
            'sanctuary', 'bike', 'sunset', 'natural']
sunny = ['sunny', 'bright', 'sun', 'sunlight', 'sunshine', 'sunlit']
easy = ['easy', 'convenient', 'convenience', 'conveniently']
luxury = ['luxury', 'luxurious', 'penthouse', 'oasis', 'gated', 'resort', 'villa', 'estate', 'mansion']
charming = ['charming', 'charm']
near_ocean = ['ocean', 'pacific', 'bay', 'marina', 'pier', 'waterfront', 'sand', 'harbor', 'wharf', 'boat', 'shore', 
              'seaport', 'alcove', 'coast', 'surf']
near_small_water = ['river', 'lake', 'ferry', 'riverside', 'pond', 'creek', 'canal']
famous = ['famous', 'fame', 'iconic']
relax = ['relax','relaxed', 'relaxation', 'unwind']
fast = ['fast', 'quick', 'quickly']
modern = ['modern', 'contemporary']
vacation = ['vacation', 'retreat']
busy = ['busy', 'bustling', 'bustle', 'lively', 'hustle']
stone = ['stone', 'marble', 'granite']
upscale = ['upscale', 'elegant', 'chic', 'exclusive', 'premium', 'prestigious', 'fancy', 'expensive', 'vanity', 'fashion']
architecture = ['architecture', 'architectural']
vintage = ['vintage', 'antique']
culture = ['culture', 'cultural']
fresh = ['fresh', 'freshly']
sport = ['sport', 'stadium', 'golf', 'basketball', 'tennis']
music = ['music', 'jazz']
safe = ['safe', 'secure', 'security']
smoke = ['smoke', 'smoking']
traffic = ['traffic', 'highway']
pet = ['pet', 'dog']
basic = ['basic', 'simple']
cheap = ['cheap', 'discount', 'bonus']
#pool = ['pool', 'swimming']

lists = [view, walk, positive, pos_sent, very_pos, new, big, near, minutes, quiet, beautiful, 
         comfortable, cozy, central, many, shopping, porch, entire, couch,
         historic, outdoors, sunny, easy, luxury, charming, near_ocean, 
         near_small_water, famous, relax, fast, modern, vacation, busy, stone, upscale,
         architecture, vintage, culture, fresh, sport, music, safe, smoke, traffic, pet, basic,
         cheap]

words2 = pd.DataFrame()

# Redo the count so that we have a single count for each group of similar words
for item in lists:
    x = 1
    words2[item[0]] = words[item[0]]
    while x < len(item):
        words2[item[0]] += words[item[x]]
        x += 1

# rename some columns
words2 = words2.rename(columns={'great':'positive', 'inviting':'pos_sent', 
                                'best':'very_pos', 'museum':'attractions', 'ocean':'near_ocean',
                                'river':'near_small_water'})
    


keep_words = ['private', 'home', 'house', 'full', 'living',
              'space', 'away', 'access', 'two', 
              'neighborhood', 'area', 'floor', 'beach', 
              'queen',
              'location', 'free', 'complimentary', 'distance', 'spacious', 'clean',
              'downtown', 'studio', 'need', 'square',
              'coffee', 'well', 'fully', 'heart', 'pizza',
              'love', 
              'garden', 'everything', 'light', 'small', 'high', 'west', 
              'table', 'air',
              'closet', 'furnished', 'village', 'open', 'nice', 'stop', 
              'front', 'short', 
              'public', 'top', 'friendly', 'work', 
              'king', 'separate', 
              'suite', 'solo',  'balcony', 'local',
              'privacy', 'basement', 'share',
              'little', 'old', 'south', 'phone', 'hidden', 'major', 'happy',
              'experience', 'north', 'additional',
              'style', 'stylish', 'hotel', 
              'garage', 'furniture', 'upper', 'lower', 'cool',
              'tree', 'unique', 'fun', 'brownstone', 'community', 'hot',
              'double', 'ideal', 'half',
              'duplex', 'fireplace', 'rental', 'office', 'lounge', 
              'circle', 'decorated', 'bridge', 'accessible', 'soho',
              'terrace', 'accommodate', 'national', 'ill', 'golden', 'complete', 'cottage',
              'island',
              'fort', 'lax', 'warm', 'vibrant', 'classic', 'explore', 'dresser', 
              'smart', 'toaster', 'broadway', 
              'glass', 'summer', 'canyon', 'bungalow', 'extremely',
              'anywhere', 'roommate', 'designed',
              'airy', 'grove', 'popular', 'facing', 'courtyard', 
              'noise', 'used', 'echo', 'personal', 'however', 'cleaning',
              'young',
              'special',
              'diverse', 'field', 'century',
              'quaint', 'bunk',
              'organic', 'far', 'burbank',
              'boardwalk', 'eclectic', 'variety', 'club',
              'sweet', 'concierge', 'escape',
              'connected', 
              'chill', 'wooden',
              'party',
              'flexible', 'boulevard', 'nook', 'energy',
              'professionally',
              'gallery', 'vista', 
              'skylight', 'suitable', 'important', 'communal', 'cabin',
              'game',
              'tour', 'social', 'breeze',
              'quarter', 'rustic', 'hammock', 'detached', 'alone', 'empty', 
              'rented', 'freedom',
              'traditional', 'soft', 'standard', 
              'weekly', 'monthly', 'uptown',
              'treat', 'underground', 
              'sorry', 'barbecue', 'parlor',
              'finished', 'foggy',
              'retail', 'remote',
              'occasionally', 
              'business', 'hardwood']

for i in keep_words:
    words2[i] = words[i]



words2.to_csv('words_final.csv', index=False)
# =============================================================================
# Amenities Processing
# =============================================================================

import re
f = lambda x : [r for r in re.sub(r'[^,a-z0-9]','',x.lower()).split(',') if len(r) > 1]
amenities = pd.get_dummies(data['amenities'].map(f).apply(pd.Series).stack()).sum(level=0)

# Loop through amenities to make sure all values are either 0 or 1. If >1, change to 1
for col in amenities:
    amenities.loc[(amenities[col]>1), col] = 1

# Consolidate duplicate amenities
group = {'accessible_disable': ['accessibleheightbed', 'accessibleheighttoilet'], 
 'has_elevator': ['elevator', 'elevatorinbuilding'],
 'firm_matress': ['firmmatress', 'firmmattress'],
 'goodpathwaytofrontdoor': ['flatsmoothpathwaytofrontdoor',  'smoothpathwaytofrontdoor'],
 'well_lit_path': ['welllitpathtoentrance', 'pathtoentrancelitatnight'],
 'wideclearancetoshower_and_toilet': ['wideclearancetoshowerandtoilet', 'wideclearancetoshowertoilet'],
 }

        
for g in group:
    amenities[g] = 0
    amenities.loc[(amenities[group[g][0]] == 1) | (amenities[group[g][1]] == 1), g] = 1
    amenities = amenities.drop(group[g], axis=1)

# instances where dog, cats, otherpets all 0 and petsliveonproperty is 1 -> make otherpets 1
amenities[(amenities['dogs']==0) & (amenities['cats']==0) & (amenities['otherpets']==0) & (amenities['petsliveonthisproperty']==1)][['cats', 'dogs', 'otherpets', 'petsliveonthisproperty']].shape   
amenities[((amenities['dogs']==1) | (amenities['cats']==1) | (amenities['otherpets']==1)) & (amenities['petsliveonthisproperty']==1)][['cats', 'dogs', 'otherpets', 'petsliveonthisproperty']].shape   

# if  petsliveonthisproperty = 1, replace otherpets with 1 (done for 2,677 instances)
amenities.loc[((amenities['dogs']==0) & (amenities['cats']==0) & (amenities['otherpets']==0) & (amenities['petsliveonthisproperty']==1)), 'otherpets'] = 1

amenities = amenities.drop(['petsliveonthisproperty'], axis=1)


################## Washer Dryer ###############################
# washer, washerdryer, dryer

# divide into washer and dryer -> any instances where washerdryer = 1, but washer and dryer are 0
amenities[(amenities['washerdryer']==1) & (amenities['washer']==0) & (amenities['dryer']==0)][['washerdryer', 'washer', 'dryer']]
# No instances

#drop washerdryer
amenities = amenities.drop(['washerdryer'], axis=1)


# Drop other unneeded vars
amenities = amenities.drop(['translationmissingenhostingamenity49', 'translationmissingenhostingamenity50'], axis=1)

amenities =amenities.rename(columns = {'gym':'amenity_gym'})


# =============================================================================
# Import Other Data
# =============================================================================

amenities = pd.read_csv('amenities_clean.csv')
words = pd.read_csv('words_final.csv')

census_income = pd.read_csv('./data/ct_median_income.csv')
census_income = census_income.drop(['Unnamed: 0'], axis=1)


zillow = pd.read_csv('./data/Zip_MedianRentalPrice_AllHomes.csv',index_col='RegionName')['2017-12']
zillow.index = [str(zip) for zip in zillow.index]

rgb = pd.read_csv('withRgb.csv',encoding='iso-8859-1')

# -----------------------------------------------------------------------------
# Clean up census income - replace negative values with 0
census_income.loc[(census_income['ct_median_income'] < 0), 'ct_median_income'] = 0
# =============================================================================
# Calculate distance between city center and airbnb property
# =============================================================================

landmarks = {'Boston': {'latitude': 42.3431969605, 'longitude': -71.0726130429},  # copley
             'NYC': {'latitude': 40.758896, 'longitude': -73.985130},  #times square
             'SF': {'latitude': 37.8096506, 'longitude': -122.410249}, # fisherman's wharf
             'LA': {'latitude': 34.101166262, 'longitude': -118.337915315}, # hollywood(chinese theater)
             'DC': {'latitude': 38.897957, 'longitude': -77.036560}, # white house
             'Chicago': {'latitude': 41.8891, 'longitude': -87.626674}  # trump tower
        }


data['distance_calc'] = 0 

for index, row in data.iterrows():
    room = (row['latitude'], row['longitude'])
    landmark = (landmarks[next(iter(set([row['city']])&set(list(landmarks.keys()))))]['latitude'], 
                          landmarks[next(iter(set([row['city']])&set(list(landmarks.keys()))))]['longitude'])
    dist = vincenty(room, landmark).kilometers
    data.set_value(index, 'distance_calc', dist)
    #print(index)
    
    
# =============================================================================
# Add in info from google places api
# =============================================================================
places = pd.read_csv('./data/latlng_edited.csv', encoding = "ISO-8859-1")
places = pd.read_csv('latlng_edited.csv', encoding = "ISO-8859-1")


# Merge into dataset by latlng combination
data['lat_round'] = round(data['latitude'], 3)
data['lng_round'] = round(data['longitude'], 3)
data['latlngcoords'] = data['lat_round'].map(str) + ','+ data['lng_round'].map(str)

data = pd.merge(data, places, on='latlngcoords', how='left')

#drop latlng, latlngcoords
data = data.drop(['lat', 'latlngcoords', 'lng', 'lat_round', 'lng_round', 'places', 'placetype', 
                  'middle_places'], axis=1)


    
####################################################
############# Consolidate Data ############################
####################################################

    
data = pd.concat([data, amenities], axis=1)
data = pd.concat([data, words], axis=1)
data = data.merge(rgb[['id','meanG','meanR','meanB']],left_on='id',right_on='id')
data = data.merge(census_income[['id','ct_median_income']],left_on='id',right_on='id')
data = data.rename(columns={'ct_median_income_y': 'ct_median_income'})
data = data.rename(columns={'meanG_y': 'meanG', 'meanR_y': 'meanR', 'meanB_y': 'meanB'})

ords2.rename(columns={'great':'positive', 'inviting':'pos_sent', 
                                'best':'very_pos', 'museum':'attractions', 'ocean':'near_ocean',
                                'river':'near_small_water'})
data['ct_median_income'] = pd.to_numeric(data['ct_median_income'])
data['home_prices_zillow'] = data['zipcode'].map(zillow)

####################################################
############# Further Processing ############################
####################################################
data0 = data.copy()
data1 = data.drop(['neighbourhood', 'city_2', 'closest_city'], axis=1)
# this is data_0309_v2.csv

# Trying data1
data = pd.read_csv('data_0309_v2.csv', encoding = "ISO-8859-1")

# =============================================================================
# Data PreProcessing after merging
# =============================================================================
'''Property Type'''

data['property_type'] = data['property_type'].replace(['Bed & Breakfast', 'Bungalow', 'Villa', 'Guest suite'], 'Guesthouse')
data['property_type'] = data['property_type'].replace(['Dorm', 'Hut', 'Treehouse'], 'Other')

data['property_type'] = data['property_type'].replace(['Camper/RV', 'Timeshare', 'Cabin', 'Hostel', 'In-law', 
    'Boutique hotel', 'Boat', 'Serviced apartment', 'Tent', 'Castle', 'Vacation home', 'Yurt', 
    'Chalet', 'Earth House', 'Tipi', 'Train', 'Cave', 'Parking Space', 
    'Casa particular', 'Lighthouse', 'Island'], 'Other2')

### Dates
data['first_review'] = pd.to_datetime(data['first_review'])
data['last_review'] = pd.to_datetime(data['last_review'])
data['host_since'] = pd.to_datetime(data['host_since'])

data['now'] = datetime.datetime.now()
data['num_days_since_last_review'] = data['now'] - data['last_review']

data['num_days_since_last_review'] = [x.days for x in data['num_days_since_last_review']]

data['host_length'] = [x.days for x in (data['now'] - data['host_since'])]

data['host_response_rate'] = data['host_response_rate'].map(lambda x: float(x.split('%')[0])/100 if isinstance(x,str) else 0)


############# Map Dummy Vars #########################
data['instant_bookable'] = data['instant_bookable'].map({'f':0,'t':1})
data['host_has_profile_pic'] = data['host_has_profile_pic'].map({'f':0,'t':1})
data['host_identity_verified'] = data['host_identity_verified'].map({'f':0,'t':1})
data['cleaning_fee'] = data['cleaning_fee'].map({False:0,True:1})

############# Check Missing Vals  ###############################

# replace by 1 if value missing:
# bathrooms, bedrooms, beds
data.update(data[['bathrooms', 'bedrooms', 'beds']].fillna(1))

#data = data.reset_index()
# replace by 0 if missing:
# host_has_profile_pic, host_identity_verified, all amenities
data.update(data[(['host_has_profile_pic', 'host_identity_verified']+amenities.columns.tolist())].fillna(0))
#data.update(data[(['host_has_profile_pic', 'host_identity_verified'])].fillna(0))
#
#for x in amenities:
#    data[x].fillna(0, inplace=True)


# replace by median: review_scores_rating, meanG, meanR, meanB, ct_median_income, host_length
replace_median = ['review_scores_rating', 'meanG', 'meanR', 'meanB', 'ct_median_income', 'host_length', 'num_days_since_last_review']
for i in replace_median:
    data[i].fillna(data[i].median(), inplace=True)

# neighborhood missing 9K values
# replace by city if neighborhood missing
data['neighbourhood'].fillna(data['city'], inplace=True)

############# Encode ###############################
data = pd.get_dummies(data, columns=['property_type', 'room_type', 'bed_type', 
                                     'cancellation_policy', 'neighbourhood', 'city_2', 
                                     'closest_city'])
    
data = pd.get_dummies(data, columns=['property_type', 'room_type', 'bed_type', 
                                     'cancellation_policy'])

data = pd.get_dummies(data, columns=['city'])

############# Drop Vars ###############################
data = data.drop(['index', 'amenities', 'description', 'first_review', 'host_since', 
                    'last_review', 'name', 'thumbnail_url', 
                    'zipcode', 'now', 'latitude', 'longitude', 'city'], axis=1)
    
    
data = data.drop(['index', 'amenities', 'description', 'first_review', 'host_since', 
                    'last_review', 'name', 'thumbnail_url', 
                    'zipcode', 'now', 'latitude', 'longitude'], axis=1)

############# Add in  ###############################
data = pd.concat([data, words2], axis=1)




############# Normalize ###############################
data['log_income'] = np.log(ddataata['ct_median_income'])


# =============================================================================
# ############# Modeling #############################
# =============================================================================

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=True)

def prep(Data):
    test_clean = Data[Data['dataset']=='test']
    train2 = Data[Data['dataset']=='train']
    train2 = train2.drop(['dataset', 'id'], axis=1)
    # split out target and predictors
    y = train2['log_price']
    x = train2.drop(['log_price'], axis=1)
    return x, y, test_clean



############# XGBoost #############################
import xgboost as xgb
from xgboost import XGBRegressor

data = pd.read_csv('data_0309_v1.csv', encoding = "ISO-8859-1")

train = data[data.dataset == "train"]
test = data[data.dataset == 'test']

x_test = test.drop(['log_price', 'dataset'], axis=1)
x_test2 = test.drop(['log_price', 'dataset', 'id'], axis=1)

x_train = train.drop(['log_price', 'id', 'dataset'], axis=1)
y_train = train['log_price']



dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test2)


xgb_params = {
    'eta': 0.037,
    'max_depth': 10,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': np.mean(y_train),
    'silent': 0
}

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=242)
mode2 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=242)




print('RMSE:',mean_squared_error(mode2.predict(dtrain), y_train)**(1/2)) 



submission = pd.DataFrame(np.column_stack([x_test.id, z3]), columns = ['id','log_price'])
submission.to_csv("hz_submission2.csv", index = False)

# model 1

import operator
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.head(30).plot()
df.head(30).plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))


# model 2
importance2 = mode2.get_fscore()
importance2 = sorted(importance2.items(), key=operator.itemgetter(1), reverse=True)

df2 = pd.DataFrame(importance2, columns=['feature', 'fscore'])
#df2['fscore'] = df2['fscore'] / df2['fscore'].sum()
plt.figure()
df2.head(50).plot()
df2.head(50).plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

import pickle
pickle.dump(mode2,open('xg_model2.dat','wb'))