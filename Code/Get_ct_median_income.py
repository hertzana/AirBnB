# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:09:45 2018

@author: Nathan Hearnsberger
"""

import os
import requests
import zipfile
import io
import shapefile
from bs4 import BeautifulSoup
import requests
from shapely.geometry import shape as Shape, Point

###Example of spatial feature extraction
###Finds the median household income of the Census tract from listing lng/lat 
###Takes a while to reload data


#Scrape zip file links to dict {stateName:zipFileURL} for census tract shape files
censusShapeURL = 'https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html'
soup = BeautifulSoup(requests.get(censusShapeURL).content,'lxml')
ctShapefiles = {option.text.strip():option.get('value') for option in soup.find(id='ct2016m').findAll('option')}

#Could replace this with a less hard-coded method
cityToState = {'NYC':'New York', 
               'SF':'California', 
               'DC':'District of Columbia', 
               'LA': 'California', 
               'Chicago': 'Illinois', 
               'Boston': 'Massachusetts'}

ctDict = {}
#Download all shapefiles needed, unzip, and add to a shapefile dict {censusTract:shapeFile}
for zipFile in data.city.map(cityToState).map(ctShapefiles).unique():
    shapefilePath = './data/census_tract_shapefiles/{}.shp'.format(zipFile.split('/')[-1].split('.')[0])
    if os.path.exists(shapefilePath): 
        print('{} already exists, using local copy'.format(shapefilePath))
    else:
        print('downloading {} ... '.format(zipFile))
        r = requests.get(zipFile)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./data/census_tract_shapefiles')
        
    shape = shapefile.Reader(shapefilePath)
    ctDict.update({ feature.record[3]: Shape(feature.shape) for feature in shape.shapeRecords()})
  
#Function that will return the census tract for given coordinates
def getFeatureforPoint(shapeDict,lng,lat):
    point = Point(lng,lat)
    for feature, shape in shapeDict.items():
        if shape.contains(point):
            return(feature)

        

#Loop through and save census data for each state 
ctIncomePath = './data/ct_income.csv'
if os.path.exists(ctIncomePath): 
    print('{} already exists, using local copy'.format(ctIncomePath))
    pd.read_csv(ctIncomePath)
else:
    ctIncome = pd.DataFrame()
    for i in range(1,57):
        url = 'https://api.census.gov/data/2016/acs/acs5?get=NAME,B19013_001E&for=tract:*&in=state:{}'.format(str(i).zfill(2))
        try:
            df = pd.DataFrame(requests.get(url).json())
        except:
            next

        ctIncome = pd.concat([ctIncome,df])

    #return series of tract to median income
    ctIncome.columns = ctIncome.iloc[0]
    ctIncome = ctIncome.drop_duplicates().iloc[1:]
    ctIncome['full_tract_name'] = '1400000US' + ct_income['state'] + ct_income['county'] + ct_income['tract']
    ctIncome = ctIncome.set_index('full_tract_name')['B19013_001E'] 
    ctIncome.to_csv(ctIncomePath)

##Should store census tract locations to id mapping also
data.apply(lambda row:\
           getFeatureforPoint(ctDict,row['longitude'],row['latitude']),axis = 1)\
           .map(ctIncome)
    
data[['id','ct_median_income']].to_csv('./data/features/ct_median_income.csv')