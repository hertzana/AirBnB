# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:09:10 2018

@author: Nathan Hearnsberger
"""


from PIL import Image
import requests
import os

##RGB Analysis - see image download script below
##BEWARE: Takes a long time to get mean RGB for dataset

def getBrightnessForImage(id=None,url=None,useImageData=True):
    if useImageData:
        path = 'images/{}.jpg'.format(id)
        if os.path.exists(path):
            image = Image.open(path)
        else:
            return(row)
    else:
        image = Image.open(requests.get(url,stream=True).raw)
    RGBs = []
    for x in range(image.width):
        for y in range(image.height):
            RGBs.append(sum(image.getpixel((x,y)))/3)
    return(sum(RGBs)/len(RGBs))   

def getRGBForImage(row,useImageData=True):
    
    if useImageData:
        id = row['id']
        path = 'images/{}.jpg'.format(id)
        if os.path.exists(path):
            image = Image.open(path)
        else:
            return(row)
    else:
        url = row['thumnail_url']
        image = Image.open(requests.get(url,stream=True).raw)
    RGBs = []
    for x in range(image.width):
        for y in range(image.height):
            rgb = image.getpixel((x,y)) 
            RGBs.append(rgb)
    transposed = np.array(RGBs).T
    meanR, meanG, meanB = np.mean(transposed[0]),np.mean(transposed[1]),np.mean(transposed[2])
    
    row['meanR'] = meanR
    row['meanG'] = meanG
    row['meanB'] = meanB
    
    return(row)


data = data.apply(getRGBForImage,axis=1)