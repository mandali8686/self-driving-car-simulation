import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

def getName(filepath):
    return filepath.split('\\')[-1]

def importData(path):
    col=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=col)
    #print(getName(data['Center'][0]))
    data['Center']=data['Center'].apply(getName)
    #print(data.head())
    print("Total:",data.shape[0])
    return data

def balanceData(data,display=True):
    nBins=31
    samplesPerBin=500
    hist,bins=np.histogram(data['Steering'],nBins)
    
    if display:
        center=(bins[:-1]+bins[1:])*0.5
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    #removing redundant data
    removeIndex=[]
    for j in range(nBins):
        binDatalist=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i]>=bins[j] and data['Steering'][i] <= bins[j+1]:
                binDatalist.append(i)
        binDatalist= shuffle(binDatalist)
        binDatalist= binDatalist[samplesPerBin:]
        removeIndex.extend(binDatalist)
    print("remove images: ",len(removeIndex))
    data.drop(data.index[removeIndex],inplace=True)
    print("Remaining images:",len(data))
    #display new data
    if display:
        hist, _ =np.histogram(data['Steering'],nBins)
        #center=(bins[:-1]+bins[1:])*0.5
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    return data

def loadData(path,data):
    imagePath=[]
    steeringList=[]
    for i in range(len(data)):
        indexedData=data.iloc[i]
        #print(indexedData)
        imagePath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steeringList.append(float(indexedData[3]))
    imagePath=np.asarray(imagePath)
    steeringList=np.asarray(steeringList)
    return imagePath,steeringList

def augmentImage(imagePath,steeringList):
    image=mpimg.imread(imagePath)
    if np.random.rand()<0.5:
        panel =iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        image=panel.augment_image(image)
    if np.random.rand()<0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        image=zoom.augment_image(image)
    if np.random.rand()<0.5:
        brightness=iaa.Multiply((0.5,1.2))
        image=brightness.augment_image(image)
    if np.random.rand()<0.5:
        image=cv2.flip(image,1)
        steeringList=-steeringList
    return image,steeringList

def preProcessing(image):
    image=image[60:135,:,:]
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    image=cv2.GaussianBlur(image,(3,3),0)
    image=cv2.resize(image,(200,66))
    image=image/255
    return image

#Send to model by batch
def batchGen(imagesPath,steeringList,batchSize, trainflag):
    while True:
        imgBatch=[]
        steeringBatch=[]
        for i in range(batchSize):
            index=random.randint(0,len(imagesPath)-1)
            if trainflag:
                image,steering=augmentImage(imagesPath[index],steeringList[index])
            else:
                image=mpimg.imread(imagesPath[index])
                steering=steeringList[index]
            image=preProcessing(image)
            imgBatch.append(image)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))

def createModel():
    model=Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001),loss='mse')
    return model

    

    

    
