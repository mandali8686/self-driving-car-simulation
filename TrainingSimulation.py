import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import cv2
import matplotlib
import imgaug
import tensorflow
from function import *

#import data
path='drive_data'
data=importData(path)

#data distribution
data=balanceData(data,display=False)

#Preparing data to a list to work with numpy
imagePath, steeringList=loadData(path,data)
print(imagePath[0],steeringList[0])

#Training data and validation data
xTrain,xVal,yTrain,yVal=train_test_split(imagePath,steeringList,test_size=0.2, random_state=5)
print("Total training images:",len(xTrain))
print("Total validation:", len(xVal))

#Preprocessing, all images augmented in training

model=createModel()
model.summary()

history=model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=20,epochs=20,validation_data=batchGen(xVal,yVal,100,0),validation_steps=20)

model.save('model.h5')
print('Model saved.')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()




