__author__ = 'satya'

import os
from PIL import Image
import numpy as np

def loadFaces():
    labels = 1

    trainingData ,trainingLabels = [] , []

    count = 0

    for directoryPath , directoryNames , fileNames in os.walk('/home/satya/Documents/dataset/faces/orl_faces'):
        for subDirectoryNames in directoryNames:
            #print subDirectoryNames

            subjectPath = os.path.join(directoryPath,subDirectoryNames)
            for eachFile in os.listdir(subjectPath):
                #print eachFile

                count+=1

                image = Image.open(os.path.join(subjectPath,eachFile))
                image = image.convert('L')

                trainingData.append(np.asarray(image))

                trainingLabels.append(labels)

            labels+=1


    return trainingData,trainingLabels


