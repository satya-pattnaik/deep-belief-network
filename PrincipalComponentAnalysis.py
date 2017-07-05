__author__ = 'satya'

import numpy as np

from LoadFaces import loadFaces

def PrincipalComponentAnalysis(trainingData):


    print 'trainingData------->SHAPE---->',trainingData.shape

    numberOfRows,numberOfColumns = trainingData.shape[0],trainingData.shape[1]

    print 'numberOfRows--->',numberOfRows
    print 'numberOfColumns--->',numberOfColumns

    mean = trainingData.mean(axis=0)

    trainingData_MeanSubtracted = trainingData - mean

    print 'trainingData_MeanSubtracted Shape--------------->',trainingData_MeanSubtracted.shape

    if numberOfRows > numberOfColumns:
        print 'numberOfRows > numberOfColumns'

        covarianceMatrix = np.dot( np.transpose(trainingData_MeanSubtracted) , trainingData_MeanSubtracted)
        #print covarianceMatrix

        print 'covarianceMatrix.shape-------------->>>',covarianceMatrix.shape

    else:
        print 'numberOfRows < numberOfColumns'

        covarianceMatrix = np.dot( trainingData_MeanSubtracted , np.transpose(trainingData_MeanSubtracted) )
        #print covarianceMatrix

        print 'covarianceMatrix.shape-------------->>>',covarianceMatrix.shape

    [eigenValues,eigenVectors] = np.linalg.eigh(covarianceMatrix)

    sortedIndices = np.argsort(-eigenValues)
    eigenValues = eigenValues[sortedIndices]
    eigenVectors = eigenVectors[:,sortedIndices]

    eigenValues = eigenValues[0:250] #250 PRINCIPAL COMPONENTS SELECTED
    eigenVectorsReduced = eigenVectors[:,0:250] #250 PRINCIPAL COMPONENTS SELECTED

    print '=======================PRINCIPAL COMPONENTS ANALYSED======================'

    return eigenVectorsReduced
    #return eigenVectors

def Flat(arrayToBeFlattened):

    flattenedTrainingData = []

    numberOfTrainingExamples = arrayToBeFlattened.shape[0]

    count = 0

    for eachRow in arrayToBeFlattened:
        eachFlattenedRow = np.asarray(eachRow).flatten()
        flattenedTrainingData.append(eachFlattenedRow)
        count += 1

    if numberOfTrainingExamples == count:
        print 'BINGO'
        return np.asarray(flattenedTrainingData)







def asRowMatrix(trainingData):

    mat = np.empty((0,trainingData[0].size))
    for eachRow in trainingData:
        mat = np.vstack((mat , np.asarray(eachRow).reshape(1,-1)))

    return mat

if __name__ == '__main__':

    trainingData,trainingLabels = loadFaces()

    #trainingData = np.asarray(asRowMatrix(trainingData))

    #flat_TrainingData = Flat(trainingData)
    #transformedFeatureVector = PrincipalComponentAnalysis(trainingData)

    #print 'transformedFeatureVector.shape------>>',transformedFeatureVector.shape

    print '(trainingLabels)------------>',trainingLabels






