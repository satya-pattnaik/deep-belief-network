__author__ = 'satya'


import cPickle
import gzip
import os
import theano as th
import theano.tensor as T
import numpy as np

def loadData(pathOfDataset):

    '''-----------------PARAMETER-------------------------
        TYPE : STRING
        PARAM FUNCTION: PATH OF THE DATASET
        --------------------------------------------------'''

    #########################################
    #            LOAD DATASET               #
    #########################################

    dataFile = gzip.open(pathOfDataset,'rb')
    trainingSet,validationSet,testSet = cPickle.load(dataFile)
    dataFile.close()

################################################################################################

    #------------------------------------LOADING A SHARED DATASET------------------------------------------
    def sharedDataset(data_inputX_classLabelY , borrow = True):

        data_inputX,data_classLabelY = data_inputX_classLabelY

        shared_inputX = th.shared(np.asarray(data_inputX,dtype=th.config.floatX),
                                borrow=borrow)

        shared_classlabelY = th.shared(np.asarray(data_classLabelY,dtype=th.config.floatX),
                                borrow=borrow)

        return shared_inputX,T.cast(shared_classlabelY,'int32')

    #-------------------------------------------------------------------------------------------------------

    trainingSet_inputX,trainingSet_classlabelY = sharedDataset(trainingSet)
    testSet_inputX,testSet_classLabelY = sharedDataset(testSet)
    validationSet_inputX,validationSet_classlabelY = sharedDataset(validationSet)

    returnValue = [(trainingSet_inputX,trainingSet_classlabelY) , (testSet_inputX,testSet_classLabelY) , (validationSet_inputX,validationSet_classlabelY)]

    return returnValue
    #-----------------DEBUGGING--------------------------

    #print trainingSet
    #print validationSet
    #print testSet

    #----------------------------------------------------
#------------------------------EXAMPLE--------------------

#x = loadData(os.path.join('/home/satya/Documents/dataset/MNIST','mnist.pkl.gz'))
#print x


