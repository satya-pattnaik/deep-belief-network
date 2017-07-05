__author__ = 'satya'

import theano as th
import numpy as np
import theano.tensor as T

class SoftMaxLayer(object):

    def __init__(self,outputFromRbm,dimensionsOfOutputFromRbm,numberOfNeurons,teacherOutput,randomNumberGenerator):

        activateInput = np.asarray((T.nnet.sigmoid(outputFromRbm)).eval())
        self.activatedInput = th.shared(activateInput,
                                        name='activatedInput',
                                        borrow=True)

        #----------------------------------------------------------------------------------------------

        self.teacherOutput = teacherOutput

        #----------------------------------------------------------------------------------------------
        weightsToSoftmaxLayer = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(dimensionsOfOutputFromRbm + numberOfNeurons)),
                high= float(6/float(dimensionsOfOutputFromRbm + numberOfNeurons)),
                size=(dimensionsOfOutputFromRbm,numberOfNeurons)
            ),
            dtype=th.config.floatX
        )
        self.weightsToSoftmaxLayer = th.shared(weightsToSoftmaxLayer,name='weights',borrow=True)

        #----------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------------

        biasToSoftmaxLayer = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(dimensionsOfOutputFromRbm + numberOfNeurons)),
                high= float(6/float(dimensionsOfOutputFromRbm + numberOfNeurons)),
                size=(numberOfNeurons,)
            ),
            dtype=th.config.floatX
        )
        self.biasToSoftmaxLayer = th.shared(biasToSoftmaxLayer,name='bias',borrow=True)

        #-----------------------------------------------------------------------------------------------

        #-----------------------------------BEFORE THE DOT PRODUCT--------------------------------------

        print 'activateInput.shape--->',activateInput.shape
        print 'weightsToSoftmaxLayer.shape--->',weightsToSoftmaxLayer.shape

        #------------------------------------------------------------------------------------------------

        inputToSoftmaxLayer = np.asarray((T.dot(activateInput,weightsToSoftmaxLayer)).eval() + biasToSoftmaxLayer)
        self.inputToSoftmaxLayer = th.shared(inputToSoftmaxLayer,name='inputToSoftmaxLayer',borrow=True)

        #-----------------------------------------------------------------------------------------------

        #-----------------------------------------SOFTMAX FUNCTION--------------------------------------


        def softMax(self,matrixToBeNormalized):

            print '=============INSIDE SOFTMAX============='

            matrix_Normalized = []

            for rowNumber,eachRow  in enumerate(matrixToBeNormalized):

                eachRow_exponential = np.exp(eachRow)
                eachRow_Normalized = eachRow_exponential/float(sum(eachRow_exponential))
                matrix_Normalized.append(eachRow_Normalized)
                print 'rowNumber-->',rowNumber,'eachRow-->',eachRow

            matrix_Normalized = np.asarray(matrix_Normalized)

            return matrix_Normalized


        #------------------------------------------------------------------------------------------------


        #-----------------STATE OF THE SOFTMAX LAYER AFTER APPLYING THE SOFTMAX FUNCTION-----------------

        stateOfSoftMaxLayer = softMax(self,inputToSoftmaxLayer)

        #------------------------------------------------------------------------------------------------

        #-------------------------------------ARGMAX FUNCTION--------------------------------------------
        def argMax(self,matrix):
            indexes = []
            for eachVector in matrix:
                max = 0
                indexOfMaxElement = 0
                for index,eachElement in enumerate(eachVector):
                    if eachElement>max:
                        max = eachElement
                        indexOfMaxElement = index

                indexes.append(indexOfMaxElement)

            return indexes

        #-----------------------------------------------------------------------------------------------

        #-----------------------------------PREDICTION OF THE SOFTMAX LAYER------------------------------

        #labelPrediction = T.argmax(stateOfSoftMaxLayer,axis=1)
        labelPrediction = argMax(self,stateOfSoftMaxLayer)

        #-----------------------------------DELTA OF THE OUTPUT LAYER------------------------------------

        print '===============DELTA OF THE OUTPUT LAYER================='
        self.delta = self.teacherOutput - labelPrediction

        #------------------------------------------------------------------------------------------------















