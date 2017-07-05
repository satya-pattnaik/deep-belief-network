__author__ = 'satya'

import numpy as np
import theano as th
import theano.tensor as T

class InputLayer(object):

    def __init__(self,inputData,dimensionalityOfInput,numberOfHiddenUnits_HiddenLayer,randomNumberGenerator):

        print '=================CREATING AN INPUT LAYER================='

        #------------------------INPUT DATA-------------------------
        self.input = th.shared(inputData.eval(),name='input',borrow=True)     #
        #-----------------------------------------------------------


        #-------------------------------------------------------
        self.numberOfHiddenUnits_NextHiddenLayer=numberOfHiddenUnits_HiddenLayer
        #-------------------------------------------------------

        weights = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(dimensionalityOfInput + self.numberOfHiddenUnits_NextHiddenLayer)),
                high= float(6/float(dimensionalityOfInput + self.numberOfHiddenUnits_NextHiddenLayer)),
                size=(dimensionalityOfInput,self.numberOfHiddenUnits_NextHiddenLayer)
            ),
            dtype=th.config.floatX
        )
        #------------WEIGHT MATRIX INITIALIZED BY UNIFORMLY DISTRIBUTED RANDOM VALUES-------------
        self.weights = th.shared(weights,name='weights',borrow=True)                             #                                      #
        #-----------------------------------------------------------------------------------------

        bias = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(dimensionalityOfInput + self.numberOfHiddenUnits_NextHiddenLayer)),
                high= float(6/float(dimensionalityOfInput + self.numberOfHiddenUnits_NextHiddenLayer)),
                size=(1)
            ),
            dtype=th.config.floatX
        )
        #-------------BIAS MATRIX INITIALIZED BY UNIFORMLY DISTRIBUTED RANDOM VALUES--------------
        self.bias = th.shared(bias,name='bias',borrow=True)                                      #                                   #
        #-----------------------------------------------------------------------------------------

        print 'self.input.shape------->',self.input.shape.eval()
        print 'self.weights.shape------>',self.weights.shape.eval()

        inputForHiddenLayer = T.dot(self.input.eval() , self.weights.eval()) + self.bias.eval()

        print 'inputForNextHiddenLayer SHAPE------>',inputForHiddenLayer.shape.eval()
        #------------INPUT FOR THE HIDDEN LAYER FROM THE INPUT LAYER---------------------------------------
        self.inputForNextHiddenLayer = th.shared(inputForHiddenLayer.eval(),name='inputForNextHiddenLayer',borrow=True)  #                                                                                       #
        #--------------------------------------------------------------------------------------------------

    def activate(self):

        inputForHiddenLayer = T.dot(self.input.eval(),self.weights.eval()) + self.bias.eval()
        print '//////////////////////////////////////////ACTIVATING INPUTS///////////////////////////////////////////////////////////////'
        print 'inputForHiddenLayer.shape=====>>>',inputForHiddenLayer.shape.eval()
        self.inputForNextHiddenLayer=th.shared(inputForHiddenLayer.eval(),name='inputForNextHiddenLayer',borrow=True)

        print '===========ACTIVATION DONE AT INPUT LAYER============'

    def setWeights(self,weightMatrix):
        print 'WEIGHTS UPDATED'
        self.weights = th.shared(weightMatrix,name='weights',borrow=True)

    def setBias(self,biasMatrix):
        print 'BIAS UPDATED'
        self.bias = th.shared(biasMatrix,name='bias',borrow=True)

    def setNewInput(self,inputData):
        self.input = th.shared(inputData.eval(),name='input',borrow=True)


