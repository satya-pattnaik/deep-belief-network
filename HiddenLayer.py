__author__ = 'satya'

__author__ = 'satya'

import numpy as np
import theano as th
import theano.tensor as T

class HiddenLayer(object):

    def __init__(self,numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer,randomNumberGenerator,activatedInput=None):

        hiddenLayerInput = th.shared(activatedInput.eval(),name='hiddenLayerInput',borrow=True)
        #---------------INPUT FROM PREVIOUS LAYER-----------------------
        self.input = hiddenLayerInput                                  #
        #---------------------------------------------------------------

        #----------------------------------------------------------------
        self.numberOfHiddenUnits = numberOfHiddenUnits
        self.numberOfHiddenUnits_NextHiddenLayer = numberOfHiddenUnits_NextHiddenLayer
        #----------------------------------------------------------------

        weights = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(numberOfHiddenUnits + numberOfHiddenUnits_NextHiddenLayer)),
                high= float(6/float(numberOfHiddenUnits + numberOfHiddenUnits_NextHiddenLayer)),
                size=(numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer)
            ),
            dtype=th.config.floatX
        )
        #------------WEIGHT MATRIX INITIALIZED BY UNIFORMLY DISTRIBUTED RANDOM VALUES-------------
        self.weights = th.shared(weights,name='weights',borrow=True)                             #                                      #
        #-----------------------------------------------------------------------------------------

        bias = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(numberOfHiddenUnits + numberOfHiddenUnits_NextHiddenLayer)),
                high= float(6/float(numberOfHiddenUnits + numberOfHiddenUnits_NextHiddenLayer)),
                size=(1,)
            ),
            dtype=th.config.floatX
        )
        #-------------BIAS MATRIX INITIALIZED BY UNIFORMLY DISTRIBUTED RANDOM VALUES--------------
        self.bias = th.shared(bias,name='bias',borrow=True)                                      #
        #-----------------------------------------------------------------------------------------

        #print '-----------DEBUG:COMPUTING DOT PRODUCT OF INPUT AND WEIGHTS IN HIDDEN LAYER-----------'
        #print 'input-->SHAPE-->',self.input.shape.eval(),'---weights--->SHAPE--->',self.weights.shape.eval()

        inputForNextHiddenLayer_Dot = T.dot(self.input.eval() , self.weights.eval()) + self.bias.eval()
        #print 'inputForNextHiddenLayer SHAPE----------->',inputForNextHiddenLayer_Dot.shape.eval()
        #------------INPUT FOR THE HIDDEN LAYER FROM THE INPUT LAYER---------------------------------------------------
        self.inputForNextHiddenLayer = th.shared(inputForNextHiddenLayer_Dot.eval(),name='inputForNextHiddenLayer',borrow=True)                                                                                        #
        #--------------------------------------------------------------------------------------------------------------

class HyperbolicTangentLayer(HiddenLayer):

    def __init__(self,numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer,randomNumberGenerator,inputToBeActivated=None):

        #-------------------------------------------------CALL SUPERCLASS __init__ METHOD----------------------------------------------------
        super(HiddenLayer,self).__init__(inputToBeActivated,numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer,randomNumberGenerator)  #
        #------------------------------------------------------------------------------------------------------------------------------------

    def setInput(self,inputToBeActivated):
        #----------------USE THE HYPERBOLIC TANGENT FUNCTION TO GET A NON-LINEAR REPRESENTATION OF THE INPUTS------------
        activatedInput = T.tanh(inputToBeActivated)                                                                     #
        #----------------------------------------------------------------------------------------------------------------
        HiddenLayer.input = th.shared(activatedInput,name='input')                                                      #
        #----------------------------------------------------------------------------------------------------------------

        gradientOfInputForNextHiddenLayer = float(1) - T.dot(activatedInput,T.transpose(activatedInput))
        #--------------------------GRADIENT OF THE INPUT FOR THE NEXT LAYER---------------------------------------------------------------------------------
        self.gradientOfInputForNextHiddenLayer = th.shared(gradientOfInputForNextHiddenLayer,name='gradientOfInputForNextHiddenLayer',borrow=True)         #
        #---------------------------------------------------------------------------------------------------------------------------------------------------

    def setDeltaHiddenLayer(self,weightDotDelta):
        if self.gradientOfInputForNextHiddenLayer.shape[0] == weightDotDelta.shape[0]:

            for eachDerivedValue,eachRowInWeightDotDelta in zip(np.nditer(self.gradientOfInputForNextHiddenLayer),xrange(0,weightDotDelta.shape[0] - 1)):
                weightDotDelta[eachRowInWeightDotDelta] = weightDotDelta[eachRowInWeightDotDelta] * eachDerivedValue

            #-------------------------------STORE THE DELTA VALUES FOR THE HIDDEN LAYER--------------
            self.deltaHiddenLayer = th.shared(weightDotDelta,name='deltaHiddenLayer',borrow=True)   #
            #----------------------------------------------------------------------------------------

        else:
            print 'PARAMETERS NOT MATCHING INSIDE setDeltaHiddenLayer'

class SigmoidLayer(HiddenLayer):

    def __init__(self,numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer,randomNumberGenerator,inputToBeActivated=None):

        print '======================CREATE A HIDDEN SIGMOID LAYER======================='

        #--------------------SQUASH THE INPUT USING SIGMOID FUNCTION------------------------------------------
        #activatedInput = T.nnet.sigmoid(inputToBeActivated)                                                  #
        #-----------------------------------------------------------------------------------------------------

        #----------------------------------CALL SUPERCLASS __init__ METHOD---------------------------------------------------------------
        super(SigmoidLayer,self).__init__(numberOfHiddenUnits,numberOfHiddenUnits_NextHiddenLayer,randomNumberGenerator,inputToBeActivated)
        #--------------------------------------------------------------------------------------------------------------------------------

    def setInput(self,inputToBeActivated):

        print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
        #----------------USE THE SIGMOID FUNCTION TO GET A NON-LINEAR REPRESENTATION OF THE INPUTS------------
        activatedInput = T.nnet.sigmoid(inputToBeActivated)                                                             #
        #----------------------------------------------------------------------------------------------------------------
        HiddenLayer.input = th.shared(activatedInput.eval(),name='input',borrow=True)                                    #
        #----------------------------------------------------------------------------------------------------------------

        gradientOfInputForNextHiddenLayer = (activatedInput.eval())*(float(1) - activatedInput.eval())

        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^WE ARE HERE WHERE WE WANT^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print 'inputToBeActivated.shape====>>',inputToBeActivated.shape
        print 'gradientOfInputForNextHiddenLayer.shape====>>>',gradientOfInputForNextHiddenLayer.shape
        #-----------------------------------------------------GRADIENT OF THE INPUT FOR THE NEXT LAYER--------------------------------------------------
        self.gradientOfInputForNextHiddenLayer = th.shared(gradientOfInputForNextHiddenLayer,name='gradientOfInputForNextHiddenLayer',borrow=True)     #
        #-----------------------------------------------------------------------------------------------------------------------------------------------

    def setDeltaHiddenLayer(self,weightDotDelta):

        print '==================SETTING DELTA OF HIDDEN LAYER==============='

        gradientOfInputForNextHiddenLayer_Shape = self.gradientOfInputForNextHiddenLayer.shape.eval()
        weightDotDelta_Shape = weightDotDelta.shape.eval()

        print 'gradientOfInputForNextHiddenLayer_Shape[0]-->',gradientOfInputForNextHiddenLayer_Shape[0]
        print 'gradientOfInputForNextHiddenLayer_Shape[1]-->',gradientOfInputForNextHiddenLayer_Shape[1]
        print 'weightDotDelta_Shape[0]-->',weightDotDelta_Shape[0]
        print 'weightDotDelta_Shape[1]-->',weightDotDelta_Shape[1]

        if (( gradientOfInputForNextHiddenLayer_Shape[0] == weightDotDelta_Shape[0] ) and
            ( gradientOfInputForNextHiddenLayer_Shape[1] == weightDotDelta_Shape[1] ) ):

            print '==================PARAMETERS MATCHING=================='
            '''weightDotDelta_Temp = np.asarray(weightDotDelta.eval())

            for eachDerivedValue,eachRowInWeightDotDelta in zip(np.nditer(self.gradientOfInputForNextHiddenLayer.eval()),xrange(0,weightDotDelta_Temp.shape[0])):

                weightDotDelta_Temp[eachRowInWeightDotDelta] = weightDotDelta_Temp[eachRowInWeightDotDelta] * eachDerivedValue'''

            deltaHidden = np.multiply(self.gradientOfInputForNextHiddenLayer.eval(),
                                      weightDotDelta.eval())

            #-------------------------------STORE THE DELTA VALUES FOR THE HIDDEN LAYER--------------
            self.deltaHiddenLayer = th.shared(deltaHidden,name='deltaHiddenLayer',borrow=True)   #
            #----------------------------------------------------------------------------------------

            print 'DELTA HIDDEN LAYER SHAPE------>',self.deltaHiddenLayer.shape.eval()

            print '====================DELTA FOR HIDDEN LAYER SET===================='

        else:
            print '======PARAMETERS NOT MATCHING======='

    def activate(self):
        print '==========ACTIVATING INPUTS============='
        #print '-----------DEBUG:COMPUTING DOT PRODUCT OF INPUT AND WEIGHTS IN HIDDEN LAYER-----------'
        #print 'input-->SHAPE-->',self.input.shape.eval(),'---weights--->SHAPE--->',self.weights.shape.eval()

        inputForNextHiddenLayer_Dot = T.dot(self.input.eval() , self.weights.eval()) + self.bias.eval()
        #print 'inputForNextHiddenLayer SHAPE----------->',inputForNextHiddenLayer_Dot.shape.eval()
        #------------INPUT FOR THE HIDDEN LAYER FROM THE INPUT LAYER---------------------------------------------------
        self.inputForNextHiddenLayer = th.shared(inputForNextHiddenLayer_Dot.eval(),name='inputForNextHiddenLayer',borrow=True)                                                                                        #
        #--------------------------------------------------------------------------------------------------------------

    def setWeights(self,weightMatrix):
        print 'WEIGHTS UPDATED'
        self.weights = th.shared(weightMatrix,name='weights',borrow=True)

    def setBias(self,biasMatrix):
        print 'BIAS UPDATED'
        self.bias = th.shared(biasMatrix,name='bias',borrow=True)


