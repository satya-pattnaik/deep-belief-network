__author__ = 'satya'


import numpy as np
import theano as th
import theano.tensor as T

class Connect(object):

    def __init__(self,fromHiddenLayer,toHiddenLayer):

        #------------------CONNECT THE FOLLOWING LAYERS------------
        self.fromHiddenLayer = fromHiddenLayer                    #
        self.toHiddenLayer = toHiddenLayer                        #
        #----------------------------------------------------------


        if(self.fromHiddenLayer.__class__.__name__ == 'InputLayer'):
            if not (self.fromHiddenLayer.numberOfHiddenUnits_NextHiddenLayer == self.toHiddenLayer.numberOfHiddenUnits):
                print 'PARAMETERS NOT MATCHING INSIDE INPUT LAYER'
                exit(1)
            #--------------OUTPUT OF INPUT LAYER IS THE INPUT FOR THE NEXT LAYER-----------
            self.inputForNextLayer = self.fromHiddenLayer.inputForNextHiddenLayer       #
            #----------------------------------------------------------------------------


        else:
            if not(self.fromHiddenLayer.numberOfHiddenUnits_NextHiddenLayer == self.toHiddenLayer.numberOfHiddenUnits):
                print 'PARAMETERS NOT MATCHING'
                exit(1)

            #--------------OUTPUT OF ONE LAYER IS THE INPUT FOR THE NEXT LAYER-----------
            self.inputForNextLayer = self.fromHiddenLayer.inputForNextHiddenLayer       #
            #----------------------------------------------------------------------------

        print '((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((('
        print 'self.inputForNextLayer.shape---->>>>',self.inputForNextLayer.shape.eval()

        #-------------SET THE INPUT FOR THE NEXT LAYER-----------
        self.toHiddenLayer.setInput(self.inputForNextLayer.eval())     #
        #--------------------------------------------------------

        print '***********************************************THIS IS WHERE I WANT TO TEST************************************************************'

        #print 'self.fromHiddenLayer.__class__.__name__----->>',self.fromHiddenLayer.__class__.__name__
        if ((self.fromHiddenLayer.__class__.__name__ == 'InputLayer') or(self.fromHiddenLayer.__class__.__name__ == 'SigmoidLayer')):
            print '===================SETTING THE WEIGHT MULTIPLIED DELTA(HIDDEN LAYER) FOR HIDDEN LAYER========================'

    def set_WeightDotDelta():
        if (self.fromHiddenLayer.__class__.__name__ == 'InputLayer'):
            print '++++++++++++++++++++++++++NOT NEEDED++++++++++++++++++++++++++++'
            exit(1)

        weightMatrixShape_fromHiddenLayer = self.fromHiddenLayer.weights.shape
        deltaMatrixShape_toHiddenLayer = self.toHiddenLayer.deltaHiddenLayer.shape

        #print 'weightMatrixShape_fromHiddenLayer--->>',weightMatrixShape_fromHiddenLayer
        #print 'deltaMatrixShape_toHiddenLayer--->>',deltaMatrixShape_toHiddenLayer

        if weightMatrixShape_fromHiddenLayer[1] == deltaMatrixShape_toHiddenLayer[1]:
            print 'DOT PRODUCT POSSIBLE'
            self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer = T.dot(self.toHiddenLayer.deltaHiddenLayer , np.transpose(self.fromHiddenLayer.weights))
            self.fromHiddenLayer.setDeltaHiddenLayer(self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer)
        else:
            print 'DOT PRODUCT NOT POSSIBLE'



    def setGradientMatrix(self):

        print '==============================CALCULATING GRADIENT FOR THE HIDDEN LAYER================================'

        #----------------COMPUTE THE GRADIENTS OF THE ERROR TERM WITH RESPECT TO WEIGHTS--------------
        deltaValues = self.toHiddenLayer.deltaHiddenLayer
        activatedInputValues = self.fromHiddenLayer.input
        transposedactivatedInputValues = activatedInputValues.T

        #print 'deltaValues---------------->>>>>',deltaValues.shape.eval()
        #print 'activatedInputValues-------->>>>',activatedInputValues.shape.eval()
        #print 'transposedactivatedInputValues.shape---->>>>',transposedactivatedInputValues.shape.eval()
        #----------------COMPUTATION OF THE GRADIENT MATRIX-----------------
        gradMatrix = []

        for eachActivatedInputValue in transposedactivatedInputValues.eval():
            for eachDeltaValue in deltaValues.eval():
                gradMatrix.append(eachActivatedInputValue*eachDeltaValue)

        gradMatrix = np.asarray(gradMatrix)

        print 'LENGTH OF GRAD MATRIX====>>>',(gradMatrix).shape

        self.gradientMatrix = th.shared(
            value=(gradMatrix),
            name='gradientMatrix',
            borrow = True
        )
        #-------------------------------------------------------------------

    '''def setJacobianMatrix(self):
        jacMatrix = []

        for eachColumn in xrange(0,self.gradientMatrix.shape[1]):
            jacMatrix.append(self.gradientMatrix[eachColumn])

        self.jacobianMatrix = th.shared(value=jacMatrix,
                                        name='jacobianMatrix',
                                        borrow=True)'''

class ConnectOutputLayer(object):

    def __init__(self,lastHiddenLayer,outputLayer):

        #--------CONNECT THE FOLLOWING TWO LAYERS----------
        self.lastHiddenLayer = lastHiddenLayer
        self.outputLayer = outputLayer
        #--------------------------------------------------

        print '-------------------------DEBUGGING INSIDE class ConnectOutputLayer---------------------------'
        #print 'self.lastHiddenLayer.numberOfHiddenUnits_NextHiddenLayer---->',self.lastHiddenLayer.numberOfHiddenUnits_NextHiddenLayer
        #print 'self.outputLayer.numberOfOutputUnits----->>',self.outputLayer.numberOfOutputUnits

        if not (self.lastHiddenLayer.numberOfHiddenUnits_NextHiddenLayer == self.outputLayer.numberOfOutputUnits):
            print 'PARAMETERS NOT MATCHING INSIDE OUTPUT LAYER'
            exit(1)

        #--------------------INPUT FOR NEXT LAYER-------------------------------#
        self.inputForNextLayer = self.lastHiddenLayer.inputForNextHiddenLayer
        #-----------------------------------------------------------------------#

        #--------------------DELTA FOR THE OUTPUT LAYER-------------------------#
        self.deltaForOutputlayer = self.outputLayer.delta_Output
        #-----------------------------------------------------------------------#

        #------------------------------------------------------------------------------------------------------------------------------------------------#
        if (self.lastHiddenLayer.__class__.__name__ == 'SigmoidLayer'):

            print '===================SETTING THE WEIGHT MULTIPLIED DELTA(OUTPUT LAYER) FOR HIDDEN LAYER========================'

            weightMatrixShape_fromHiddenLayer = self.lastHiddenLayer.weights.shape
            deltaMatrixShape_toHiddenLayer = self.outputLayer.delta_Output.shape

            #print 'weightMatrixShape_fromHiddenLayer SHAPE---->>',weightMatrixShape_fromHiddenLayer.eval()
            #print 'deltaMatrixShape_toHiddenLayer SHAPE---->>',deltaMatrixShape_toHiddenLayer

            if weightMatrixShape_fromHiddenLayer[1].eval() == deltaMatrixShape_toHiddenLayer[1]:
                print '===========DOT PRODUCT POSSIBLE============'
                self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer = T.dot(self.outputLayer.delta_Output,np.transpose(self.lastHiddenLayer.weights.eval()))

                #print 'self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer.shape--->',self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer.shape.eval()
                self.lastHiddenLayer.setDeltaHiddenLayer(self.weightFromHiddenLayer_Dot_DeltaToHiddenLayer)


            else:
                print 'DOT PRODUCT NOT POSSIBLE'

    #------------------------------------------------------------------------------------------------------------------------------------------------#
    def setGradientMatrix(self):

        print '======================SETTING THE GRADIENT FOR OUTPUT CONNECTION=========================='

        #----------------COMPUTE THE GRADIENTS OF THE ERROR TERM WITH RESPECT TO WEIGHTS--------------
        deltaValues = self.outputLayer.delta_Output
        activatedInputValues = self.lastHiddenLayer.input
        transposedactivatedInputValues =activatedInputValues.T

        #print 'deltaValues-->Shape-->',deltaValues.shape
        #print 'transposedactivatedInputValues.shape--->',transposedactivatedInputValues.shape.eval()

        gradientValuesCollected = []
        #----------------COMPUTATION OF THE GRADIENT MATRIX---------------

        #for eachHiddenNeuron in xrange(transposedactivatedInputValues.shape[0].eval()):
        #    deltaValuesCollected.append(deltaValues)

        #deltaValuesCollected=np.asarray(deltaValuesCollected)

        #print 'deltaValuesCollected.shape---->',deltaValuesCollected.shape

        for eachActivatedInputValue in transposedactivatedInputValues.eval():
            #gradient = []
            for eachdeltaValue in deltaValues:
                gradientValuesCollected.append(eachActivatedInputValue*eachdeltaValue)
            #gradientValuesCollected.append(gradient)

        gradMatrix = np.asarray(gradientValuesCollected)
        #print 'gradMatrix.shape---->',gradMatrix.shape
        self.gradientMatrix = th.shared(
            value=np.asarray(gradMatrix),
            name='gradientMatrix',
            borrow = True
        )
        #print 'GRADIENT MATRIX SHAPE OUTER LAYER--->',self.gradientMatrix.shape.eval()
        #-----------------------------------------------------------------

    def setInputForOutputLayer(self):
        self.outputLayer.inputToBeActivated = self.lastHiddenLayer.inputForNextHiddenLayer




