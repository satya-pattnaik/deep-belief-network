__author__ = 'satya'

__author__ = 'satya'

from RestrictedBoltzmannMachines import RestrictedBoltzmannMachines
from Output_SoftmaxLayer import SoftMaxLayer
from PrincipalComponentAnalysis import PrincipalComponentAnalysis,asRowMatrix
from LoadFaces import loadFaces
import numpy as np
import theano as th
################################################
from InputLayer import InputLayer
from HiddenLayer import SigmoidLayer
from Connect import Connect,ConnectOutputLayer
from OutputLayer import SigmoidOutputLayer
################################################

trainingData,traininglabels = loadFaces()
trainingData = np.asarray(asRowMatrix(trainingData))
transformedFeatureVector = PrincipalComponentAnalysis(trainingData)


traininglabels = np.asarray(traininglabels)

####################################################################

inputFaces = th.shared(value=transformedFeatureVector[:1,:],name='inputFaces',borrow=True)

randomNumberGenerator = np.random.RandomState(2222)

print 'transformedFeatureVector[:1,:].shape------------>',transformedFeatureVector[:1,:].shape
print 'traininglabels[1]----------->',traininglabels[:1].shape

#------------------------------
energyBasedModel = RestrictedBoltzmannMachines(inputData=inputFaces,
                                            numberOfVisibleUnits=250,
                                            numberOfHiddenUnits=500,
                                            randomNumberGenerator=randomNumberGenerator,
                                            learningRate=0.03
                                            )

weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates ,inputForNextRBM= energyBasedModel.weightUpdation_biasUpdation(5)
#--------------------------------------------------------------------------------------------------------------------------------------------

weights_Updated = energyBasedModel.weights.eval()
print 'weights_Updated.shape---->>>',weights_Updated.shape
print 'weightUpdates.shape---->>>',weightUpdates.shape.eval()
weightUpdates = weightUpdates.eval()
weights_Updated -= weightUpdates

biasOfVisibleUnits_Updated = energyBasedModel.biasForVisibleUnits.eval()
biasOfVisibleUnits_updates = biasOfVisibleUnits_updates[0].eval()
biasOfVisibleUnits_Updated -= biasOfVisibleUnits_updates

biasOfHiddenUnits_Updated = energyBasedModel.biasForHiddenUnits .eval()
biasOfHiddenUnits_updates = biasOfHiddenUnits_updates[0].eval()
biasOfHiddenUnits_Updated -= biasOfHiddenUnits_updates

print '#################################WEIGHTS UPDATED#################################'
#--------------------------------------------------------------------------------------------------------------------------------------------

energyBasedModel.setWeights_And_Bias(weights_Updated,biasOfHiddenUnits_Updated,biasOfVisibleUnits_Updated)

weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates ,inputForNextRBM= energyBasedModel.weightUpdation_biasUpdation(5)

#--------------------------------------------------------------------------------------------------------------------------------------------

print '#######################SECOND RBM MODEL####################'

energyBasedModel_ForHiddenLayer = RestrictedBoltzmannMachines(inputData=inputForNextRBM,
                                            numberOfVisibleUnits=500,
                                            numberOfHiddenUnits=40,
                                            randomNumberGenerator=randomNumberGenerator,
                                            learningRate=0.03
                                            )

weightUpdates_ForHiddenLayer , biasOfHiddenUnits_updates_ForHiddenLayer , biasOfVisibleUnits_updates_ForHiddenLayer ,inputForNextRBM_ForHiddenLayer= energyBasedModel_ForHiddenLayer.weightUpdation_biasUpdation(5)

#---------------------------------------------------------------------------------------------
weights_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.weights.eval()
print 'weights_Updated.shape---->>>',weights_Updated.shape
print 'weightUpdates.shape---->>>',weightUpdates.shape.eval()
weightUpdates_ForHiddenLayer = weightUpdates_ForHiddenLayer.eval()
weights_Updated_ForHiddenLayer -= weightUpdates_ForHiddenLayer

biasOfVisibleUnits_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.biasForVisibleUnits.eval()
biasOfVisibleUnits_updates_ForHiddenLayer = biasOfVisibleUnits_updates_ForHiddenLayer[0].eval()
biasOfVisibleUnits_Updated_ForHiddenLayer -= biasOfVisibleUnits_updates_ForHiddenLayer

biasOfHiddenUnits_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.biasForHiddenUnits .eval()
biasOfHiddenUnits_updates_ForHiddenLayer = biasOfHiddenUnits_updates_ForHiddenLayer[0].eval()
biasOfHiddenUnits_Updated_ForHiddenLayer -= biasOfHiddenUnits_updates_ForHiddenLayer


#---------------------------------------------------------------------------------------------

energyBasedModel_ForHiddenLayer.setWeights_And_Bias(weights_Updated_ForHiddenLayer , biasOfHiddenUnits_Updated_ForHiddenLayer , biasOfVisibleUnits_Updated_ForHiddenLayer)

weightUpdates_ForHiddenLayer , biasOfHiddenUnits_updates_ForHiddenLayer , biasOfVisibleUnits_updates_ForHiddenLayer ,inputForNextRBM_ForHiddenLayer= energyBasedModel_ForHiddenLayer.weightUpdation_biasUpdation(5)



#____________________________________________________________________________________________________________________________________________________________

inputFaces = th.shared(value=transformedFeatureVector[:1,:],name='inputFaces',borrow=True)

#------------------------------------------------------BUILD THE NEURAL NETWORK-----------------------------------------------------------------

inputLayer = InputLayer(inputData=inputFaces,dimensionalityOfInput=250,numberOfHiddenUnits_HiddenLayer=500,randomNumberGenerator=randomNumberGenerator)
hiddenLayer_One = SigmoidLayer(numberOfHiddenUnits=500,numberOfHiddenUnits_NextHiddenLayer=40,randomNumberGenerator=randomNumberGenerator,
                              inputToBeActivated=inputLayer.inputForNextHiddenLayer)
outputLayer = SigmoidOutputLayer(inputToBeActivated=hiddenLayer_One.inputForNextHiddenLayer,numberOfOutputUnits=40,teacherOutputs=traininglabels[:1])

#-------------------------------------------------------------------------------------------------------------------------------------------------

print '*******************************************************SET THE WEIGHTS OF RBM***************************************************************'
inputLayer.setWeights(weightUpdates.eval())
hiddenLayer_One.setWeights(weightUpdates_ForHiddenLayer.eval())

inputLayer.activate()
#------------------------------------------------CONNECT THE NETWORK------------------------------------------------------------------------------
connection_InputLayer_To_HiddenLayer = Connect(inputLayer,hiddenLayer_One)
hiddenLayer_One.activate()
connection_HiddenLayer_To_OuterLayer = ConnectOutputLayer(hiddenLayer_One,outputLayer)
connection_HiddenLayer_To_OuterLayer.setInputForOutputLayer()
#-------------------------------------------------------------------------------------------------------------------------------------------------

outputLayer.setDelta()
connection_HiddenLayer_To_OuterLayer.setGradientMatrix()
connection_InputLayer_To_HiddenLayer.setGradientMatrix()



#################################################################################################################################################################


for eachFace in xrange(1,45):

    print '****************************************************************************************************************************************************************'
    print '================================================================================================================================================================'
    print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TRAINING FACE ->',eachFace,'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

    inputFaces = th.shared(value=transformedFeatureVector[eachFace-1:eachFace,:],name='inputFaces',borrow=True)

    randomNumberGenerator = np.random.RandomState(2222)

    #------------------------------
    energyBasedModel = RestrictedBoltzmannMachines(inputData=inputFaces,
                                            numberOfVisibleUnits=250,
                                            numberOfHiddenUnits=500,
                                            randomNumberGenerator=randomNumberGenerator,
                                            learningRate=0.03
                                            )

    weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates ,inputForNextRBM= energyBasedModel.weightUpdation_biasUpdation(1)
    #--------------------------------------------------------------------------------------------------------------------------------------------

    weights_Updated = energyBasedModel.weights.eval()
    print 'weights_Updated.shape---->>>',weights_Updated.shape
    print 'weightUpdates.shape---->>>',weightUpdates.shape.eval()
    weightUpdates = weightUpdates.eval()
    weights_Updated -= weightUpdates

    biasOfVisibleUnits_Updated = energyBasedModel.biasForVisibleUnits.eval()
    biasOfVisibleUnits_updates = biasOfVisibleUnits_updates[0].eval()
    biasOfVisibleUnits_Updated -= biasOfVisibleUnits_updates

    biasOfHiddenUnits_Updated = energyBasedModel.biasForHiddenUnits .eval()
    biasOfHiddenUnits_updates = biasOfHiddenUnits_updates[0].eval()
    biasOfHiddenUnits_Updated -= biasOfHiddenUnits_updates

    print '#################################WEIGHTS UPDATED#################################'
    #--------------------------------------------------------------------------------------------------------------------------------------------

    energyBasedModel.setWeights_And_Bias(weights_Updated,biasOfHiddenUnits_Updated,biasOfVisibleUnits_Updated)

    weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates ,inputForNextRBM= energyBasedModel.weightUpdation_biasUpdation(1)

    #--------------------------------------------------------------------------------------------------------------------------------------------

    print '#######################SECOND RBM MODEL####################'

    energyBasedModel_ForHiddenLayer = RestrictedBoltzmannMachines(inputData=inputForNextRBM,
                                            numberOfVisibleUnits=500,
                                            numberOfHiddenUnits=40,
                                            randomNumberGenerator=randomNumberGenerator,
                                            learningRate=0.03
                                            )

    weightUpdates_ForHiddenLayer , biasOfHiddenUnits_updates_ForHiddenLayer , biasOfVisibleUnits_updates_ForHiddenLayer ,inputForNextRBM_ForHiddenLayer= energyBasedModel_ForHiddenLayer.weightUpdation_biasUpdation(1)

    #---------------------------------------------------------------------------------------------
    weights_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.weights.eval()
    print 'weights_Updated.shape---->>>',weights_Updated.shape
    print 'weightUpdates.shape---->>>',weightUpdates.shape.eval()
    weightUpdates_ForHiddenLayer = weightUpdates_ForHiddenLayer.eval()
    weights_Updated_ForHiddenLayer -= weightUpdates_ForHiddenLayer

    biasOfVisibleUnits_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.biasForVisibleUnits.eval()
    biasOfVisibleUnits_updates_ForHiddenLayer = biasOfVisibleUnits_updates_ForHiddenLayer[0].eval()
    biasOfVisibleUnits_Updated_ForHiddenLayer -= biasOfVisibleUnits_updates_ForHiddenLayer

    biasOfHiddenUnits_Updated_ForHiddenLayer = energyBasedModel_ForHiddenLayer.biasForHiddenUnits .eval()
    biasOfHiddenUnits_updates_ForHiddenLayer = biasOfHiddenUnits_updates_ForHiddenLayer[0].eval()
    biasOfHiddenUnits_Updated_ForHiddenLayer -= biasOfHiddenUnits_updates_ForHiddenLayer


    #---------------------------------------------------------------------------------------------

    energyBasedModel_ForHiddenLayer.setWeights_And_Bias(weights_Updated_ForHiddenLayer , biasOfHiddenUnits_Updated_ForHiddenLayer , biasOfVisibleUnits_Updated_ForHiddenLayer)

    weightUpdates_ForHiddenLayer , biasOfHiddenUnits_updates_ForHiddenLayer , biasOfVisibleUnits_updates_ForHiddenLayer ,inputForNextRBM_ForHiddenLayer= energyBasedModel_ForHiddenLayer.weightUpdation_biasUpdation(1)



    #____________________________________________________________________________________________________________________________________________________________

    inputLayer.setNewInput(inputFaces)

    print '*******************************************************SET THE WEIGHTS OF RBM***************************************************************'
    inputLayer.setWeights(weightUpdates.eval())
    hiddenLayer_One.setWeights(weightUpdates_ForHiddenLayer.eval())

    inputLayer.activate()
    #------------------------------------------------CONNECT THE NETWORK------------------------------------------------------------------------------
    connection_InputLayer_To_HiddenLayer = Connect(inputLayer,hiddenLayer_One)
    hiddenLayer_One.activate()
    connection_HiddenLayer_To_OuterLayer = ConnectOutputLayer(hiddenLayer_One,outputLayer)
    connection_HiddenLayer_To_OuterLayer.setInputForOutputLayer()
    #-------------------------------------------------------------------------------------------------------------------------------------------------

    outputLayer.setTeacher(traininglabels[eachFace-1:eachFace])
    outputLayer.setDelta()
    connection_HiddenLayer_To_OuterLayer.setGradientMatrix()
    connection_InputLayer_To_HiddenLayer.setGradientMatrix()


print '==================================================TRAINING COMPLETE==================================================='