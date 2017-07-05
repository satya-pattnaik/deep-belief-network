__author__ = 'satya'

from LoadDataSet import loadData
import os
from RestrictedBoltzmannMachines import RestrictedBoltzmannMachines
import numpy as np
import theano as th

(trainingSet_inputX,trainingSet_classlabelY),(testSet_inputX,testSet_classLabelY),(validationSet_inputX,validationSet_classlabelY) = loadData(
    os.path.join('/home/satya/Documents/dataset/MNIST','mnist.pkl.gz'))

randomNumberGenerator = np.random.RandomState(1234)
smallDataSet = trainingSet_inputX[:5,:]

print 'smallDataSet-------------->>',smallDataSet[:,:10].eval()

energyBasedModel = RestrictedBoltzmannMachines(inputData=smallDataSet,
                                               numberOfVisibleUnits=28 *28,
                                               numberOfHiddenUnits=28 *28*2,
                                               randomNumberGenerator=randomNumberGenerator,
                                               learningRate=0.03
                                               )

weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates = energyBasedModel.weightUpdation_biasUpdation(10)

print 'weightUpdates---->',weightUpdates[1,:30].eval()

print 'biasUpdatesHidden------>',biasOfHiddenUnits_updates[1,:30].eval()

print 'biasUpdatesVisible---->',biasOfVisibleUnits_updates[1,:30].eval()


#weightUpdates , biasOfHiddenUnits_updates , biasOfVisibleUnits_updates = energyBasedModel.weightUpdation_biasUpdation(10)

energyBasedModel.weights -= weightUpdates
energyBasedModel.biasForHiddenUnits -= biasOfHiddenUnits_updates
energyBasedModel.biasForVisibleUnits -= biasOfVisibleUnits_updates

print '=======================MODEL TRAINED==============================='



