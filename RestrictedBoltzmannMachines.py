__author__ = 'satya'

import theano as th
import theano.tensor as T
import numpy as np

class RestrictedBoltzmannMachines(object):

    def __init__(self,inputData,numberOfVisibleUnits,numberOfHiddenUnits,randomNumberGenerator,learningRate):

        self.input = inputData

        self.numberOfVisibleUnits = numberOfVisibleUnits
        self.numberOfHiddenUnits = numberOfHiddenUnits
        self.learningRate = learningRate

        weights = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfVisibleUnits,self.numberOfHiddenUnits)
            ),
            dtype=th.config.floatX
        )
        #------------WEIGHTS FOR CONNECTION BETWEEN VISIBLE AND HIDDEN LAYERS----------
        self.weights = th.shared(weights,name='weights',borrow=True)                  #
        #------------------------------------------------------------------------------

        biasForVisibleUnits = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfVisibleUnits,)
            )
        )
        biasForHiddenUnits = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfHiddenUnits,)
            )
        )

        #---------------------------BIAS VALUES FOR VISIBLE AND HIDDEN UNITS-------------------------------------
        self.biasForVisibleUnits = th.shared(biasForVisibleUnits,name='biasForVisibleUnits',borrow=True)        #
        self.biasForHiddenUnits = th.shared(biasForHiddenUnits,name='biasForHiddenUnits',borrow=True)           #
        #--------------------------------------------------------------------------------------------------------


    def propagateUp_And_SampleHidden(self,visibleUnits):

        '''CALCULATE THE MEAN PROBABILITY OF THE HIDDEN UNITS BY PROPAGATING THE VISIBLE UNITS TO THE HIDDEN ONES'''

        #print '----INSIDE--- propagateUp_And_SampleHidden'

        #print 'visibleUnits.shape-->',visibleUnits.shape,'self.weights.shape-->',self.weights.shape.eval()
        #print 'visibleUnits-->TYPE',type(visibleUnits),'------','self.weights.type--->',type(self.weights)

        if 'numpy' in str(type(visibleUnits)):
            if visibleUnits.shape[1] == self.weights.shape[0].eval() :
                preSigmoidMean = T.dot(visibleUnits,self.weights)


                if preSigmoidMean.shape[1].eval() == self.biasForHiddenUnits.shape.eval():
                    #print 'PARAMETERS SIZE MATCHING INSIDE'
                    #print 'self.biasForHiddenUnits',self.biasForHiddenUnits
                    preSigmoidMeanWithBias = preSigmoidMean + self.biasForHiddenUnits.eval()
                else:
                    print 'PARAMETER SIZE MISMATCH INSIDE'

            else:
                print 'PARAMETER SIZE MISMATCH OUTSIDE'

            self.meanProbabilityOfEachHiddenNeuron = T.nnet.sigmoid(preSigmoidMeanWithBias)                                                     #

            #print 'self.meanProbabilityOfEachHiddenNeuron-->',self.meanProbabilityOfEachHiddenNeuron.shape.eval()

        else:
            if visibleUnits.shape[1].eval() == self.weights.shape[0].eval() :
                preSigmoidMean = T.dot(visibleUnits,self.weights)


                if preSigmoidMean.shape[1].eval() == self.biasForHiddenUnits.shape.eval():
                    #print 'PARAMETERS SIZE MATCHING INSIDE'
                    #print 'self.biasForHiddenUnits',self.biasForHiddenUnits
                    preSigmoidMeanWithBias = preSigmoidMean + self.biasForHiddenUnits.eval()
                else:
                    print 'PARAMETER SIZE MISMATCH INSIDE'

            else:
                print 'PARAMETER SIZE MISMATCH OUTSIDE'

            self.meanProbabilityOfEachHiddenNeuron = T.nnet.sigmoid(preSigmoidMeanWithBias)                                                     #

            #print 'self.meanProbabilityOfEachHiddenNeuron-->',self.meanProbabilityOfEachHiddenNeuron.shape.eval()



    def sample_HiddenValue(self,givenVisibleUnitsSample):

        #print '---INSIDE sample_HiddenValue---'

        self.propagateUp_And_SampleHidden(givenVisibleUnitsSample)

        requiredShape = self.meanProbabilityOfEachHiddenNeuron.shape.eval()




        hiddenSample = []

        for eachMeanProbability,seed in zip(np.nditer(self.meanProbabilityOfEachHiddenNeuron.eval(),flags=['refs_ok']),
                                            xrange(self.meanProbabilityOfEachHiddenNeuron.size.eval())) :

            randomVariable = np.random.RandomState(seed*1000 + seed*20 +seed*33)
            uniformRandomVariable = randomVariable.uniform()


            #hiddenSampleForEachNeuron=[]
            #for eachHiddenMean in np.nditer(eachMeanProbability,flags=['refs_ok']):

                # if uniformRandomVariable >= eachHiddenMean :
            if T.ge(uniformRandomVariable,eachMeanProbability).eval() == 1:

                hiddenSample.append(0)
                # elif uniformRandomVariable<eachHiddenMean:
            elif T.le(uniformRandomVariable,eachMeanProbability).eval() == 1:

                hiddenSample.append(1)

            #hiddenSample.append(hiddenSampleForEachNeuron)
        np_hiddenSample = th.shared(np.asarray(hiddenSample),name='np_hiddenSample',borrow=True)

        #print 'hiddenSample.shape.eval()',np_hiddenSample.shape.eval(),'----------------------->REQUIRED SHAPE--->',requiredShape
        matrix_hiddenSample = th.shared(np.reshape(np_hiddenSample.eval(),(requiredShape[0],requiredShape[1])),name='matrix_hiddenSample',borrow=True)
        #print 'matrix_hiddenSample',matrix_hiddenSample.shape
        #---POSITIVE OR NEGATIVE SAMPLED VALUES OF THE HIDDEN NEURONS--#
        return matrix_hiddenSample                                     #
        #--------------------------------------------------------------#

    def propagateDown_And_SampleVisible_Negative(self,hiddenSample):

        #print 'INSIDE --------propagateDown_And_SampleVisible_Negative------------'

        #print '-------HIDDEN SAMPLE------>',hiddenSample



        if hiddenSample.shape[1].eval() == self.weights.shape[1].eval():
            transposeWeights = T.transpose(self.weights)
            preSigmoidMean = T.dot(hiddenSample.eval(),transposeWeights)

            #print 'preSigmoidMean--->',preSigmoidMean

            #print preSigmoidMean.shape[1].eval() , '<---preSigmoidMean--DEBUG--biasForVisibleUnits--->',self.biasForVisibleUnits.shape.eval()

            if preSigmoidMean.shape[1].eval() == self.biasForVisibleUnits.shape.eval():
                #print 'PARAMETERS MATCH INSIDE propagateDown_And_SampleVisible_Negative()'
                preSigmoidMeanWithBias = preSigmoidMean + self.biasForVisibleUnits
            else:
                print 'PARAMETER SIZE MISMATCH'

        else:
            print 'PARAMETER SIZE MISMATCH'

        #------------------------------------------------------NEGATIVE MEAN PROBABILITIES OF HIDDEN NEURONS------------------------------------------------#
        self.meanProbabilityOfEachVisibleNeuron = T.nnet.sigmoid(preSigmoidMeanWithBias)                                                                         #
        #self.meanProbabilityOfEachVisibleNeuron = th.shared(meanProbabilityOfEachVisibleNeuron.eval(),name='meanProbabilityOfEachVisibleNeuron',                   #
        #                                                    borrow=True)                                                                                    #
        #---------------------------------------------------------------------------------------------------------------------------------------------------#

    def sample_VisibleValue(self,givenHiddenUnitsSample):

        #print '------INSIDE sample_VisibleValue------'

        #print 'TYPE----------->givenHiddenUnitsSample------->',type(givenHiddenUnitsSample)

        if 'numpy' in str(type(givenHiddenUnitsSample)):
            #print '--------BINGO-------'
            self.propagateDown_And_SampleVisible_Negative(th.shared(givenHiddenUnitsSample,name='givenHiddenUnitsSample',borrow=True))
        else:

            self.propagateDown_And_SampleVisible_Negative(th.shared(givenHiddenUnitsSample.eval(),name='givenHiddenUnitsSample',borrow=True))

        requiredShape = self.meanProbabilityOfEachVisibleNeuron.shape.eval()

        visibleSample = []

        for eachMeanProbability,seed in zip( np.nditer(self.meanProbabilityOfEachVisibleNeuron.eval(),flags=['refs_ok'])
                                            , xrange(self.meanProbabilityOfEachVisibleNeuron.size.eval())):

            #visibleMean = []
            #for eachVisibleNeuronMean in eachMeanProbability:
            randomVariable = np.random.RandomState(seed*1000 + seed*20 +seed*33)
            uniformRandomVariable = randomVariable.uniform()

                #if uniformRandomVariable < eachMeanProbability:
            if T.lt(uniformRandomVariable,eachMeanProbability).eval() == 1:
                visibleSample.append(1)

                #elif uniformRandomVariable >= eachMeanProbability:
            elif T.ge(uniformRandomVariable,eachMeanProbability).eval() == 1:
                visibleSample.append(0)


            #visibleSample.append(visibleMean)

        reshapedMatrix_visibleSample = th.shared(np.reshape(visibleSample,(requiredShape[0],requiredShape[1])),name='reshapedMatrix_visibleSample',borrow=True)


        #---NEGATIVE SAMPLED VALUES OF THE HIDDEN NEURONS--#
        if 'numpy' in str(type(givenHiddenUnitsSample)):
            return reshapedMatrix_visibleSample.eval()

        else:
            return reshapedMatrix_visibleSample                #
        #--------------------------------------------------#

    #------------------------------------------------GIBBS SAMPLING------------------------------------------------#

    def propagateUp_Positive_And_ExtractFirstHiddenSamples(self):

        #print 'INSIDE propagateUp_Positive_And_ExtractFirstHiddenSamples'

        initialHiddenSamples = self.sample_HiddenValue(self.input)

        return initialHiddenSamples

    def GibbsSample_givenHiddenSamples(self,givenHiddenSamples):

        #print 'INSIDE GibbsSample_givenHiddenSamples'

        visibleUnitSamples = th.shared(np.asarray(self.sample_VisibleValue(givenHiddenSamples.eval())),
                                       name='visibleUnitSamples',borrow=True)
        return visibleUnitSamples

    def GibbsSample_givenVisibleSamples(self,givenVisibleSamples):

        #print 'INSIDE GibbsSample_givenVisibleSamples'

        hiddenUnitSamples = th.shared(np.asarray(self.sample_HiddenValue(givenVisibleSamples).eval()),
                                      name='hiddenUnitSamples',borrow=True)
        return hiddenUnitSamples

    #-------------------------------------------------------------------------------------------------------------- #

    def weightUpdation_biasUpdation(self,cyclesOfGibbsSamplerForContrastiveDivergence):

        #print '---------INSIDE WEIGHTUPDATION AND BIASUPDATION----------','--cyclesOfGibbsSamplerForContrastiveDivergence-->',cyclesOfGibbsSamplerForContrastiveDivergence

        positiveHiddenSamples = th.shared(np.asarray(self.propagateUp_Positive_And_ExtractFirstHiddenSamples().eval()),
                                                     name='positiveHiddenSamples',borrow=True)

        #print self.input.T,'<--------------self.input.shape'
        #print (self.input.T).shape[1],'<---T.dot(T.transpose(self.input)-----positiveHiddenSamples--->',(positiveHiddenSamples.shape[0])

        positiveHiddenMeans_Dot_inputVector = T.dot((self.input.T),positiveHiddenSamples)

        temporaryVisibleUnits = []
        temporaryHiddenUnits = []

        for iter in xrange(cyclesOfGibbsSamplerForContrastiveDivergence):

            if iter == 0:
                print '----------------THE ONE WITH ITER==0-----------------------'

                temporaryVisibleUnits = th.shared(np.asarray(self.GibbsSample_givenHiddenSamples(positiveHiddenSamples).eval()),
                                                             name='temporaryVisibleUnits',borrow=True)

                temporaryHiddenUnits = th.shared(np.asarray(self.GibbsSample_givenVisibleSamples(temporaryVisibleUnits).eval()),
                                                            name='temporaryHiddenUnits',borrow=True)

            else:
                print '----------------THE ONE WITH ITER>0-----------------------'

                temporaryVisibleUnits = th.shared(np.asarray( self.GibbsSample_givenHiddenSamples(temporaryHiddenUnits).eval()),
                                                  name='temporaryVisibleUnits',borrow=True)

                temporaryHiddenUnits = th.shared(np.asarray(self.GibbsSample_givenVisibleSamples(temporaryVisibleUnits.eval()).eval()),
                                                 name='temporaryHiddenUnits',borrow=True)

        print '----------------------GIBBS SAMPLING DONE----------------------'

        negativeHiddenMeans_Dot_negativeVisibleMeans = T.dot(T.transpose(temporaryVisibleUnits),temporaryHiddenUnits)

        self.hiddenLayerStates = temporaryHiddenUnits
        self.visibleLayerStates = temporaryVisibleUnits


        weightUpdate = ( self.learningRate * (positiveHiddenMeans_Dot_inputVector - negativeHiddenMeans_Dot_negativeVisibleMeans) ) / float(self.numberOfVisibleUnits)

        biasOfHiddenUnitsUpdate = (self.learningRate * (positiveHiddenSamples - temporaryHiddenUnits)) / float(self.numberOfHiddenUnits)

        biasOfVisibleUnitsUpdate = (self.learningRate * (self.input - temporaryVisibleUnits)) / float(self.numberOfVisibleUnits)

        return weightUpdate , biasOfHiddenUnitsUpdate , biasOfVisibleUnitsUpdate ,temporaryHiddenUnits

    def setWeights_And_Bias(self,weightMatrix,biasForHiddenUnitsMatrix,biasForVisibleUnitsMatrix):

        self.weights = th.shared(weightMatrix,name='weights',borrow=True)
        self.biasForHiddenUnits = th.shared(biasForHiddenUnitsMatrix,name='biasForHiddenUnits',borrow=True)
        self.biasForVisibleUnits = th.shared(biasForVisibleUnitsMatrix,name='biasForVisibleUnits',borrow=True)



    def setNewInput(self,newInput):
        self.input = newInput

    def calculateFreeEnergy(self):
        visibleUnits_States = self.input.eval()
        biasVisible = self.biasForVisibleUnits.eval()

        hiddenUnits_States = self.hiddenLayerStates.eval()
        biasHidden = self.biasForHiddenUnits.eval()

        weights = self.weights.eval()

        hidden_Dot_Weights = np.dot(hiddenUnits_States,np.transpose(weights))

        energy = -np.dot(biasVisible,np.transpose(visibleUnits_States)) - np.dot(biasHidden,np.transpose(hiddenUnits_States)) - np.dot(hidden_Dot_Weights,np.transpose(visibleUnits_States))

        return energy














