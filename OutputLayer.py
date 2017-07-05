__author__ = 'satya'

import numpy as np
import theano as th
import theano.tensor as T

##############
def softMax(matrixToBeNormalized):

    print '=============INSIDE SOFTMAX============='

    matrix_Normalized = []

    for rowNumber,eachRow  in enumerate(matrixToBeNormalized.eval()):

        eachRow_exponential = np.exp(eachRow)
        eachRow_Normalized = eachRow_exponential/float(sum(eachRow_exponential))
        matrix_Normalized.append(eachRow_Normalized)
        #print 'rowNumber-->',rowNumber,'eachRow-->',eachRow

    matrix_Normalized = np.asarray(matrix_Normalized)

    return matrix_Normalized

#----------------------- THE ARGMAX FUNCTION-------------------------------------------

def argMax(matrix):
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

########################################################################################################################
def appender(number):
    neuronStates = []
    if number is 1:
        neuronStates.append(1)
        for i in xrange(39):
            neuronStates.append(0)

    elif number is 40:
        for i in xrange(39):
            neuronStates.append(0)
        neuronStates.append(40)

    else:
        leftZeros = number-1
        rightZeros = 40 - number
        for i in xrange(leftZeros):
            neuronStates.append(0)

        neuronStates.append(number)

        for i in xrange(rightZeros):
            neuronStates.append(0)

    return neuronStates

########################################################################################################################
'''
def extractError(deltaMatrix):
    error = 0
    for eachDeltaValue in np.nditer(deltaMatrix):
        if eachDeltaValue<0:
            error = eachDeltaValue
            break
    print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ERROR DEECTED---->',error,'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

    for index in xrange(40):

        deltaMatrix[0,index] = error

    return deltaMatrix
'''

class OutputLayer(object):

    def __init__(self,predictedOutput,numberOfOutputUnits,teacherOutputs):

        #------------------NUMBER OF OUTPUT UNITS----------------------------------------
        #self.numberOfOutputUnits = th.shared(numberOfOutputUnits,name='',borrow=True)
        self.numberOfOutputUnits = numberOfOutputUnits
        #--------------------------------------------------------------------------------

        #---------------------------TEACHER'S OUTPUTS------------------------------------
        #teacher_Outputs = th.shared(teacherOutputs,name='teacher_Outputs',borrow=True)
        teacherValues = []
        for eachTeacherValue in np.nditer(teacherOutputs):
            teacherValues.append(appender(eachTeacherValue))

        self.teacher_Outputs = np.asarray(teacherValues)
        #--------------------------------------------------------------------------------



class HyperbolicTangentOutputLayer(OutputLayer):

    def __init__(self,inputToBeActivated,numberOfOutputUnits,teacherOutputs):

        predictedOutput = T.tanh(inputToBeActivated)

        #----------------------------CALL SUPERCLASS __init__ METHOD--------------------------------
        super(HyperbolicTangentOutputLayer,self).__init__(predictedOutput,numberOfOutputUnits,teacherOutputs)       #
        #-------------------------------------------------------------------------------------------

         #-----------------STATE OF THE SOFTMAX LAYER AFTER APPLYING THE SOFTMAX FUNCTION-----------------

        print '=============SET stateOfSoftMaxLayer============='

        stateOfSoftMaxLayer = self.softMax(predictedOutput)

        #------------------------------------------------------------------------------------------------

class SigmoidOutputLayer(OutputLayer):

    def __init__(self,inputToBeActivated,numberOfOutputUnits,teacherOutputs):

        self.inputToBeActivated = inputToBeActivated

        print 'INSIDE---->SigmoidOutputLayer'

        predictedOutput = T.nnet.sigmoid(inputToBeActivated)

        #----------------------------CALL SUPERCLASS __init__ METHOD--------------------------------
        super(SigmoidOutputLayer,self).__init__(predictedOutput,numberOfOutputUnits,teacherOutputs)       #
        #-------------------------------------------------------------------------------------------

        #------------------------THE SOFTMAX FUNCTION-------------------------------------------

        def softMax(self,matrixToBeNormalized):

            print '=============INSIDE SOFTMAX============='

            matrix_Normalized = []

            for rowNumber,eachRow  in enumerate(matrixToBeNormalized.eval()):

                eachRow_exponential = np.exp(eachRow)
                eachRow_Normalized = eachRow_exponential/float(sum(eachRow_exponential))
                matrix_Normalized.append(eachRow_Normalized)
                #print 'rowNumber-->',rowNumber,'eachRow-->',eachRow

            matrix_Normalized = np.asarray(matrix_Normalized)

            return matrix_Normalized

        #----------------------- THE ARGMAX FUNCTION-------------------------------------------

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

        #-----------------STATE OF THE SOFTMAX LAYER AFTER APPLYING THE SOFTMAX FUNCTION-----------------

        print '=============SET stateOfSoftMaxLayer============='



        stateOfSoftMaxLayer = softMax(self,predictedOutput)

        print 'stateOfSoftMaxLayer--->>LENGTH-->',stateOfSoftMaxLayer.shape
        #--------------------------------------------------------------------------------------------------

        #-----------------------------------PREDICTION OF THE SOFTMAX LAYER------------------------------

        #labelPrediction = T.argmax(stateOfSoftMaxLayer,axis=1)
        labelPrediction = np.asarray(argMax(self,stateOfSoftMaxLayer))

        prediction = []
        for eachPrediction in labelPrediction:
            prediction.append(appender(eachPrediction))

        self.predicted_Output = np.asarray(prediction)

        print 'LABEL PREDICTION----->',self.predicted_Output
        print 'TEACHER SAYS----->',self.teacher_Outputs
        print 'LABEL PRODUCTION SHAPE----->',self.predicted_Output.shape
        print 'TEACHER SAYS SHAPE----->',self.teacher_Outputs.shape

        #----------------------------DELTA OUTPUT--------------------------------------

        deltaOutput = self.teacher_Outputs - self.predicted_Output

        print 'DELTA OUTPUT---->>>',(deltaOutput)

        #----------DIFFERENCE BETWEEN PREDICTED OUTPUT AND TEACHER OUTPUT--------------

        self.delta_Output = (deltaOutput)

        #------------------------------------------------------------------------------

    def setDelta(self):

        print '===============SETTING DELTA AT OUTPUT LAYER================'

        predictedOutput = T.nnet.sigmoid(self.inputToBeActivated)

        stateOfSoftMaxLayer = softMax(predictedOutput)


        #--------------------------------------------------------------------------------------------------

        #-----------------------------------PREDICTION OF THE SOFTMAX LAYER------------------------------

        #labelPrediction = T.argmax(stateOfSoftMaxLayer,axis=1)
        labelPrediction = np.asarray(argMax(stateOfSoftMaxLayer))

        prediction = []
        for eachPrediction in labelPrediction:
            prediction.append(appender(eachPrediction))

        self.predicted_Output = np.asarray(prediction)

        print 'LABEL PREDICTION----->',self.predicted_Output
        print 'TEACHER SAYS----->',self.teacher_Outputs
        print 'LABEL PRODUCTION SHAPE----->',self.predicted_Output.shape
        print 'TEACHER SAYS SHAPE----->',self.teacher_Outputs.shape

        #----------------------------DELTA OUTPUT--------------------------------------

        deltaOutput = self.teacher_Outputs - self.predicted_Output

        #----------DIFFERENCE BETWEEN PREDICTED OUTPUT AND TEACHER OUTPUT--------------

        self.delta_Output = (deltaOutput)

        #------------------------------------------------------------------------------

        print '===============DELTA AT OUTPUT LAYER SET===================='

    def setTeacher(self,teacherOutputs):
        teacherValues = []
        for eachTeacherValue in np.nditer(teacherOutputs):
            teacherValues.append(appender(eachTeacherValue))

        self.teacher_Outputs = np.asarray(teacherValues)