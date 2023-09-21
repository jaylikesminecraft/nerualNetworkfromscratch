import math
import random
import matplotlib.pyplot as plt
import csv
import json
import timeit

#foraward propogation
def network(x: list, networkWeights: list):
    inputs = x
    #store the outputs of the hidden neurons
    hiddenOutputs = []
    #the inputs are considered the first hidden layer
    hiddenOutputs.append(inputs)
    # i is the layer
    for i in range(len(networkWeights)):
        layerOutput = []
        # j in the neuron's weights in that layer
        for j in range(len(networkWeights[i])):
            #select the neuron to work on
            weights = networkWeights[i][j]
            #save the output of that neuron
            layerOutput.append(sigmoidEstimation(inputs, weights))
        #in the next layer use the outputs from this layer as the inputs
        hiddenOutputs.append(layerOutput)
        inputs = layerOutput

    return inputs, hiddenOutputs

def D_sigmoid(z):
    return z * (1-z)

def networkBackPropogation(x: list, y: list, networkWeights: list, iterations, step):

    neuronError = []
    mse = []
    numLayers = len(networkWeights)
    finalLayer = numLayers-1

    #build an array structure to store the error for each neuron
    for layers in range(len(networkWeights)):
        layer = []
        for neuron in range(len(networkWeights[layers])):
            layer.append(0)
        neuronError.append(layer)
            
    for i in range(iterations):
        #for every piece of data
        errorAtInteration = []
        for j in range(len(y)):
            
            for numberOfOutputNeurons in range(len(networkWeights[finalLayer])):
                
                #forward propogation
                output, hiddenOutputs = network(x[j], networkWeights)
                error = y[j][numberOfOutputNeurons] - output[numberOfOutputNeurons]
                #save error to graph
                errorAtInteration.append(error**2)
                #derivative of error
                dError = error * D_sigmoid(output[numberOfOutputNeurons])

                #get the values from the layer just before
                hiddenOutput = hiddenOutputs[finalLayer]

                #start the backpropogation from the output layer
                for weight in range(len(networkWeights[finalLayer][numberOfOutputNeurons])):
                    if weight == 0:
                        networkWeights[finalLayer][numberOfOutputNeurons][weight] += step * dError
                    else:
                        neuronError[finalLayer-1][weight-1] = (dError * D_sigmoid(hiddenOutput[weight-1]) * networkWeights[finalLayer][numberOfOutputNeurons][weight])
                        networkWeights[finalLayer][numberOfOutputNeurons][weight] += step * hiddenOutput[weight-1] * dError

                #start backpropogating through each layer
                for currentLayer in reversed(range(finalLayer)):
                    #print("hiddenlayer")
                    hiddenOutput = hiddenOutputs[currentLayer]

                    for neuron in range(len(networkWeights[currentLayer])):
                        for weight in range(len(networkWeights[currentLayer][neuron])):
                            if weight == 0:
                                #adjust the bias
                                networkWeights[currentLayer][neuron][weight] += step * neuronError[currentLayer][neuron]
                            else:
                                #adjust the weights
                                networkWeights[currentLayer][neuron][weight] += step * neuronError[currentLayer][neuron] * hiddenOutput[weight-1]

                        for weight in range(len(networkWeights[currentLayer][neuron])):
                            if not weight == 0:
                                neuronError[currentLayer][neuron] = (dError * D_sigmoid(hiddenOutput[weight-1]) * networkWeights[currentLayer][neuron][weight])
            mse.append(max(errorAtInteration))
        #if the maximum error is low enough stop early
        if mse[i] < 0.005:
            print("skipping further iterations, max error =",mse[i])
            break
    return networkWeights, mse

def sigmoidEstimation(x: list, weights: list):
    sum = weights[0] # bias
    for i in range(len(x)):
        sum += weights[i+1] * x[i]
    return sigmoid(sum)

def sigmoid(z):
    a = 1 / (1 + math.exp((-1) * z))
    return a

def printMatrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

def buildNetwork(neuronsPerLayer):

    networkWeights = []
    weightsPerNeuron = neuronsPerLayer[0]
    neuronsPerLayer.pop(0)
    for layers in range(len(neuronsPerLayer)):
        networkWeights.append([])
        for neurons in range((neuronsPerLayer[layers])):
            neuron = [1] #default bias
            for weights in range((weightsPerNeuron)):
                neuron.append(random.uniform(-1,1))
            networkWeights[layers].append(neuron)
        weightsPerNeuron = neuronsPerLayer[layers]
    
    return networkWeights

def printResuts(x, y, networkWeights):

    accuracyForNumber = [0,0,0,0,0,0,0,0,0,0]
    numbersSeen =       [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(x)):
        
        output = network(x[i], networkWeights)
        output = output[0]
        acutalNum = y[i].index(1)
        numbersSeen[acutalNum] += 1
        predicted = max(output)
        predictedNum = output.index(predicted)

        if predictedNum == acutalNum:
            accuracyForNumber[acutalNum] += 1
    
    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] /= numbersSeen[i]

    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] = round(accuracyForNumber[i], 3)
    
    return accuracyForNumber, average(accuracyForNumber)

    # print(accuracyForNumber)
    # print(round(average(accuracyForNumber),4))
    
def buildDataset(filename, delim, batchSize, numOfBatches, start):
    
    batchX = []
    batchY = []
    
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        
        i = -1
        currentSize = -1
        currentNumOfBatches = 0

        x = []
        y = []

        for row in reader:
            
            i+=1
            if i <= start:          
                continue
            currentSize+=1
            
            dataX = []
            dataY = [0,0,0,0,0,0,0,0,0,0]
            dataY[int(row[0])] = 1
            skip = 0
            for pixel in range(len(row)):
                if skip == 0:
                    skip +=1
                    continue
                #convert intergers to floats
                dataX.append(float(row[pixel])/ 255)# / 255
            x.append(dataX)
            y.append(dataY)

            if currentSize == batchSize:
                batchX.append(x)
                batchY.append(y)
                x = []
                y = []
                currentSize = -1
                currentNumOfBatches += 1
                if currentNumOfBatches == numOfBatches:
                    break
                
    return batchX, batchY
            
def renderData(x):
    
    # ██ ▓▓ ░░
    for i in range(28):
        string = ""
        for j in range(28):
            if x[i * 28 + j] > 120/255:
                string += "  "
            elif x[i * 28 + j] > 80/255:
                string +="░░"
            elif x[i * 28 + j] > 40/255:
                string +="▓▓"
            else:
                string += "██"
        print(string)

def saveNetwork(network, filename):
    with open(filename, 'w') as outfile:
        json.dump(network, outfile)

def loadNetwork(filename):
    network = []
    with open(filename, newline='') as outfile:
        network = json.load(outfile)
    return network

def average(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    return sum/len(list)

def printTestResuts(x, y, networkWeights):

    accuracyForNumber = [0,0,0,0,0,0,0,0,0,0]
    numbersSeen =       [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(x)):
        
        output = network(x[i], networkWeights)
        output = output[0]

        acutalNum = y[i].index(1)
        numbersSeen[acutalNum] += 1
        predicted = max(output)
        predictedNum = output.index(predicted)

        if predictedNum == acutalNum:
            accuracyForNumber[acutalNum] += 1
    
    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] /= numbersSeen[i]

    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] = round(accuracyForNumber[i], 3)
    
    return (str(accuracyForNumber) + str(average(y[0])))

def train(numOfBatches, iterations, stepSize, batchX, batchY, networkWeights, checkFitness: bool):
    
    testX, testY = buildDataset('mnist_test.csv', ",", 750, 1,1)

    fitnessAtBatch = []

    #measure time
    t_0 = timeit.default_timer()

    for i in range(numOfBatches):
        batchT0 = timeit.default_timer()
        X = batchX[i]
        Y = batchY[i]
        networkWeights, mse = networkBackPropogation(X, Y, networkWeights, iterations, stepSize)
        batchT1 = timeit.default_timer()
        elapsed_batch_time = round((batchT1-batchT0), 3)
        print("batch "+ str(i) + " completed in " + str(round(elapsed_batch_time, 3)) + "s" )

        #check progress as training goes on
        if checkFitness:
            eachNumberResult, fitness = printResuts(testX[0], testY[0], networkWeights)
            print(str(eachNumberResult) + " fitness = " + str(fitness))
            fitnessAtBatch.append(fitness)

    #finished time
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0), 3)
    print(f"Elapsed time: {elapsed_time} s")

    return networkWeights, fitnessAtBatch 


numOfBatches = 10
batchSize = 15
iterationsPerBatch = 2
stepSize = 0.1
#data set
# 'mnist_train.csv'
# 'failed.csv'
batchX, batchY = buildDataset('mnist_train.csv', ",", batchSize, numOfBatches,5000)

#networkWeights = loadNetwork("2hiddenLayersMoreTraining.json")
networkWeights = buildNetwork([784,40,25,15,10])

#train the network with mini batch descent
networkWeights, fitnessAtBatch = train(numOfBatches, iterationsPerBatch, stepSize, batchX, batchY, networkWeights, True)

saveNetwork(networkWeights, "newNetwork.json")

plt.plot(fitnessAtBatch)
plt.show()
