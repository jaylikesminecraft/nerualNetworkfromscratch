import math
import random
import csv
import json

#foraward propogation
def network(x: list, networkWeights: list):
    inputs = x
    #store the outputs of the hidden neurons
    #the inputs are considered the first hidden layer
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
        inputs = layerOutput

    return inputs

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

def buildNetwork(neuronsPerLayer, numberOfXInputs):

    networkWeights = []
    weightsPerNeuron = numberOfXInputs

    for layers in range(len(neuronsPerLayer)):
        networkWeights.append([])
        for neurons in range((neuronsPerLayer[layers])):
            neuron = [1] #default bias
            for weights in range((weightsPerNeuron)):
                neuron.append(random.uniform(-1,1))
            networkWeights[layers].append(neuron)
        weightsPerNeuron = neuronsPerLayer[layers]
    
    return networkWeights

def average(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    return sum/len(list)

def resuts(x, y, networkWeights):

    accuracyForNumber = [0,0,0,0,0,0,0,0,0,0]
    numbersSeen =       [0,0,0,0,0,0,0,0,0,0]

    failed = []

    for i in range(len(x)):
        
        output = network(x[i], networkWeights)

        acutalNum = y[i].index(1)
        numbersSeen[acutalNum] += 1
        predicted = max(output)
        predictedNum = output.index(predicted)

        if predictedNum == acutalNum:
            accuracyForNumber[acutalNum] += 1
        else:
            #print("fail")
            data = []
            data.append(acutalNum)
            for pixel in range(len(x[i])):
                data.append(x[i][pixel] * 255)
            if (not acutalNum == 0) and (not acutalNum == 1):
                failed.append(data)
    
    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] /= numbersSeen[i]

    for i in range(len(accuracyForNumber)):
        accuracyForNumber[i] = round(accuracyForNumber[i], 3)
    
    print(accuracyForNumber)
    print(round(average(accuracyForNumber),4))

    return failed
        
def visualResult(x, answer, networkWeights, onlyFailed):
    output = network(x, networkWeights)
    predicted = max(output)
    predictedNum = output.index(predicted)

    if onlyFailed:
        if not predictedNum == answer:
            renderData(x)
            for i in range(len(output)):
                output[i] = round(output[i], 2)
            print("  0    1    2    3    4     5    6    7    8    9  acutal=" ,answer)
            print(str(output))
    else:
        renderData(x)
        for i in range(len(output)):
            output[i] = round(output[i], 2)
        print("  0    1    2    3    4     5    6    7    8    9  acutal=" ,answer)
        print(str(output))

    return
    
#import csv
#returns batches of data
def buildDataset(filename, delim, batchSize, numOfBatches, includeAnswer):
    batchX = []
    batchY = []
    answer = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        
        i = -1
        currentSize = -1
        currentNumOfBatches = 0
        x = []
        y = []
        
        for row in reader:
            
            i+=1
            currentSize+=1
            if i == 0:     
                continue
            
            dataX = []
            dataY = [0,0,0,0,0,0,0,0,0,0]
            dataY[int(row[0])] = 1
            skip = 0
            for pixel in range(len(row)):
                if skip == 0:
                    answer.append(float(row[0]))
                    skip +=1
                    continue
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
                
    if includeAnswer == True:
        return batchX, batchY, answer
    
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

def loadNetwork(filename):
    network = []
    with open(filename, newline='') as outfile:
        network = json.load(outfile)
    return network

def saveNetwork(network, filename):
    with open(filename, 'w') as outfile:
        json.dump(network, outfile)

def writeCSV(filename, failed):
    with open(filename, 'w',newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for i in range(len(failed)):
            writer.writerow(failed[i])

def networkDescription(networkWeights):
    networkDescription = []
    networkDescription.append(len(networkWeights[0][0]))
    for layer in range(len(networkWeights)):
        networkDescription.append(len(networkWeights[layer]))
    print(networkDescription)

networkWeights = loadNetwork("32-hiddenLayer.json")
#networkWeights = loadNetwork("newNetwork.json")

networkDescription(networkWeights)

testX, testY, answer = buildDataset('mnist_test.csv', ",", 1000, 1, True)
failed = resuts(testX[0], testY[0], networkWeights)

writeCSV("failed.csv",failed)

testX, testY, answer = buildDataset('failed.csv', ",", 10, 1, True)
for i in range(len(testX[0])):
    visualResult(testX[0][i], answer[i], networkWeights, False)