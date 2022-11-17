####################################################
####    Assignment-2                           #####
####    Supervised Learning                    #####
####    Author: Utsav Mehta and Umang Singla   #####
####            (20CS10069)     (20CS10068)    #####
####################################################

import numpy
import pandas
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# function to calculate the accuracy of the predicted output
def calculate_accuracy(testOutputs, testLabels):
    # Calculate the accuracy of the model
    correct = 0
    for i in range(len(testLabels)):
        if testOutputs[i] == testLabels[i]:
            correct += 1
    accuracy = correct / len(testLabels)
    return accuracy

# function for forward feature seleciton method
# It returns the best features for which model give greater acurracy
def forward_selection(data , model):
    best_features = []
    trainData, testData = random_divide(data)
    trainFeatures, trainLabels = encode_categorial_variable(trainData)
    testFeatures, testLabels = encode_categorial_variable(testData)
    initial_features = numpy.random.permutation(len(trainFeatures[0]))
    initial_accuracy = 0
    while len(initial_features) > 0:
        accuracy = []
        for i in initial_features:
            tdata = trainFeatures[:,i]
            tdata = tdata.reshape(len(tdata), 1)
            tmodel = model.fit(tdata, trainLabels)
            tdata = testFeatures[:,i]
            tdata = tdata.reshape(len(tdata), 1)
            accuracy.append(calculate_accuracy(testLabels,tmodel.predict(tdata)))

        if max(accuracy) >= initial_accuracy:
            initial_accuracy = max(accuracy)
            best_features.append(initial_features[numpy.argmax(accuracy)])
            initial_features = numpy.delete(initial_features, numpy.argmax(accuracy))
        else:
            break

    return best_features

# function for ensemble learnign model using max voting technique
def ensemble_learning(model1, model2, model3, data):
    # Ensemble learning using voting classifier
    trainData, testData = random_divide(data)
    trainFeatures, trainLabels = encode_categorial_variable(trainData)
    testFeatures, testLabels = encode_categorial_variable(testData)
    model1.fit(trainFeatures, trainLabels)
    model2.fit(trainFeatures, trainLabels)
    model3.fit(trainFeatures, trainLabels)
    pred1 = model1.predict(testFeatures)
    pred2 = model2.predict(testFeatures)
    pred3 = model3.predict(testFeatures)

    pred = []

    # classifying the data to the class to which max vote is given
    for i in range(len(pred1)):
        if pred1[i] == pred2[i]:
            pred.append(pred1[i])
        elif pred1[i] == pred3[i]:
            pred.append(pred1[i])
        elif pred2[i] == pred3[i]:
            pred.append(pred2[i])
        else:
            pred.append(pred1[i])

    return calculate_accuracy(pred, testLabels)
    
# function to devide the data into features and labels
def encode_categorial_variable(data):
    # divide the data into features and labels
    labels = data[:,0]
    features = data[:,1:]
    # encode the labels
    labels = labels.astype(int)
    return features, labels

# function for standard scalar normalisation on the data
def standard_scalar_normalisation(data):
    # Standardise the numpy data (except the first column)
    data = data.to_numpy()
    for columns in range(0, len(data[0])):
        data[:, columns] = (data[:, columns] - numpy.mean(data[:, columns])) / numpy.std(data[:, columns])
    return data

# function to read the data from csv file
def read_data(filePath):
    dataset = pandas.read_csv(filePath, sep=',', header=None)
    dataset = pandas.read_csv("lung-cancer.data", header=None)
    dataset.replace(to_replace ="?", value = "", inplace=True) 
    for i in [4, 38]:
        x = dataset[i].mode()[0]
        dataset[i].replace(to_replace = "", value = x, inplace = True)
        dataset[i] = dataset[i].astype(int)

    return dataset

# function to devide the data randomly into 80:20 ratio
def random_divide(data):
    # Randomly divide the data into 80% training and 20% testing
    indexes = numpy.random.permutation(len(data))
    trainData = data[indexes[:int(len(data) * 0.8)]]
    testData = data[indexes[int(len(data) * 0.8):]]
    return trainData, testData
    
def main():
    ##### PART 1 #####
    print("********************************************** PART 1 **********************************************")
    # Name of the dataset file
    dataFilePath = 'lung-cancer.data'
    # Read the dataset 
    data = read_data(dataFilePath)
    # Standardise the data
    print("Performing standard scalar normalisation...")
    data = standard_scalar_normalisation(data)
    # Randomly divide the data into 80% training and 20% testing
    print("Randomly dividing the dataset into 80% training and 20% testing...")
    trainData, testData = random_divide(data)
    trainFeatures, trainLabels = encode_categorial_variable(trainData)
    testFeatures, testLabels = encode_categorial_variable(testData)
    print("Done!")
    
    ##### PART 2 #####
    print("********************************************** PART 2 **********************************************")
    # Create a SVM classifier with linear kernel
    print("Training Linear Support Vector Machine")
    linearSVM = svm.SVC(kernel='linear')
    linearSVM.fit(trainFeatures, trainLabels)
    testOutputLinearSVM = linearSVM.predict(testFeatures)
    accuracyLinearSVM = calculate_accuracy(testOutputLinearSVM, testLabels)
    print("Accuracy of Linear SVM: ", accuracyLinearSVM)

    # Create a SVM classifier with Quadratic kernel
    print("Training Quadratic Support Vector Machine")
    quadraticSVM = svm.SVC(kernel='poly', degree=2)
    quadraticSVM.fit(trainFeatures, trainLabels)
    testOutputQuadraticSVM = quadraticSVM.predict(testFeatures)
    accuracyQuadraticSVM = calculate_accuracy(testOutputQuadraticSVM, testLabels)
    print("Accuracy of Quadratic SVM: ", accuracyQuadraticSVM)
    
    # Create a SVM classifier with RBF kernel
    print("Training Radial Basis Function Support Vector Machine")
    radialBasisSVM = svm.SVC(kernel='rbf')
    radialBasisSVM.fit(trainFeatures, trainLabels)
    testOutputRadialBasisSVM = radialBasisSVM.predict(testFeatures)
    accuracyRadialBasisSVM = calculate_accuracy(testOutputRadialBasisSVM, testLabels)
    print("Accuracy of Radial Basis SVM: ", accuracyRadialBasisSVM)
    
    print("Done!")

    print("********************************************** PART 3 **********************************************")
    optimizer = 'sgd'
    learning_rate = 0.001
    batch_size = 32
    print("Training MLP Classifier")
    print("With optimizer: ", optimizer, " learning rate: ", learning_rate, " batch size: ", batch_size)
    
    # Create a MLP classifier with 1 hidden layer of 16 neurons
    print("Training MLP Classifier with 1 hidden layer of 16 neurons")
    MLPClassifier_16 = MLPClassifier(hidden_layer_sizes=(16,), solver=optimizer, learning_rate_init=learning_rate, batch_size=batch_size, max_iter=100000)
    MLPClassifier_16.fit(trainFeatures, trainLabels)
    testOutputMLPClassifier_16 = MLPClassifier_16.predict(testFeatures)
    accuracyMLPClassifier_16 = calculate_accuracy(testOutputMLPClassifier_16, testLabels)
    print("Accuracy of MLP Classifier with 1 hidden layer of 16 neurons: ", accuracyMLPClassifier_16)
    
    # Create a MLP classifier with 2 hidden layer of 256 and 16 neurons
    print("Training MLP Classifier with 2 hidden layer of 256 neurons and 16 neurons")
    MLPClassifier_256_16 = MLPClassifier(hidden_layer_sizes=(256, 16), solver=optimizer, learning_rate_init=learning_rate, batch_size=batch_size, max_iter=100000)
    MLPClassifier_256_16.fit(trainFeatures, trainLabels)
    testOutputMLPClassifier_256_16 = MLPClassifier_256_16.predict(testFeatures)
    accuracyMLPClassifier_256_16 = calculate_accuracy(testOutputMLPClassifier_256_16, testLabels)
    print("Accuracy of MLP Classifier with 2 hidden layer of 256 neurons and 16 neurons: ", accuracyMLPClassifier_256_16)
    
    print("Done!")

    ##### PART 4 #####
    print("********************************************** PART 4 **********************************************")
    bestModel = None
    # Choosing the best model among the 2 MLP Classifiers and observing its behaviur by varying the learning rate
    if accuracyMLPClassifier_16 > accuracyMLPClassifier_256_16:
        bestModel = MLPClassifier_16
        print("Best Model: MLP Classifier with 1 hidden layer of 16 neurons")
        print("Training the above model with different learning rates")
        learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        accuracies = []
        for lr in learning_rates:
            MLPClassifier_16_lr = MLPClassifier(hidden_layer_sizes=(16,), solver=optimizer, learning_rate_init=lr, batch_size=batch_size, max_iter=10000)
            testOutputMLPClassifier_16_lr = MLPClassifier_16_lr.fit(trainFeatures, trainLabels).predict(testFeatures)
            accuracyMLPClassifier_16_lr = calculate_accuracy(testOutputMLPClassifier_16_lr, testLabels)
            accuracies.append(accuracyMLPClassifier_16_lr)
            print("Accuracy of MLP Classifier with learning rate: ", lr, " is: ", accuracyMLPClassifier_16_lr)
    else:
        bestModel = MLPClassifier_256_16
        print("Best Model: MLP Classifier with 2 hidden layer of 256 neurons and 16 neurons")
        print("Training the above model with different learning rates")
        learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        accuracies = []
        for lr in learning_rates:
            MLPClassifier_256_16_lr = MLPClassifier(hidden_layer_sizes=(256, 16), solver=optimizer, learning_rate_init=lr, batch_size=batch_size, max_iter=10000)
            testOutputMLPClassifier_256_16_lr = MLPClassifier_256_16_lr.fit(trainFeatures, trainLabels).predict(testFeatures)
            accuracyMLPClassifier_256_16_lr = calculate_accuracy(testOutputMLPClassifier_256_16_lr, testLabels)
            accuracies.append(accuracyMLPClassifier_256_16_lr)
            print("Accuracy of MLP Classifier with learning rate: ", lr, " is: ", accuracyMLPClassifier_256_16_lr)
    print("Done!")

    # Plot the graph
    plt.plot(learning_rates, accuracies)
    plt.savefig("q2_Figure.png")

    ##### PART 5 #####
    print("********************************************** PART 5 **********************************************")
    # calling forward selection function
    print("Applying forward feature selection")
    feat_cols = forward_selection(data, bestModel)
    print("Best Features are: ",feat_cols)

    ##### PART 6 #####
    print("********************************************** PART 6 **********************************************")
    print("Applying Ensemble learning (Max voting technique) using SVM with Quadratic, SVM with Radial Basis Function and MLP Classifier with 1 hidden layer of 16 neurons")
    
    # calculating the accuracy using ensemble learning model
    accuracy_final = ensemble_learning(quadraticSVM,radialBasisSVM,bestModel, data)
    print("Accuracy of Ensemble learning (Max voting technique) using SVM with Quadratic, SVM with Radial Basis Function and MLP Classifier with 1 hidden layer of 16 neurons: ", accuracy_final)

    
if __name__ == '__main__':
    main()

