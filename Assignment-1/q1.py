####################################################
####    Assignment-1                           #####
####    Decision Tree Algorithm                #####
####    Author: Utsav Mehta and Umang Singla   #####
####            (20CS10069)     (20CS10068)    #####
####################################################


import queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class Node:
    """
    Node class for the decision tree.
    """

    def __init__(self,attribute,depth=0,label=None):
        self.attribute = attribute  # attribute of node according to which further division is done
        self.children = {}          # dictionary of children of the node
        self.label = label          # label of the node if it is a leaf
        self.depth = 0              # depth of the node in the tree
        self.id = None              # id of the node in the tree
    
    def addChild(self,x):
        """
        method to add a child to the node
        """
        self.children[x[0]] = x[1]
    
    def isLeaf(self):
        """
        method to check if the node is a leaf
        Returns:
        True if the node is a leaf, False otherwise
        """
        if len(self.children)==0:
            return True
        return False

    def format_string(self):
        """
        Generates the string to be displayed in each node while printing the tree.
        Returns:
            str: A string to be displayed, depending on whether the node is a leaf or not.
        """
        if self.isLeaf():
            outcome = 'Yes' if self.label == 1 else 'No'
            return f'{self.attribute}\n{outcome}'
        else:
            return "Test"


def getEntropy(y):
    """
    function to calculate the entropy of a given dataset
    Input: pd.Series (Collection of 0's and 1's)
    Returns: float (Entropy)
    """
    total = len(y)
    valCnt = y.value_counts()
    ans = 0
    for i in range(len(valCnt)):
        tmp = valCnt[valCnt.index[i]]/total
        ans -= tmp*np.log(tmp)
    return ans

def getIG(x,y, attr):
    """
    Function to calculate the information gain of a given attribute
    Input:
        x: pd.DataFrame (Features)
        y: pd.Series    (Labels)
        attr: str       (Attribute)
    Returns:
        float           (Information Gain)
    """
    initialEntropy = getEntropy(y=y)
    total = len(x[attr])
    valCnt = x[attr].value_counts()
    finalEntropy = 0
    entropies = []
    for i in range(len(valCnt)):
        yi = y[x[attr]==valCnt.index[i]]
        entropies.append(getEntropy(yi))
    for i in range(len(valCnt)):
        tmp = valCnt[valCnt.index[i]]/total
        finalEntropy += tmp*entropies[i]
    return initialEntropy - finalEntropy

def getInfoGainList(x,y,attrs):
    """
    Function to calculate the information gain of all attributes
    Input:
        x: pd.DataFrame (Features)
        y: pd.Series    (Labels)
        attrs: list     (List of attributes)
    Returns:
        list            (List of information gain of all attributes)
    """
    infoGain = []
    for attr in attrs:
        infoGain.append(getIG(x,y,attr))
    return infoGain

def getNextNode(x,y,depth=0,max_depth=15):
    """
    Function to get the next node in the decision tree
    Input:
        x: pd.DataFrame   (Features)
        y: pd.Series      (Labels)
        depth: int        (Depth of the node)
        max_depth: int  (Maximum depth of the tree)
    Returns:
        Node            (Next node in the decision tree)
    """
    if depth == max_depth:
        return None

    # All current Attributes
    attributes = x.columns.values
    if(len(attributes)==0):
        return None
    
    if(len(attributes)==1):
        valCnt = y.value_counts()
        n = Node(attribute=attributes[0])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        n.depth = depth
        return n
    
    # find information gain of all attributes 
    InfoGainPerAttr = getInfoGainList(x,y,attributes)
    
    # find attribute with maximum information gain
    i_max = np.argmax(InfoGainPerAttr)

    if InfoGainPerAttr[i_max]<0.0001:
        valCnt = y.value_counts()
        n = Node(attribute=attributes[i_max])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        n.depth = depth
        return n
    
    # create a node with that as attribute
    n = Node(attribute=attributes[i_max])
    valCnt = y.value_counts()
    n.label = valCnt.index[0]
    n.depth = depth
    # find all dataSet seperated based on attribute with max information gain
    valCnt = x[attributes[i_max]].value_counts()
    for i in range(len(valCnt)):
        xi = x[x[attributes[i_max]]==valCnt.index[i]]
        yi = y[x[attributes[i_max]]==valCnt.index[i]]
        xi = xi.drop(columns=attributes[i_max])
        child = getNextNode(x=xi,y=yi,depth=depth+1,max_depth=max_depth)
        if child is not None:
            n.addChild((valCnt.index[i],child))
    return n

def decisionTree(TrainData,max_depth=15):
    """
    Function to build the decision tree
    Input:
        TrainData: pd.DataFrame (Training Data)
        max_depth: int         (Maximum depth of the tree)
    Returns:
        Node                 (Root node of the decision tree)
    """
    x = TrainData.drop(columns="Response")
    y = TrainData["Response"]
    return getNextNode(x,y,max_depth=max_depth,depth=0)

def find_depth(root):
    """
    Function to find the depth of the tree
    Input:
        root: Node (Root node of the tree)
    Returns:
        int       (Depth of the tree)
    """
    if root is None:
        return 0
    if root.isLeaf():
        return 1
    depth = 0
    for child in root.children.values():
        depth = max(depth,find_depth(child))
    return depth+1

def trainTestSplit(dataset, testRatio,shuffle=True):
    """
    Function to split the dataset into train and test
    Input:
        dataset: pd.DataFrame (Dataset)
        testRatio: float   (Ratio of test data)
        shuffle: bool       (Whether to shuffle the dataset or not)
    Returns:
        Tuple(pd.DataFrame, pd.DataFrame)
    """
    training_data = dataset.sample(frac=testRatio)
    test_data = dataset.drop(training_data.index)
    return (training_data,test_data)

def predict(decisionTreeNode, x):
    """
    Function to predict the label of a given data point
    Input:
        decisionTreeNode: Node  (Root node of the decision tree)
        x: pd.Series        (Data point)
    Returns:
        int                 (Predicted label)
    """
    if decisionTreeNode.isLeaf():
        return decisionTreeNode.label
    x_bar = x[decisionTreeNode.attribute]
    if x_bar in decisionTreeNode.children:
        newNode = decisionTreeNode.children[x_bar]
    else :
        return decisionTreeNode.label
    return predict(newNode,x)

def split_data_to_labels(dataset):
    """
    Function to split the dataset into features and labels
    Input:
        dataset: pd.DataFrame   (Dataset)
    Returns:
        Tuple(pd.DataFrame, pd.Series)
    """
    data = dataset.drop(columns=['Response'],axis = 1)
    labels = dataset['Response']
    return (data, labels)

def get_pred_accuracy(tree, test):
    """
    Function to get the accuracy of the model
    Input:
        tree: Node          (Root node of the decision tree)
        test: pd.DataFrame  (Test data)
    Returns:
        float               (Accuracy of the model)
    """
    test_data, test_labels = split_data_to_labels(test)
    preds = pd.Series([predict(tree, row) for row in test_data.to_dict(orient='records')])
    accuracy = np.mean(preds.reset_index(drop=True) == test_labels.reset_index(drop=True)) * 100
    return (preds, accuracy)

def train(trainSet):
    """
    Function to train the model
    Input:
        trainSet: pd.DataFrame      (Training data)
    Returns:
        Node                        (Root node of the decision tree)
    """
    decisionTreeRoot = decisionTree(trainSet)
    if decisionTreeRoot is None:
        return
    return decisionTreeRoot

def get_best_tree(dataset):
    """
    Function to get the best tree
    Input:
        dataset: pd.DataFrame
    Returns:
        Tuple(Node, float)
    """
    best_tree = Node("test")
    best_accuracy = 0

    for i in range(0,10):
        trainSet, testSet = trainTestSplit(dataset=dataset,testRatio=0.8)
        decisionTreeRoot = train(trainSet)
        _,accuracy = get_pred_accuracy(decisionTreeRoot,testSet)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tree = decisionTreeRoot

    return (best_tree, best_accuracy)

def prune(tree_node,accuracy,validation_set):
    """
    Function to prune the tree
    Input:
        tree_node: Node
        accuracy: float
        validation_set: pd.DataFrame
    Returns:
        Void
    """
    if tree_node.isLeaf():
        return
    childrn = tree_node.children
    tree_node.children = {}
    _,accuracy_without_pruning = get_pred_accuracy(tree_node,validation_set)
    if accuracy_without_pruning > accuracy:
        return
    tree_node.children = childrn
    for child in tree_node.children.values():
        prune(child,accuracy,validation_set)

def print_tree(root):
    """
    Prints the decision tree in a neat format using the graphviz package.
    Args:
        file (str): File name with which the image of the tree is to be saved. 
    """
    queue = []
    queue.append(root)
    depth = -1
    while queue:
        cur = queue.pop(0)
        if cur.depth != depth:
            depth = cur.depth
            print("######### Level ",depth," #########")
        print("Attribute of the node, ",cur.attribute)
        print("\n")
        for child in cur.children.values():
            queue.append(child)



def vary_depth_nodes(dataset):
    """
    Function to vary the depth of the tree and plot the accuracy
    Input:
        dataset: pd.DataFrame
    Returns:
        Void
    """
    depths = []
    accuracy = []
    for i in range(1,11):
        decisionTreeRoot = decisionTree(dataset,max_depth=i)
        depth = find_depth(decisionTreeRoot)
        accuracy.append(get_pred_accuracy(decisionTreeRoot,dataset)[1])
        depths.append(depth)

    depths = np.array(depths)
    accuracy = np.array(accuracy)
    plt.plot(depths,accuracy)
    plt.title("Depth vs Accuracy")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.savefig('Depth_vs_Accuracy.png')
    plt.show()
    

def segment_data(dataset):
    """
    Function to convert the continous data into categorical data
    Input:
        dataset: pd.DataFrame
    Returns:
        Void
    """
    dataset.drop(columns=['id'],inplace = True)
    dataset.replace("Male", 1, inplace=True)
    dataset.replace("Female", 0, inplace=True)
    dataset.replace("No", 0, inplace=True)
    dataset.replace("Yes", 1, inplace=True)
    dataset.replace("> 2 Years", 2, inplace=True)
    dataset.replace("1-2 Year", 1, inplace=True)
    dataset.replace("< 1 Year", 0, inplace=True)

    # Coverting the continous data into categorical data
    m = dataset['Age'].min()
    M = dataset['Age'].max()
    width = (M-m+1)/10
    for i in range(10):
        dataset['Age'] = dataset['Age'].replace(range(int(m+i*width),int(m+(i+1)*width)),i)

    m = dataset['Region_Code'].min()
    M = dataset['Region_Code'].max()
    width = (M-m+1)/10
    for i in range(10):
        dataset['Region_Code'] = dataset['Region_Code'].replace(range(int(m+i*width),int(m+(i+1)*width)),i)

    m = dataset['Annual_Premium'].min()
    M = dataset['Annual_Premium'].max()
    width = (M-m+1)/10
    for i in range(10):
        dataset['Annual_Premium'] = dataset['Annual_Premium'].replace(range(int(m+i*width),int(m+(i+1)*width)),i)

    m = dataset['Policy_Sales_Channel'].min()
    M = dataset['Policy_Sales_Channel'].max()
    width = (M-m+1)/10
    for i in range(10):
        dataset['Policy_Sales_Channel'] = dataset['Policy_Sales_Channel'].replace(range(int(m+i*width),int(m+(i+1)*width)),i)

    m = dataset['Vintage'].min()
    M = dataset['Vintage'].max()
    width = (M-m+1)/10
    for i in range(10):
        dataset['Vintage'] = dataset['Vintage'].replace(range(int(m+i*width),int(m+(i+1)*width)),i)
    
def main(): 
    """
    Main function
    """
    dataset = pd.read_csv('Dataset_C.csv')
    segment_data(dataset)

    print("Finding Desicion Tree with Best Accuracy.....\n")
    decisionTree_root, best_accuracy = get_best_tree(dataset)
    print("Depth of the Desicion Tree is:",find_depth(decisionTree_root))
    print("Accuracy of Decision Tree is: ",best_accuracy)
    print("\n")

    print("Pruning the Decision Tree.....\n")
    train_data, test_data = trainTestSplit(dataset=dataset,testRatio=0.8)
    prune(decisionTree_root,best_accuracy,test_data)

    _,accuracy = get_pred_accuracy(decisionTree_root,test_data)
    print("Accuracy of the Decision Tree after pruning: ",accuracy)
    print("Depth of the Desicion Tree after pruning is:",find_depth(decisionTree_root))
    print("\n")

    print("Printing Pruned Tree.....\n")
    print_tree(decisionTree_root)
    print("\n")

    print("Varying the depth of the tree and plotting the accuracy.....\n")
    vary_depth_nodes(dataset)


if __name__ == "__main__":
    original_stdout = sys.stdout
    with open('output_q1.txt','w') as f:
        sys.stdout = f
        main()
        sys.stdout = original_stdout
