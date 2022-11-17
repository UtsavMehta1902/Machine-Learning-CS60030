####################################################
####    Assignment-2                           #####
####    Unsupervised Learning                  #####
####    Author: Utsav Mehta and Umang Singla   #####
####            (20CS10069)     (20CS10068)    #####
####################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import normalized_mutual_info_score


'''
    Fetch and preprocess the data
'''


def preProcess():

    # Fetching the data
    dataset = pd.read_csv("lung-cancer.data", header=None)

    # Replace missing values with mode of the column
    dataset.replace(to_replace="?", value="", inplace=True)
    for i in [4, 38]:
        x = dataset[i].mode()[0]
        dataset[i].replace(to_replace="", value=x, inplace=True)
        dataset[i] = dataset[i].astype(int)

    # Split the dataset into features and labels
    labels = dataset[0]
    features = dataset.drop([0], axis=1)
    return labels, features


'''
    Class for PCA
    Attributes:
        dataset: dataset to be reduced
        principal_components: principal components of the dataset
    Methods:
        transform_dataset: returns the transformed dataset
        plot_PCA: plots the PCA
'''


class _PCA_:
    def __init__(self, dataset):
        self.dataset = dataset
        self.principal_components = None

    def transform_dataset(self):
        # Apply Standard Scaler Normalization
        self.dataset = StandardScaler().fit_transform(self.dataset)

        # Apply PCA
        self.principal_components = PCA(n_components=0.95)
        self.principal_components.fit(self.dataset)
        pca_dataset = self.principal_components.transform(self.dataset)

        # return the transformed dataset
        return pca_dataset

    def plot_PCA(self):

        # Plot the PCA
        plt.plot(np.cumsum(self.principal_components.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.show()


'''
    Class for K-Means Clustering
    Attributes:
        data: dataset to be clustered
        labels: labels of the dataset
        k_val: number of clusters
        len_data: length of the dataset
        len_features: number of features in the dataset
        cluster_assignment: cluster assignment of each data point
        cluster_labels: labels of each cluster
        cluster_cost: cost of each cluster
    Methods:
        train_dataset: trains the dataset
        test_model: tests the model
        distEuclid: returns the euclidean distance between two points
        update_assignment: updates the cluster assignment
        update_rep: updates the cluster representatives
        update_label: updates the cluster labels
        avg_cost_clusters: calculates the average cluster cost
        convergence: checks if the model has converged
'''


class K_Means_Clustering():
    def __init__(self, data, labels, k_val, cluster_rep):
        self.data = data
        self.labels = labels
        self.k_val = k_val
        self.cluster_rep = cluster_rep
        self.len_data = self.data.shape[0]
        self.len_features = self.data.shape[1]
        self.cluster_assignment = np.zeros(self.len_data, dtype=int)
        self.cluster_label = -1*np.ones(self.k_val, dtype=int)
        self.cluster_cost = []

    def distEuclid(self, val1, val2):

        # Calculate the euclidean distance between two points
        return np.linalg.norm(val1-val2, 2)

    def update_assignment(self):

        # Update the cluster assignment
        for i in range(self.len_data):
            self.cluster_assignment[i] = np.argmin(
                np.array([self.distEuclid(self.data[i], rep) for rep in self.cluster_rep]))

    def update_rep(self):

        # Update the cluster representatives
        for i in range(self.k_val):
            cluster_member = np.where(self.cluster_assignment == i)[0]

            # If the cluster is empty, choose a zero data point
            if cluster_member.shape[0] != 0:
                self.cluster_rep[i] = np.mean(
                    self.data[cluster_member], axis=0)
            else:
                self.cluster_rep[i] = np.zeros(self.len_features)

    def update_label(self):

        # Update the cluster labels
        for i in range(self.k_val):
            cluster_member = np.where(self.cluster_assignment == i)[0]
            if cluster_member.shape[0]:
                self.cluster_label[i] = np.argmax(np.bincount(
                    [self.labels[cluster_member[j]] for j in range(cluster_member.shape[0])]))
            else:
                self.cluster_label[i] = -1

    def avg_cost_clusters(self):

        # Calculate the average cluster cost
        cost_clusters = [self.distEuclid(
            self.data[i], self.cluster_rep[self.cluster_assignment[i]]) for i in range(self.len_data)]
        return np.mean(np.array(cost_clusters))

    def convergence(self):

        # Check if the model has converged, if the cost has not changed by more than 1e-4%
        return np.absolute(self.cluster_cost[-1] - self.cluster_cost[-2]) < 1e-6

    def train_dataset(self, iter_max=100):
        for i in range(iter_max):
            self.update_assignment()
            self.update_rep()
            self.update_label()
            self.cluster_cost.append(self.avg_cost_clusters())
            if i > 1 and self.convergence():
                break
        return

    def test_model(self, test_x):
        clusters_pred = [np.argmin(np.array(
            [self.distEuclid(x, rep) for rep in self.cluster_rep])) for x in test_x]
        labels_pred = self.cluster_label[np.array(clusters_pred)]
        return labels_pred


'''
    Computes the Normalized Mutual Information Score
    Parameters:
        clusters: cluster assignment of the data points
        labels: labels of the data points
'''


def compute_NMI(clusters, labels):

    k = 4
    c = 4
    N = len(clusters)

    labels = labels.to_numpy()

    # calculate cluster and label probabilities
    prob_clus = [np.sum(clusters == i) / N for i in range(k)]
    prob_label = [np.sum(labels == i) / N for i in range(c)]

    # calculate joint probability
    prob_label_clus = np.zeros((c, k))
    for i in range(N):
        prob_label_clus[labels[i], clusters[i]] += 1
    prob_label_clus /= N

    # calculate entropy
    Entropy_cost = -1 * sum([prob_clus[i] * np.log(prob_clus[i])
                            for i in range(k) if prob_clus[i] > 0])
    Entropy_label = -1 * sum([prob_label[i] * np.log(prob_label[i])
                             for i in range(c) if prob_label[i] > 0])

    # calculate mutual information
    Ent_lab_clus = -1 * sum([prob_clus[i] * np.sum([(prob_label_clus[j, i] * np.log(prob_label_clus[j, i]))
                            for j in range(c) if prob_label_clus[j, i] > 0]) for i in range(k)])

    # calculate normalized mutual information
    Mutual_Info = Entropy_label - Ent_lab_clus
    NMI = (2 * Mutual_Info) / (Entropy_label + Entropy_cost)

    return NMI


def main():
    print("######################################----Part 1-----######################################\n\n")
    print("Beginning to Fetch and Process the data")
    labels, features = preProcess()

    print("Beginning Principal Component Analysis")
    pca = _PCA_(features)
    pca_dataset = pca.transform_dataset()

    print("The plot of PCA vs Cumulative Explained Variance is as shown.")
    pca.plot_PCA()

    print("\n######################################----Part 2-----######################################\n\n")

    print("Splitting the Data into Training and Testing Sets")
    X_train, X_test, y_train, y_test = train_test_split(
        pca_dataset, labels, test_size=0.2)
    y_train = [y for y in y_train]

    print("The values of NMI for different values of K in K-Means Clustering are as shown --")
    NMI_scores = []
    for k in range(2, 9):
        init_rep = np.random.uniform(size=(k, X_train.shape[1]))

        KMC = K_Means_Clustering(X_train, y_train, k, init_rep)
        KMC.train_dataset()
        y_pred = KMC.test_model(X_test)
        NMI = round(compute_NMI(y_pred, y_test), 4)
        print("NMI Score for k = ", k, " is ", NMI)
        NMI_scores.append(NMI)

    print("\nNMI score is maximum for k = ", np.argmax(
        NMI_scores)+2, " with NMI score = ", np.max(NMI_scores))

    print("\n######################################----Part 3-----######################################\n\n")
    print("The plot of Normalized Mutual Information vs K is as shown.")
    plt.plot(range(2, 9), NMI_scores)
    plt.xlabel("Number of clusters - k")
    plt.ylabel("NMI score")
    plt.show()


if __name__ == "__main__":
    main()
