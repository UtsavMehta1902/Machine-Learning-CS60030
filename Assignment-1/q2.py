####################################################
####    Assignment-1                           #####
####    Naive Bayes Algorithm                  #####
####    Author: Utsav Mehta and Umang Singla   #####
####            (20CS10069)     (20CS10068)    #####
####################################################


import time
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

"""
    Names of the columns in the dataset
"""
COLS = [
    [
        "Gender",
        "Age",
        "Driving_License",
        "Region_Code",
        "Previously_Insured",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Annual_Premium",
        "Policy_Sales_Channel",
        "Vintage",
    ],
    [
        "Age",
        "Region_Code",
        "Annual_Premium",
        "Policy_Sales_Channel",
        "Vintage",
    ],
    [
        "Gender",
        "Driving_License",
        "Previously_Insured",
        "Vehicle_Age",
        "Vehicle_Damage",
    ],
]


class DataProcess:

    """
    Remove the data points with abrupt values

    @param data: The dataset
    @return: The dataset with removed outliers
    """

    def remove_outliers(self, data):
        stats = []
        new_data = []
        for col in COLS[0]:
            sd_feature = data[col].std()
            mean_feature = data[col].mean()
            stats.append((mean_feature, sd_feature))

        for idx, row in data.iterrows():
            take = True
            for i in range(1, len(row) - 1):
                mean, sd = stats[i - 1]
                if row[i] > mean + 3 * sd or row[i] < mean - 3 * sd:
                    take = False
                    break
            if take:
                new_data.append(row)
        return pd.DataFrame(new_data)

    """
        Normalize some attribute values

        @param data: The dataset
        @return: The dataset with normalized values
    """

    def normalize_features(self, data):
        cols_normalize = [
            "Age",
            "Region_Code",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
        ]
        for col in COLS[0]:
            if col in cols_normalize:
                data[col] = data[col] / (data[col].max() - data[col].min())
        return data

    """
        Encoding the categorical variables

        @param data: The dataset
        @return: The dataset with encoded values
    """

    def dataEncoder(self, data):
        # Applying Label Encoding to the categorical variables
        label_encoder = LabelEncoder()
        data["Gender"] = label_encoder.fit_transform(data["Gender"])
        data["Vehicle_Age"] = label_encoder.fit_transform(data["Vehicle_Age"])
        data["Vehicle_Damage"] = label_encoder.fit_transform(data["Vehicle_Damage"])
        return data


class Utility:

    """
    Segregate the classes in the dataset according to the Response (0 or 1)

    @param X_train: The training dataset attributes
    @param y_train: The training dataset outputs
    @return: The segregated classes
    """

    def class_segregation(self, X_train, y_train):
        segregate_classes = {}
        for idx, row in X_train.iterrows():
            class_value = y_train[idx]
            if class_value not in segregate_classes:
                segregate_classes[class_value] = []
            segregate_classes[class_value].append(row)
        return segregate_classes

    """
    Calculate the mean and standard deviation of each attribute in the dataset

    @param X_train: The training dataset attributes
    @return: The mean and standard deviation of each attribute as a list
    """

    def statistics_data(self, X_train):
        as_df = pd.DataFrame(X_train)
        stats = []
        for col in COLS[0]:
            if col in COLS[1]:
                stats.append((len(as_df[col]), as_df[col].mean(), as_df[col].std()))
            else:
                stat_per_val = {}
                for val in as_df[col].unique():
                    stat_per_val[val] = as_df[as_df[col] == val][col].count()
                stats.append((len(as_df[col]), stat_per_val))
        return stats

    """
        Calculate the mean and standard deviation of each Response value (0 and 1)

        @param X_train: The training dataset attributes
        @param y_train: The training dataset outputs
        @return: The mean and standard deviation of each Response value as a dictionary
    """

    def class_statistics(self, X_train, y_train):
        class_segregate = self.class_segregation(X_train, y_train)
        stats = {}
        for class_value, rows in class_segregate.items():
            stats[class_value] = self.statistics_data(rows)
        return stats

    """
    Calculate the accuracy of the model, using the predicted outputs and the actual outputs

    @param y_test: The actual outputs of the testing dataset
    @param y_pred: The predicted outputs of the testing dataset
    @return: The accuracy of the model (in percentage)
    """

    def accuracy_score(self, y_test, predictions):
        works = 0
        y_test = y_test.to_numpy()
        for i in range(len(predictions)):
            prob = predictions[i]
            if prob[0] > prob[1] and y_test[i] == 0:
                works += 1
            elif prob[0] < prob[1] and y_test[i] == 1:
                works += 1
        return works * 100 / len(predictions)


class Probability_Estimators:
    """
    Calculate the probability of occurrence of a data value using Gaussian Naive Bayes

    @param x: The data value
    @param mean: The mean of the attribute
    @param sd: The standard deviation of the attribute
    @return: The probability of occurrence of the data value
    """

    def probability_calculate_gnb(self, x, mean, sd):
        exponent = np.exp(-(((x - mean) / sd) ** 2 / float(2)))
        return (1 / (np.sqrt(2 * np.pi) * sd)) * exponent

    """
        Calculate the probability of occurrence of a data value using Naive Bayes

        @param x: The data value
        @param tot: The total number of data points with the class value
        @param stat: The number of data points in training dataset with the class value and the data value(x)
        @param alpha: The smoothing factor used for laplace correction
        @return: The probability of occurrence of data value
    """

    def probability_calculate_nb(self, x, tot, stat, alpha):
        return (stat[x] + alpha) / float(tot + alpha * len(COLS[0]))


class NaiveBayes:

    """
    Calculate the probability of occurrence of each data point in each class

    @param class_stats: The mean and standard deviation of each attribute in each class
    @param X_test: The testing dataset attributes
    @return: The probability of occurrence of each data point in each class
    """

    def calc_probability_class(self, class_stats, X_test, prob_est, alpha):
        total_sample_size = sum(
            [class_stats[class_value][0][0] for class_value in class_stats]
        )
        prob_row = []
        for idx, row in X_test.iterrows():
            probabilities = {}
            for class_value, stats_data in class_stats.items():
                probabilities[class_value] = stats_data[0][0] / total_sample_size

                for i in range(len(stats_data)):
                    if COLS[0][i] in COLS[1]:
                        count, mean, sd = stats_data[i]
                        probabilities[
                            class_value
                        ] *= prob_est.probability_calculate_gnb(row[i + 1], mean, sd)
                    else:
                        tot, stat = stats_data[i]
                        probabilities[class_value] *= prob_est.probability_calculate_nb(
                            row[i + 1], tot, stat, alpha
                        )
            prob_row.append(probabilities)
        return prob_row

    """
        Split the dataset into 10 folds

        @param X_train: The training dataset attributes
        @param folds: The number of folds to split the dataset into
        @return: The 10 folds of the dataset
    """

    def split_cross_validation(self, X_train, folds):
        X_train = X_train.sample(frac=1)
        splits = []
        for i in range(folds):
            splits.append(
                X_train.iloc[
                    i * len(X_train) // folds : (i + 1) * len(X_train) // folds
                ]
            )
        return splits

    """
        Apply the naive bayes algorithm on the dataset

        @param X_train: The training dataset attributes
        @param y_train: The training dataset outputs
        @param X_test: The testing dataset attributes
        @return: The predicted outputs of the testing dataset
                and the statistics of the classes formed (Response - 0 and 1)
    """

    def naive_bayes(self, X_train, y_train, X_test, utils, prob_est, alpha):
        class_stats = utils.class_statistics(X_train, y_train)
        prob_row = self.calc_probability_class(class_stats, X_test, prob_est, alpha)
        return prob_row, class_stats

    """
        Create the k-fold cross validation model, uses naive bayes algorithm, and split the dataset into 10 folds
        
        @param X_train: The training dataset attributes
        @param folds: The number of folds to split the dataset into
        @param y_train: The training dataset outputs
        @return: The best accuracy result of the model, the statistics of the classes formed (Response - 0 and 1)
    """

    def naive_bayes_with_k_fold(self, X_train, folds, y_train, utils, prob_est):
        new_data = X_train.assign(Response=y_train)
        splits = self.split_cross_validation(new_data, folds)
        split_response = []
        for split in splits:
            split_response.append(split["Response"])
            split.drop(["Response"], axis=1, inplace=True)
        best_accuracy = -1
        best_summary = None

        for fold in range(folds):
            X_test = splits[fold]
            X_train = pd.concat([splits[i] for i in range(folds) if i != fold])
            y_train = pd.concat([split_response[i] for i in range(folds) if i != fold])
            y_test = split_response[fold]
            prediction, summary = self.naive_bayes(
                X_train, y_train, X_test, utils, prob_est, 0
            )
            accuracy = utils.accuracy_score(y_test, prediction)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_summary = summary

        return best_summary


def main():

    start = time.time()

    # Declaring the datatypes of various attributes to avoid warnings
    d_type = {
        "id": int,
        "Gender": "string",
        "Age": int,
        "Driving_License": int,
        "Region_Code": int,
        "Previously_Insured": int,
        "Vehicle_Age": "string",
        "Vehicle_Damage": "string",
        "Annual_Premium": int,
        "Policy_Sales_Channel": "string",
        "Vintage": int,
        "Response": bool,
    }

    # Read the CSV dataset file
    file = open("Dataset_C.csv", "r")

    print("\n\nReading the dataset...")
    data = pd.read_csv(file, usecols=d_type)
    print("Time taken to read the dataset: ", time.time() - start, " seconds")
    print("Dataset read successfully")

    # Generate objects for the classes
    utils = Utility()
    data_processor = DataProcess()
    prob_est = Probability_Estimators()

    print("\n\nPreprocessing the dataset...")
    time_temp = time.time()

    # Encode the categorical attributes
    data = data_processor.dataEncoder(data)

    # Remove the outlier data values from the dataset
    data = data_processor.remove_outliers(data)

    # Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(["Response"], axis=1), data["Response"], test_size=0.2, random_state=0
    )

    # Normalize the dataset
    X_train = data_processor.normalize_features(X_train)
    X_test = data_processor.normalize_features(X_test)

    print("Time taken to preprocess the dataset: ", time.time() - time_temp, " seconds")
    print("Dataset preprocessed successfully")

    print("\n\nThe final set of features formed are: \n")
    cols = data.columns
    for i in range(1, len(cols)-1):
        print(i, " - ", cols[i])
    
# ------------------------------------------------------------------------------------------------------

    print("\n\nApplying Naive Bayes algorithm with 10-fold cross correction...")
    time_temp = time.time()

    # Create the naive bayes object from the corresponding class
    naive_bayes = NaiveBayes()
    n_folds = 10  # The number of folds to split the dataset into

    # Apply the naive bayes algorithm with k-fold repetition on the dataset
    best_summary = naive_bayes.naive_bayes_with_k_fold(
        X_train, n_folds, y_train, utils, prob_est
    )
    y_test = pd.DataFrame(y_test)

    # Calculate the predicitions of the model on the test dataset
    prob_row = naive_bayes.calc_probability_class(best_summary, X_test, prob_est, 0)

    # Calculate the accuracy of the model
    accuracy = utils.accuracy_score(y_test, prob_row)
    time_temp = time.time() - time_temp
    print(
        "Time taken to apply Naive Bayes algorithm with 10-fold cross correction: ",
        time_temp,
        " seconds",
    )
    print("Accuracy of the model: ", accuracy)

# ------------------------------------------------------------------------------------------------------

    time_temp = time.time()
    print("\n\nApplying Naive Bayes algorithm with Laplace smoothing...")
    n_folds = 1
    alpha = 1

    # Apply the naive bayes algorithm with laplace correction on the dataset
    prob_row, _ = naive_bayes.naive_bayes(
        X_train, y_train, X_test, utils, prob_est, alpha
    )

    # Calculate the accuracy of the model
    accuracy = utils.accuracy_score(y_test, prob_row)
    time_temp = time.time() - time_temp
    print(
        "Time taken to apply Naive Bayes algorithm with Laplace smoothing: ",
        time_temp,
        " seconds",
    )
    print("Accuracy of the model: ", accuracy)

# ------------------------------------------------------------------------------------------------------
    print("\n\nTime taken to run the program: ", time.time() - start, " seconds\n\n")


if __name__ == "__main__":
    orig_stdout = sys.stdout
    with open("output_q2.txt", "w") as file:
        sys.stdout = file
        main()
        sys.stdout = orig_stdout
