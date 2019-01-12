# import standard libraries
import numpy as np
import os as os
import pandas as pd

# main function to perform data exploration
def main():
    # set the working directory
    setwd()

    # set a random seed
    np.random.seed(123)

    # explore the training data
    explore_train()

    # explore the testing data
    explore_test()

# set the working directory
def setwd():
    os.chdir(os.path.join(os.getcwd(), "..", "data", "numerai_dataset"))

# load the training dataset
def load_train():
    print "Loading train data ..."
    train = pd.read_csv("numerai_training_data.csv", header=0)
    print "Loaded."
    return train

# explore the training data
def explore_train():
    # load the training data
    train_data = load_train()

    # show train data information
    print "Shape of train data: ", train_data.shape
    print "Columns of train data: ", list(train_data.columns)
    print "Describe train data: ", train_data.describe()
    print "Check for null values in the train data: ", train_data.isnull().any()

# load the testing dataset
def load_test():
    print "Loading test data ..."
    test = pd.read_csv("numerai_tournament_data.csv", header=0)
    print "Loaded."
    return test

# explore the testing data
def explore_test():
    # load the testing data
    test_data = load_test()

    # show test data information
    print "Shape of test data: ", test_data.shape
    print "Columns of test data: ", list(test_data.columns)
    print "Describe test data: ", test_data.describe()
    print "Check for null values in the test data: ", test_data.isnull().any()

if __name__ == '__main__':
    main()
