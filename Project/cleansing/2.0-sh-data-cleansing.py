# import standard libraries
import numpy as np
import os as os
import pandas as pd

# main function to perform data cleansing
def main():
    # set the working directory to grab the data
    setwd(os.path.join(os.getcwd(), "..", "data", "numerai_dataset"))

    # set a random seed
    np.random.seed(123)

    # load the train and test data
    train_df = load_train()
    test_df = load_test()

    # get all the validation data
    validation = test_df[test_df["data_type"]=="validation"]

    # get all the test data
    test = test_df[test_df["data_type"]=="test"]

    # list of tournaments
    tournaments = ["bernie", "charles", "elizabeth", "jordan", "ken"]

    # set the working directory to spit the data
    setwd(os.path.join(os.getcwd(), "..", "..", "cleansing", "data"))

    for index in range(0, len(tournaments)):
        # get the tournament name
        tournament = tournaments[index]

        # create a directory for storing tournament data if it does not exist already
        if not os.path.exists(tournament):
            os.mkdir(tournament)

        # list of columns to drop
        drop = ["id", "era", "data_type"]

        for all_tour in tournaments:
            if all_tour != tournament:
                drop.append("target_" + all_tour)

        # training set for the tournament
        train_tour = train_df.drop(drop, axis = 1)

        # validation set for the tournament
        valid_tour = validation.drop(drop, axis = 1)

        # test set for the tournament
        test_tour = test.drop(drop, axis = 1)

        # save all three train, validation and test into respective csv files
        train_tour.to_csv(os.path.join(os.getcwd(), tournament, "2." + str(index + 1) + ".1-sh-" + tournament + "-train.csv"), index=False)
        valid_tour.to_csv(os.path.join(os.getcwd(), tournament, "2." + str(index + 1) + ".2-sh-" + tournament + "-valid.csv"), index=False)
        test_tour.to_csv(os.path.join(os.getcwd(), tournament, "2." + str(index + 1) + ".3-sh-" + tournament + "-test.csv"), index=False)

# set the working directory
def setwd(dir):
    os.chdir(dir)

# load the training dataset
def load_train():
    print "Loading train data ..."
    train = pd.read_csv("numerai_training_data.csv", header=0)
    print "Loaded."
    return train

# load the testing dataset
def load_test():
    print "Loading test data ..."
    test = pd.read_csv("numerai_tournament_data.csv", header=0)
    print "Loaded."
    return test

if __name__ == "__main__":
    main()
