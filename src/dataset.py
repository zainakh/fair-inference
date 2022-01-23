import pandas as pd
import numpy as np


def load_adult_dataset(dataset_path='../data/'):
    """
    Loads the UCI adult dataset from the data/ folder and returns
    the train and test sets.
    """
    train = pd.read_csv(dataset_path + 'adult.data', header=None)
    test = pd.read_csv(dataset_path + 'adult.test', skiprows=1, header=None)
    return train, test
