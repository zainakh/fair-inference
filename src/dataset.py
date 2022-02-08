from datetime import date
import pandas as pd
import numpy as np
import pickle
from collections import namedtuple
from sklearn.preprocessing import StandardScaler


class SyntheticDataset:
    def __init__(self, old_path=None, save_path='../data/synthetic/', gamma_shift_src=[0.0], gamma_shift_tar=[0.0],
                 gamma_A=0.0, C_src=0, C_tar=1, N=1000, verbose=False):

        if old_path:
            with open(old_path, 'rb') as f:
                data_src, data_tar, sensitive_feature, non_separating_feature = pickle.load(f)
        else:
            data_src, data_tar, sensitive_feature, non_separating_feature = \
                create_synthetic_src_and_tar_dataset(gamma_shift_src, gamma_shift_tar, gamma_A,
                                                     C_src, C_tar, N, verbose)

        self.source = data_src
        self.target = data_tar
        self.sensitive_feature = sensitive_feature
        self.outcome_feature = non_separating_feature
        self.datetime = date.today().strftime("%m-%d-%Y")
        if not old_path:
            self.save_dataset(save_path + self.datetime)

    def save_dataset(self, path):
        filename = path + '.pkl'
        data = (self.source, self.target, self.sensitive_feature, self.outcome_feature)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'{filename} saved')


def create_synthetic_src_and_tar_dataset(gamma_shift_src=[0.0], gamma_shift_tar=[0.0], gamma_A=0.0,
                                         C_src=0, C_tar=1, N=1000, verbose=False):
    """
    Create both source and target datasets with varying shifts such that classifiers that learn on the
    source datasets will have to be generalizable in order to perform well on the target (test) datasets

    :param gamma_shift_src: List of shifts for source dataset (reasonable shifts are between 0-15)
    :param gamma_shift_tar: List of shifts for target dataset
    :param gamma_A: Gamma for generating sensitive variable
    :param C_src: Context for source dataset
    :param C_tar: Context for target dataset
    :param N: Number of samples per dataset
    :param verbose: If the DGP should be verbose
    :return: Source and target datasets, sensitive feature, and non-separating feature
    """
    data_src, sensitive_feature, non_separating_feature = \
        create_synthetic_flu_dataset(gamma_shift_src, gamma_A, C_src, N, verbose)
    data_tar, _, _ = create_synthetic_flu_dataset(gamma_shift_tar, gamma_A, C_tar, N, verbose)

    return data_src, data_tar, sensitive_feature, non_separating_feature


def create_synthetic_flu_dataset(gamma_shifts=[0.0], gamma_A=0.0, C=0, N=1000, verbose=False):
    """
    Modified version of synthetic domain adaption dataset used in https://github.com/ChunaraLab/fair_domain_adaptation.

    Variables
    C: context (0 or 1)
    A: age group (0 or 1)
    R: risk
    T: temperature
    Y: flu (0 or 1)

    :param gamma_shifts: Shift in dataset construction
    :param gamma_A: Shift in sensitive variable
    :param C: Context variable (0 or 1)
    :param N: Number of data samples (int)
    :param verbose: If creation of dataset should be verbose
    :return: dataset, sensitive_feature, and non_separating_feature
    """

    def random_logit(x):
        z = 1. / (1 + np.exp(-x))
        s = np.random.binomial(n=1, p=z)

        return 2 * s - 1

    # Regression coefficients
    gamma_AC = 0.2
    gamma_TC = 0.2

    gamma_RA = -0.1
    gamma_YA = -0.8
    gamma_YR = 0.8
    gamma_TY = 0.8
    gamma_TR = 0.1
    gamma_TA = -0.8

    scale_T = 1.0
    scale_e = 0.64
    scale_A = 1.0
    scale_Y = 1.0

    # Source datasets
    data = []
    for gamma_shift in gamma_shifts:
        C_vec = np.repeat(a=C, repeats=N)
        A = random_logit(
            scale_A * (gamma_A + gamma_shift * gamma_AC * C_vec + np.random.normal(loc=0.0, scale=scale_e, size=N)))
        R = gamma_RA * A + np.random.normal(loc=0.0, scale=1 + scale_e, size=N)  # N(0,1)
        Y_src = random_logit(scale_Y * (gamma_YA * A + gamma_YR * R + np.random.normal(loc=0.0, scale=scale_e, size=N)))
        T = gamma_TY * Y_src + gamma_TR * R + gamma_TA * A + \
            np.random.normal(loc=0.0,
                             scale=scale_T + gamma_shift * gamma_TC * C_vec,
                             size=N)
        Y_src = (Y_src + 1) / 2
        X_src = np.stack([A, R, T], axis=1)
        data.append((gamma_shift, X_src, Y_src))

    sensitive_feature = 0  # A
    non_separating_feature = 2  # T

    if verbose:
        print("Coefficients: g_shifts:{},gA:{},gAC:{},gRA:{},gYA:{},gYR:{},gTC:{},gTY:{},gTR:{},gTA:{}" \
              .format(gamma_shifts, gamma_A, gamma_AC, gamma_RA, gamma_YA, gamma_YR, gamma_TC,
                      gamma_TY, gamma_TR, gamma_TA))

    return data, sensitive_feature, non_separating_feature


def load_synthetic_flu_dataset(data, target, data_test, target_test, smaller=False, scaler=True):
    """
    Loads a target or source dataset that is split into train/test components for use in training or prediction.

    :param data: List of size N containing features described in flu dataset
    :param target: Target variable to predict, Y (data is split into gamma, X, Y)
    :param data_test: The test (target) dataset of covariates (X)
    :param target_test: The test (target) array containing the target variable (Y)
    :param smaller: If a smaller version of the dataset should be used
    :param scaler: If standard_scaler() should be used
    :return:
    """
    len_train = len(data[:, -1])
    if scaler:
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data_test = scaler.transform(data_test)

    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(data[:len_train // 20, :-1], target[:len_train // 20])
        data_test = namedtuple('_', 'data, target')(data_test, target_test)
    else:
        data = namedtuple('_', 'data, target')(data, target)
        data_test = namedtuple('_', 'data, target')(data_test, target_test)

    return data, data_test
