import random
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import glob

import torch
from torch.utils.data import Dataset, DataLoader

def chunk_signals(data, labels, n_chunks):
    """
    Split each signal in data into n_chunks and modify accordingly the labels

    Parameters
    -----------
    data : numpy array of shape (N, L) where N is the number of instances and L
           is the length of the data
    label : numpy array of shape (N, )    
    
    return
    ------
    chuncked_data : ndarray, chunked data of shape (N * n_chunks, L)
    chuncked_labels : ndarray, corresponding labels of shape (N * n_chunks, )

    """
    # Each signal in the ESR dataset has 4097 points
    # 4097 - 1 = 4096 for easy division
    data = data[:, :-1]

    assert data.shape[-1] % n_chunks == 0, "sequence length must be divisable by n_chunks"

    # Split the signals and Tile labels
    chuncked_data = np.concatenate(np.split(data, n_chunks, axis=1), axis=0)
    chuncked_labels = np.tile(labels, n_chunks)

    return chuncked_data, chuncked_labels


def load_bonn_dataset_as_ndarray(dataset_dir, nb_to_letter, n_chunks, shuffle=True,
                                 random_state=42):
    """
    Load original Bonn EEG dataset from txt files as numpy arrays
    
    Args : 
    -------
    dataset_dir  : root directory of the Bonn dataset, which contains five sub-folders, each for a class
    nb_to_letter : dictionary that maps number-format class used in the preprocessed Kaggle version
                    of the dataset to letter-format class used in the original dataset
                    (link to original dataset : 
                    https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)
    n_chunks     : integer, number of chunks that each signal will be split to
    return :
    -------- 
            data  : ndarray of shape (num_samples, sequence_length)
            label : ndarray of shape (num_samples, )
    """
    data = []
    label = []
    for lab_nb in range(len(nb_to_letter)):
        lab_letter = nb_to_letter[lab_nb]
        ext = 'TXT' if lab_letter == 'C' else 'txt' 
        
        txt_paths = glob.glob(os.path.join(dataset_dir, f'set_{lab_letter}', f'*.{ext}'))
        txt_paths.sort()
        print(f'{len(txt_paths)} txt files of set {lab_letter} found')
        for path in txt_paths:
            data.append(np.loadtxt(path))
            label.append(lab_nb)

    data = np.stack(data, axis=0)
    label = np.array(label, dtype=int)
    
    if shuffle:
        np.random.seed(random_state) # make it possible that on each run, same samples are picked as test set 
        indices = np.random.permutation(len(data))
        data = data[indices]
        label = label[indices]
        print(f'*** data and labels are shuffled with seed {random_state} ***')

    # chunk the signals into n_chunks
    if n_chunks > 1:
        data, label = chunk_signals(data, label, n_chunks)
    return data, label


class BonnDataset(Dataset):
    """
    Sub-class of pytorch Dataset for the original ESR dataset
    (https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(-1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


def build_dataset_arrays(csv_path, VAL_RATIO=0.2, TEST_RATIO=0.1, SPLIT_SEED=42):
    """
    Load the preprocessed version of the ESR dataset into DataFrame and perform
    train/valid/test split up
    
    The dataset is available : 
    https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
    """
    ESR = pd.read_csv(csv_path)
    ESR = ESR.drop("Unnamed", axis=1)

    # adjust class label for nn.CrossEntropyLoss, range [0,C)
    ESR['y'] = ESR['y'].replace(5, 0)

    # Train/val/test split up
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO,
                                random_state=SPLIT_SEED)
    for train_index, test_index in split.split(ESR, ESR["y"]):
        strat_train_set = ESR.loc[train_index]
        test_set = ESR.loc[test_index]

    split = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                random_state=SPLIT_SEED)
    strat_train_set = strat_train_set.reset_index(drop=True)
    for train_index, test_index in split.split(strat_train_set, strat_train_set["y"]):
        train_set = strat_train_set.loc[train_index]
        val_set = strat_train_set.loc[test_index]

    X_train = train_set.drop(['y'],axis=1).values
    y_train = train_set['y'].values

    X_val = val_set.drop(['y'],axis=1).values
    y_val = val_set['y'].values

    X_test = test_set.drop(['y'],axis=1).values
    y_test = test_set['y'].values

    return {'X_train' : X_train,
            'y_train' : y_train,
            'X_val'   : X_val,
            'y_val'   : y_val,
            'X_test'  : X_test,
            'y_test'  : y_test,
            }


class EEG_BONN(Dataset):
    """
    Sub-class of pytorch Dataset for the preprocessed version of ESR dataset
    (https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
    """
    def __init__(self, X, y, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype).unsqueeze(-1) # RNN demands shape (N,L,F)
                                            # L = time steps, F = feature dims
        self.y = torch.tensor(y, dtype=torch.long)

        # Standardization
        m = torch.mean(self.X, dim=1, keepdim=True)
        s = torch.std(self.X, dim=1, keepdim=True)
        eps = 1e-7

        self.X = (self.X - m) / (s + eps)

    def __len__(self):
        return self.X.size(0)

    # loads and returns a sample from the dataset at the given index
    def __getitem__(self, ix):
        sample = self.X[ix,:,:]
        label = self.y[ix]

        return sample, label


def get_loader_from_np_arrays(X, y, batch_size, split):
    shuffle = True if split == 'train' else False

    dataset = EEG_BONN(X, y) # Standardization takes place in the class
    if shuffle:
        print('shuffled data loader')
    loader = DataLoader(dataset, batch_size, shuffle=shuffle)

    return loader


