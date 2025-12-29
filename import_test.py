from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import gc

datalabel = "NEO"

def data_label():
    return datalabel

def MCNN_data_load(DATA_TYPE):

    neo_train = "/mnt/D/jupyter/juan/neo/datasets/prottrans/pos_train_rag.npy"
    neo_test = "/mnt/D/jupyter/juan/neo/datasets/prottrans/pos_test_rag.npy"
    # neo_test = "/mnt/D/jupyter/juan/neo/public_neo/RAG_TSA.npy"
    # neo_test = "C:/jupyter/globe/Neoantigen_Breast/dataset/esm2/neo_mhc2_test.npy"
    
    other_train = "/mnt/D/jupyter/juan/neo/datasets/prottrans/neg_train_rag.npy"
    other_test = "/mnt/D/jupyter/juan/neo/datasets/prottrans/neg_test_rag.npy"
    # other_test = "/mnt/D/jupyter/juan/neo/public_neo/RAG_TAA.npy"
    # other_test = "C:/jupyter/globe/Neoantigen_Breast/dataset/esm2/other_mhc2_test.npy"

    # Load and shuffle training data
    x_train, y_train = data_load(neo_train, other_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Load and shuffle testing data
    x_test, y_test = data_load(neo_test, other_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    return x_train, y_train, x_test, y_test


def data_load(pos, neg):
    """
    Load positive and negative .npy files and create one-hot labels.
    """
    pos_file = np.load(pos)
    neg_file = np.load(neg)

    pos_label = np.ones(pos_file.shape[0])
    neg_label = np.zeros(neg_file.shape[0])

    x = np.concatenate([pos_file, neg_file], axis=0)
    y = np.concatenate([pos_label, neg_label], axis=0)
    y = tf.keras.utils.to_categorical(y, 2)

    gc.collect()
    return x, y
