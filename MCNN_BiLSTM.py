import h5py
import os
import pickle

from tqdm import tqdm
from time import gmtime, strftime

import numpy as np
import math

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow import keras

##

from sklearn.model_selection import KFold

import gc

import time
from sklearn.model_selection import KFold

import import_test as load_data
import argparse

# Argument Parsing
parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-maxseq", "--MAXSEQ", type=int, default=500)
parser.add_argument("-f", "--FILTER", type=int, default=256)
parser.add_argument("-w", "--WINDOW", nargs='+', type=int, default=[2, 4, 6])
parser.add_argument("-nf", "--NUM_FEATURE", type=int, required=True, help="Number of features (e.g., 1024 for ProtTrans)")
parser.add_argument("-hi", "--HIDDEN", type=int, default=500)
parser.add_argument("-drop", "--DROPOUT", type=float, default=0.7)
parser.add_argument("-ep", "--EPOCHS", type=int, default=20)
parser.add_argument("-lstm_units", "--LSTM_UNITS", type=int, default=64)
parser.add_argument("-df", "--DATA_FEATURE", type=str, default="pt")
parser.add_argument("-imb", "--imbalance_mod", type=str, default="None", help="Imbalance method: 'SMOTE', 'ADASYN', 'RANDOM'")
parser.add_argument("-csv", "--csv_path", type=str, default="MSCNN_log.csv")
parser.add_argument("-test", "--test_path", type=str, default="IndependentTest.csv")
parser.add_argument("-k", "--KFold", type=int, default=5)
parser.add_argument("-validation_mode", "--validation_mode", type=str, default="cross")
args = parser.parse_args()

# Constants
MAXSEQ = args.MAXSEQ
NUM_FILTER = args.FILTER
WINDOW_SIZES = args.WINDOW
csv_file_path = args.csv_path
ind_file_path = args.test_path
DATA_FEATURE = args.DATA_FEATURE
DROPOUT = args.DROPOUT
NUM_HIDDEN = args.HIDDEN
IMBALANCE = args.imbalance_mod
MODE = args.validation_mode
BATCH_SIZE = 256
NUM_CLASSES = 2
CLASS_NAMES = ['Negative', 'Positive']
NUM_FEATURE = args.NUM_FEATURE
EPOCHS = args.EPOCHS
LSTM_UNITS = args.LSTM_UNITS
K_Fold = args.KFold

print("FEATURE:", DATA_FEATURE)
print("NUM_FILTER:", NUM_FILTER)
print("WINDOW_SIZES:", WINDOW_SIZES)
print("IMBALANCE:", IMBALANCE)

import datetime

write_data=[]
a=datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(DATA_FEATURE)
write_data.append(BATCH_SIZE)
write_data.append(NUM_HIDDEN)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
write_data.append(MODE)
write_data.append(IMBALANCE)


def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))

import os
import pickle
import time

def SAVEROC(fpr, tpr, AUC):
    data_to_save = {
        "fpr": fpr,
        "tpr": tpr,
        "AUC": AUC
    }

    # Use relative path
    folder = "./PKL2"
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist

    file_name = f"mcnn_bilstm_{IMBALANCE}_{DATA_FEATURE}_Independent_{int(time.time())}.pkl"
    file_path = os.path.join(folder, file_name)

    with open(file_path, "wb") as file:
        pickle.dump(data_to_save, file)

    print(f"ROC data saved to: {os.path.abspath(file_path)}")
"----------------------------------------------------------------------------------------------------"
class DeepScan(Model):
    def __init__(self,
                 input_shape=(MAXSEQ, NUM_FEATURE),
                 window_sizes=[2, 4, 6], 
                 num_filters=64,    # Default to 64 as per your MC-LSTM description
                 lstm_units=64,     # Default to 64 as per your MC-LSTM description
                 num_hidden=1000,
                 dropout_rate=0.7):
        super(DeepScan, self).__init__()
        
        self.window_sizes = window_sizes
        self.conv1d_layers = []
        self.bilstm_layers = []

        # Define parallel branches for each window size
        for window_size in self.window_sizes:
            # 1. Conv1D Layer: Scans for local patterns (e.g., 3-residue motifs)
            self.conv1d_layers.append(
                layers.Conv1D(filters=num_filters,
                              kernel_size=window_size,
                              activation='relu', 
                              padding='valid', # Output length will be L - w + 1
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_initializer='glorot_uniform')
            )
            
            # 2. BiLSTM Layer: Processes the sequence forwards and backwards
            # return_sequences=False ensures we get a fixed-size summary vector (last hidden states)
            self.bilstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(units=lstm_units, return_sequences=False)
                )
            )

        # Fully Connected Layers for Classification
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.fc1 = layers.Dense(num_hidden,
                                activation='relu',
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer='glorot_uniform')
        
        self.fc2 = layers.Dense(NUM_CLASSES,
                                activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        # Adaptation for your data loader:
        # Your data is likely (Batch, 1, MAXSEQ, NUM_FEATURE) for the old Conv2D model.
        # We need (Batch, MAXSEQ, NUM_FEATURE) for Conv1D/LSTM.
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1) 

        branch_outputs = []
        
        # Iterate through every branch (window size)
        for i in range(len(self.window_sizes)):
            # Step 1: Convolution (capturing local motifs)
            x_conv = self.conv1d_layers[i](x)
            
            # Step 2: NO Max-Pooling (passed directly to LSTM as per description)
            
            # Step 3: BiLSTM (capturing sequential dependencies)
            # Output is a vector of size 2 * lstm_units (e.g., 128)
            x_lstm = self.bilstm_layers[i](x_conv)
            
            branch_outputs.append(x_lstm)

        # Step 4: Concatenate all branch vectors
        # e.g., if 3 windows, output shape is (Batch, 3 * 128) -> (Batch, 384)
        if len(branch_outputs) > 1:
            x = layers.Concatenate(axis=1)(branch_outputs)
        else:
            x = branch_outputs[0]

        # Step 5: Classification
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        output = self.fc2(x) 
        return output
# model fit batch funtion
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        return np.array(batch_data), np.array(batch_labels)
    
"----------------------------------------------------------------------------------------------------"

# Example usage:
x_train,y_train,x_test,y_test= load_data.MCNN_data_load(DATA_FEATURE)
print(x_train.shape)
print(x_train.dtype)
print(y_train.shape)
print(x_test.shape)
print(x_test.dtype)
print(y_test.shape)

def IMBALANCE_funct(IMBALANCE,x_train,y_train):
    if(IMBALANCE)=="None":
        return x_train,y_train
    else:
        from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
    
        # 將 x_train 的形狀重新整形為二維
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        print(x_train_2d.shape)
        print(y_train.shape)
        #print(y_train.shape)
        # 創建 SMOTE 物件
        if IMBALANCE=="SMOTE":
            imbalance = SMOTE(random_state=42)
        elif IMBALANCE=="ADASYN":
            imbalance = ADASYN(random_state=42)
        elif IMBALANCE=="RANDOM":
            imbalance = RandomOverSampler(random_state=42)
        
    
        # 使用 fit_resample 進行過採樣
        x_train_resampled, y_train_resampled = imbalance.fit_resample(x_train_2d, y_train)
    
        # 將 x_train_resampled 的形狀恢復為四維
        x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 1, MAXSEQ, NUM_FEATURE)
    
        print(x_train_resampled.shape)
        print(y_train_resampled.shape)
    
        x_train=x_train_resampled
        y_train=y_train_resampled
        
        del x_train_resampled
        del y_train_resampled
        del x_train_2d
        gc.collect()
    
        import tensorflow as tf
        y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
        return x_train,y_train

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def model_test(model, x_test, y_test):

    print(x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)
    #tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    # display.plot()
    

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    Prec=TP/(TP+FP)
    Recall=TP/(TP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}\n')
    # if MODE == 'ind':
    #     SAVEROC(fpr,tpr,AUC)
    
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC

if(MODE=="cross"):
	time_log("Start cross")
	
	kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
	results=[]
	i=1
	for train_index, test_index in kfold.split(x_train):
		print(i,"/",K_Fold,'\n')
		# 取得訓練和測試數據
		X_train, X_test = x_train[train_index], x_train[test_index]
		Y_train, Y_test = y_train[train_index], y_train[test_index]
		
		print(X_train.shape)
		print(X_test.shape)
		print(Y_train.shape)
		print(Y_test.shape)
		X_train,Y_train=IMBALANCE_funct(IMBALANCE,X_train,Y_train)
		generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)
		# 重新建模
		model = DeepScan(
		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.build(input_shape=X_train.shape)
		# 在測試數據上評估模型
		history=model.fit(
			generator,
			epochs=EPOCHS,
			callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
			verbose=1,
			shuffle=True
		)
		TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, X_test, Y_test)
		results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
		i+=1
		
		del X_train
		del X_test
		del Y_train
		del Y_test
		gc.collect()
		
	mean_results = np.mean(results, axis=0)
	print(f'TP={mean_results[0]:.4}, FP={mean_results[1]:.4}, TN={mean_results[2]:.4}, FN={mean_results[3]:.4}, Sens={mean_results[4]:.4}, Spec={mean_results[5]:.4}, Acc={mean_results[6]:.4}, MCC={mean_results[7]:.4}, AUC={mean_results[8]:.4}\n')
	write_data.append(mean_results[0])
	write_data.append(mean_results[1])
	write_data.append(mean_results[2])
	write_data.append(mean_results[3])
	write_data.append(mean_results[4])
	write_data.append(mean_results[5])
	write_data.append(mean_results[6])
	write_data.append(mean_results[7])
	write_data.append(mean_results[8])

if(MODE=="ind"):
    x_train,y_train=IMBALANCE_funct(IMBALANCE,x_train,y_train)
    print(f"After using {IMBALANCE} method, x_train {x_train.shape}, y_train {y_train.shape}")
    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
    time_log("Start Model Train")
    # model = DeepScan(
    # 	num_filters=NUM_FILTER,
    # 	num_hidden=NUM_HIDDEN,
    # 	window_sizes=WINDOW_SIZES)
    # Or to use your command line arguments:
    model = DeepScan(
        num_filters=NUM_FILTER,
        lstm_units=LSTM_UNITS,          # You might want to add an argument for this
        num_hidden=NUM_HIDDEN,
        window_sizes=WINDOW_SIZES
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()
    model.fit(
    	generator,
    	epochs=EPOCHS,
    	shuffle=True,
    )
    time_log("End Model Train")
    time_log("Start Model Test")
    TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC = model_test(model, x_test, y_test)

    # model.save_weights(f"./saved_weights/Model_{str(DATA_FEATURE)}_{IMBALANCE}.h5")
    # del model
    # tf.keras.backend.clear_session()
    # gc.collect()
    
    write_data.append(TP)
    write_data.append(FP)
    write_data.append(TN)
    write_data.append(FN)
    write_data.append(Sens)
    write_data.append(Spec)
    write_data.append(Acc)
    write_data.append(MCC)
    write_data.append(AUC)
       
    time_log("End Model Test")

    # save_csv(write_data,a)

def save_csv(write_data,a):
    import csv
    b=datetime.datetime.now()
    write_data.append(b-a)
    open_csv=open(csv_file_path,"a")
    write_csv=csv.writer(open_csv)
    write_csv.writerow(write_data)

save_csv(write_data,a)