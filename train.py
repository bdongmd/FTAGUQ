import numpy as np
import os
import h5py
import model
import tensorflow as tf

from keras.models import load_model
from keras.utils import np_utils
import keras
import argparse

parser = argparse.ArgumentParser(
    description='Options for making the training files'
)
parser.add_argument('-i', '--input_file', type=str,
		default="input/MC16d_hybrid-training_sample-NN.h5",
		help='Set name of preprocessed input training file')
parser.add_argument('-v', '--validate_file', type=str,
		default="input/MC16d_ttbar-test-validation_sample-NN.h5",
		help='Set name of preprocessed validation file')
parser.add_argument('-b', '--batch_size', type=int,
		default=3000,
		help='Set batch size')
parser.add_argument('-e', '--epoch', type=int,
		default=300,
		help='Set epoch')

args = parser.parse_args()
h5f_train = h5py.File(args.input_file, 'r')
h5f_test = h5py.File(args.validate_file, 'r')

X_train = h5f_train['X_train'][:]
Y_train = h5f_train['Y_train'][:]
X_test = h5f_test['X_test'][:]
Y_test = h5f_test['Y_test'][:]

InputShape=44
h_layers=[72, 57, 60, 48, 36, 24, 12, 6]
lr = 0.005
drops=[0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
dropout=True
batch_size = args.batch_size

DL1model, batch_size = model.NN_model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=dropout, batch_size=batch_size)
DL1model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='models/model_{epoch}',
        save_freq='epoch'
    )
]

params = {
    'batch_size': batch_size,
    'X': X_train,
    'Y': Y_train,
}

history = DL1model.fit(X_train, Y_train,
                    batch_size = args.batch_size,
                    epochs = args.epoch,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks
                    )

DL1model.save("models/DL1_hybrid_2M_b{}_e{}.h5".format(args.batch_size, args.epoch))
