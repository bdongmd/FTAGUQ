import tensorflow as tf
import tensorflow.keras as keras

import sys
sys.path.append('DL1_model/')
from models.maxout_layers import Maxout1D

def private_DL1Model(InputShape, h_layers, lr=0.01, drops=None, dropout=True, batch_size=3000):
	In = keras.layers.Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = keras.layers.Dense(h, activation="relu",kernel_initializer='glorot_uniform')(x)
		x = keras.layers.Dropout(drops[i])(x, training=dropout)
		x = keras.layers.BatchNormalization()(x)

	predictions = keras.layers.Dense(3, activation='softmax', kernel_initializer='glorot_uniform')(x)

	model = keras.models.Model(inputs=In, outputs=predictions)

	model_optimizer = keras.optimizers.Adam(lr=lr)

	model.compile(
			loss = 'categorical_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy'])

	return model, batch_size
