import tensorflow as tf
import tensorflow.keras as keras

def NN_model(InputShape, h_layers, lr, drops, dropout):
	In = keras.layers.Input(shape=[InputShape,])
	x = In
	for i, unit in enumerate(h_layers[:]):
		x = keras.layers.Dense(unit, activation="linear",kernel_initializer='glorot_uniform')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Dropout(drops[i])(x, training=dropout)

	predictions = keras.layers.Dense(3, activation='softmax', kernel_initializer='glorot_uniform')(x)

	model = keras.models.Model(inputs=In, outputs=predictions)

	model_optimizer = keras.optimizers.Adam(learning_rate=lr)

	model.compile(
			loss = 'categorical_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy'])

	return model

