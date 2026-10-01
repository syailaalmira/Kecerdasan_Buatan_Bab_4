import time

import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
X_test = X_test.reshape((-1, 28, 28, 1)).astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def build_model(add_convolution_block=False):
	model = Sequential(
		[
			Input(shape=(28, 28, 1)),
			Conv2D(32, (5, 5), activation="relu"),
			MaxPooling2D(),
			Dropout(0.2),
		]
	)

	if add_convolution_block:
		model.add(Conv2D(32, (3, 3), activation="relu"))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dense(num_classes, activation="softmax"))
	model.compile(
		loss="categorical_crossentropy",
		optimizer="adam",
		metrics=["accuracy"],
	)
	return model


configurations = [
	("Baseline", False),
	("Extra Conv2D + Pool + Dropout", True),
]

print("Model                            Training time (s)   Test accuracy")
for name, add_convolution_block in configurations:
	keras.backend.clear_session()
	keras.utils.set_random_seed(7)
	model = build_model(add_convolution_block)

	start_time = time.perf_counter()
	model.fit(
		X_train,
		y_train,
		validation_data=(X_test, y_test),
		epochs=10,
		batch_size=200,
		verbose=0,
	)
	training_time = time.perf_counter() - start_time
	_, accuracy = model.evaluate(X_test, y_test, verbose=0)
	print(f"{name:<32} {training_time:>10.2f}          {accuracy:.2%}")