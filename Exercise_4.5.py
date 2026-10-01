from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential


def build_alexnet(dense_units):
	model = Sequential(
		[
			Input(shape=(224, 224, 3)),
			Conv2D(96, kernel_size=(11, 11), activation="relu", strides=(4, 4)),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(256, kernel_size=(11, 11), activation="relu"),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(384, kernel_size=(3, 3), activation="relu"),
			Conv2D(384, kernel_size=(3, 3), activation="relu"),
			Conv2D(256, kernel_size=(3, 3), activation="relu"),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Flatten(),
			Dense(dense_units[0], activation="relu"),
			Dropout(0.4),
			Dense(dense_units[1], activation="relu"),
			Dropout(0.4),
			Dense(dense_units[2], activation="relu"),
			Dropout(0.4),
			Dense(17, activation="softmax"),
		]
	)
	return model


example_model = build_alexnet((4096, 4096, 1000))
modified_model = build_alexnet((2048, 2048, 512))

example_parameters = example_model.count_params()
modified_parameters = modified_model.count_params()
parameter_difference = modified_parameters - example_parameters
percentage_change = parameter_difference / example_parameters * 100

print(f"Example dense units (4096, 4096, 1000): {example_parameters:,} parameters")
print(f"Modified dense units (2048, 2048, 512): {modified_parameters:,} parameters")
print(f"Parameter change: {parameter_difference:+,} ({percentage_change:+.2f}%)")
