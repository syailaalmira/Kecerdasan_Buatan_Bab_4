from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from keras.models import Sequential


def build_lenet(first_filters, second_filters):
	model = Sequential(
		[
			Input(shape=(32, 32, 1)),
			Conv2D(first_filters, kernel_size=(3, 3), activation="relu"),
			AveragePooling2D(pool_size=(2, 2)),
			Conv2D(second_filters, kernel_size=(3, 3), activation="relu"),
			AveragePooling2D(pool_size=(2, 2)),
			Flatten(),
			Dense(120, activation="relu"),
			Dense(84, activation="relu"),
			Dense(10, activation="softmax"),
		]
	)
	return model


example_model = build_lenet(6, 16)
modified_model = build_lenet(8, 24)

example_parameters = example_model.count_params()
modified_parameters = modified_model.count_params()
parameter_difference = modified_parameters - example_parameters
percentage_increase = parameter_difference / example_parameters * 100

print("Example 4.4 filters (6, 16):")
example_model.summary()
print("Modified filters (8, 24):")
modified_model.summary()
print(f"Example total parameters: {example_parameters:,}")
print(f"Modified total parameters: {modified_parameters:,}")
print(f"Additional parameters: {parameter_difference:,} ({percentage_increase:.2f}%)")
