from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential


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
		model.add(Conv2D(64, (3, 3), activation="relu"))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dense(2, activation="softmax"))
	return model


baseline_model = build_model()
modified_model = build_model(add_convolution_block=True)

baseline_parameters = baseline_model.count_params()
modified_parameters = modified_model.count_params()
parameter_difference = modified_parameters - baseline_parameters
percentage_change = parameter_difference / baseline_parameters * 100

print(f"Example 4.7 parameters: {baseline_parameters:,}")
print(f"With added Conv2D block: {modified_parameters:,}")
print(f"Change: {parameter_difference:+,} ({percentage_change:+.2f}%)")
