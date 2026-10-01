import time

from keras import utils
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(
	X_data, y_data, test_size=0.3, random_state=7
)


def train_and_measure(neurons):
	utils.set_random_seed(7)
	model = Sequential(
		[
			Dense(neurons, activation="relu", input_shape=[X_train.shape[1]]),
			Dense(neurons, activation="relu"),
			Dense(1),
		]
	)
	model.compile(
		optimizer="adam",
		loss="mean_squared_error",
		metrics=["accuracy"],
	)

	start_time = time.perf_counter()
	model.fit(
		X_train,
		y_train,
		batch_size=50,
		validation_split=0.2,
		epochs=100,
		verbose=0,
	)
	training_time = time.perf_counter() - start_time
	_, accuracy = model.evaluate(X_test, y_test, verbose=0)
	return training_time, accuracy


baseline_time, baseline_accuracy = train_and_measure(10)
modified_time, modified_accuracy = train_and_measure(5)

print("Dense neurons | Training time (s) | Test accuracy")
print(f"10            | {baseline_time:17.2f} | {baseline_accuracy:.2%}")
print(f"5             | {modified_time:17.2f} | {modified_accuracy:.2%}")
