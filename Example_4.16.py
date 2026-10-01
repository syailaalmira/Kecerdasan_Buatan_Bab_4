# Example 4.16 AutoencoderKeras.py
# Modified from:

# https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

#
# Load the MNIST handwritten digital images =============================
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)
# Build Autoencoder =====================================================
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
def build_autoencoder(img_shape, code_size):
	# The encoder
	encoder = Sequential()
	encoder.add(InputLayer(shape=img_shape))
	encoder.add(Flatten())
	encoder.add(Dense(code_size))
	# The decoder
	decoder = Sequential()
	decoder.add(InputLayer(shape=(code_size,)))
	decoder.add(Dense(int(np.prod(img_shape))))
	decoder.add(Reshape(img_shape))
	return encoder, decoder
IMG_SHAPE = x_train[0].shape
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)
inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
print(autoencoder.summary())
# Train Autoencoder =====================================================
history = autoencoder.fit(x=x_train, y=x_train, epochs=20,
validation_data=(x_test, x_test))
# Plot the training results =============================================
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Plot the original, encoded and decoded images =========================
import matplotlib.pyplot as plt
def show_image(x):
	plt.imshow(np.clip(x + 0.5, 0, 1))


def visualize(img,encoder,decoder):
	"""Draw the original, encoded, and reconstructed images."""
	# img[None] has the batch dimension expected by the model.
	code = encoder.predict(img[None], verbose=0)[0]
	reco = decoder.predict(code[None], verbose=0)[0]
	plt.subplot(1,3,1)
	plt.title("Original")
	show_image(img)
	plt.subplot(1,3,2)
	plt.title("Code")
	plt.imshow(code.reshape([code.shape[-1]//2,-1]))
	plt.subplot(1,3,3)
	plt.title("Reconstructed")
	show_image(reco)
	plt.show()


for i in range(5):
	img = x_test[i]
	visualize(img,encoder,decoder)