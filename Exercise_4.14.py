import cv2
import numpy as np
from tensorflow.keras.applications import (
	densenet,
	inception_v3,
	mobilenet_v2,
	resnet50,
	vgg16,
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# Change this value to select a different classifier.
selected_model = 'vgg16'

if selected_model == 'vgg16':
	model = vgg16.VGG16(weights='imagenet')
	preprocess_input = vgg16.preprocess_input
	image_size = 224
elif selected_model == 'resnet50':
	model = resnet50.ResNet50(weights='imagenet')
	preprocess_input = resnet50.preprocess_input
	image_size = 224
elif selected_model == 'mobilenetv2':
	model = mobilenet_v2.MobileNetV2(weights='imagenet')
	preprocess_input = mobilenet_v2.preprocess_input
	image_size = 224
elif selected_model == 'densenet201':
	model = densenet.DenseNet201(weights='imagenet')
	preprocess_input = densenet.preprocess_input
	image_size = 224
elif selected_model == 'inceptionv3':
	model = inception_v3.InceptionV3(weights='imagenet')
	preprocess_input = inception_v3.preprocess_input
	image_size = 299
else:
	raise ValueError(f'Unknown model selection: {selected_model}')

model.summary()
camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Could not open the camera.')

try:
	while camera.isOpened():
		ret, cam_frame = camera.read()
		if not ret:
			break

		frame = cv2.resize(cam_frame, (image_size, image_size))
		image = np.asarray(frame)
		image = np.expand_dims(image, 0)
		image = preprocess_input(image)
		predictions = model.predict(image, verbose=0)
		label = decode_predictions(predictions, top=1)[0][0]

		cv2.putText(
			cam_frame,
			f'{selected_model}: {label[1]}, {label[2]:.1%}',
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.9,
			(0, 255, 0),
			2,
		)
		cv2.imshow('Classification', cam_frame)
		if cv2.waitKey(30) & 0xFF == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()
