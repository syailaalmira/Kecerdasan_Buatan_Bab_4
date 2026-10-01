import time

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


model_specs = [
	('VGG16', vgg16.VGG16, vgg16.preprocess_input, 224),
	('ResNet50', resnet50.ResNet50, resnet50.preprocess_input, 224),
	('MobileNetV2', mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input, 224),
	('DenseNet201', densenet.DenseNet201, densenet.preprocess_input, 224),
	('InceptionV3', inception_v3.InceptionV3, inception_v3.preprocess_input, 299),
]

models = []
for model_name, classifier, preprocess, image_size in model_specs:
	model = classifier(
		input_shape=(image_size, image_size, 3),
		weights='imagenet',
		classes=1000,
	)
	models.append((model_name, model, preprocess, image_size))

camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Could not open the camera.')

total_times = {model_name: 0.0 for model_name, _, _, _ in models}
frame_count = 0

try:
	while camera.isOpened():
		ret, cam_frame = camera.read()
		if not ret:
			break

		results = []
		for model_name, model, preprocess, image_size in models:
			frame = cv2.resize(cam_frame, (image_size, image_size))
			image = np.asarray(frame)
			image = np.expand_dims(image, 0)
			image = preprocess(image)

			start = time.perf_counter()
			predictions = model.predict(image, verbose=0)
			elapsed = time.perf_counter() - start
			total_times[model_name] += elapsed

			label = decode_predictions(predictions, top=1)[0][0]
			results.append((model_name, label[1], label[2], elapsed))

		frame_count += 1
		for line_number, (model_name, label, confidence, elapsed) in enumerate(
			results
		):
			text = f'{model_name}: {label} {confidence:.1%} ({elapsed:.3f}s)'
			cv2.putText(
				cam_frame,
				text,
				(10, 25 + line_number * 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(0, 255, 0),
				2,
			)

		cv2.imshow('Image Classifier Comparison', cam_frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()

if frame_count:
	print('\nAverage inference time:')
	for model_name, total_time in total_times.items():
		print(f'{model_name}: {total_time / frame_count:.4f} seconds')
