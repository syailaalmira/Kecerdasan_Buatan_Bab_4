import time

import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from tensorflow.keras.preprocessing.image import img_to_array


image_size = 224
models = {
	'VGG16': (VGG16(weights='imagenet'), preprocess_vgg16),
	'VGG19': (VGG19(weights='imagenet'), preprocess_vgg19),
}

camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Could not open the camera.')

try:
	while camera.isOpened():
		ok, cam_frame = camera.read()
		if not ok:
			break

		frame = cv2.resize(cam_frame, (image_size, image_size))
		numpy_image = img_to_array(frame)
		image_batch = np.expand_dims(numpy_image, axis=0)

		results = {}
		for model_name, (model, preprocess) in models.items():
			processed_image = preprocess(image_batch.copy())
			start = time.perf_counter()
			predictions = model.predict(processed_image, verbose=0)
			elapsed = time.perf_counter() - start
			result = decode_predictions(predictions, top=1)[0][0]
			results[model_name] = (result[1], result[2], elapsed)

		for line_number, (model_name, (label, confidence, elapsed)) in enumerate(
			results.items()
		):
			text = (
				f'{model_name}: {label} {confidence:.1%} '
				f'({elapsed:.3f}s)'
			)
			cv2.putText(
				cam_frame,
				text,
				(10, 30 + line_number * 35),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 0, 0) if model_name == 'VGG16' else (0, 255, 0),
				2,
			)

		cv2.imshow('VGG16 vs VGG19 webcam classification', cam_frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()
