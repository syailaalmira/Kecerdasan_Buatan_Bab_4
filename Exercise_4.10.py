import argparse
import time
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image


parser = argparse.ArgumentParser(
	description='Compare VGG16 and VGG19 image classification performance.'
)
parser.add_argument(
	'image_path',
	nargs='?',
	default='Elephant.jpg',
	help='Path to the image (default: Elephant.jpg)',
)
parser.add_argument(
	'--runs',
	type=int,
	default=3,
	help='Number of inference runs used for timing (default: 3)',
)
args = parser.parse_args()

if args.runs < 1:
	parser.error('--runs must be at least 1')

img_path = Path(args.image_path)
if not img_path.is_absolute():
	img_path = Path(__file__).resolve().parent / img_path
if not img_path.is_file():
	parser.error(f'Image file not found: {img_path}')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


def classify_and_measure(model):
	model.predict(x, verbose=0)
	start = time.perf_counter()
	for _ in range(args.runs):
		predictions = model.predict(x, verbose=0)
	elapsed = (time.perf_counter() - start) / args.runs
	return predictions, elapsed


models = {
	'VGG16': VGG16(weights='imagenet'),
	'VGG19': VGG19(weights='imagenet'),
}

for model_name, model in models.items():
	predictions, average_time = classify_and_measure(model)
	results = decode_predictions(predictions, top=5)[0]

	print(f'\n{model_name}')
	print(f'Average inference time: {average_time:.4f} seconds')
	for _, label, confidence in results:
		print(f'  {label}: {confidence:.2%}')
