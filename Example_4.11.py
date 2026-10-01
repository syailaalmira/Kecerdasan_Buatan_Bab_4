import argparse
from pathlib import Path

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

parser = argparse.ArgumentParser(description='Classify an image with VGG16.')
parser.add_argument(
	'image_path',
	nargs='?',
	default='Elephant.jpg',
	help='Path to the image (default: Elephant.jpg)',
)
args = parser.parse_args()

model = VGG16(weights='imagenet')
img_path = Path(args.image_path)
if not img_path.is_absolute():
	img_path = Path(__file__).resolve().parent / img_path
if not img_path.is_file():
	parser.error(f'Image file not found: {img_path}')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# prediction
predictions = model.predict(x)
results = decode_predictions(predictions)
print(results)