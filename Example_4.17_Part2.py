from keras.applications.vgg19 import VGG19

model = VGG19()

n=0
for layer in model.layers:
    print(n,layer.name)
    n += 1