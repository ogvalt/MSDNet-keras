import os
import numpy as np
import keras
from keras import backend as K
from keras.datasets.cifar import load_batch
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from msdnet import MSDNet_cifar
from LoggingCallback import LoggingCallback

def load_data(path=os.path.join(".", "cifar-100-python"), label_mode='fine'):
    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def step_decay(init_learn_rate):
    def schedule(epoch):
        if 150 < epoch <= 225:
            return init_learn_rate / 10
        elif epoch > 225:
            return init_learn_rate / 100
        else:
            return init_learn_rate

    return LearningRateScheduler(schedule)


def logger():
    def print_fn(msg):
        with open(os.path.join(".", "report.txt"), "a") as f:
            f.write(msg + "\n")
    return LoggingCallback(print_fn)


# Training parameters
batch_size = 64  # orig paper trained all networks with batch_size=128
epochs = 300
num_classes = 100
learning_rate = 1e-4

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'msdnet_cifar100.h5py'

# Load the CIFAR100 data.
(x_train, y_train), (x_test, y_test) = load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# subtract pixel mean
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = MSDNet_cifar(input_shape=input_shape, nb_classes=num_classes)
model.summary()

# opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-4, nesterov=True)
opt = RMSprop(lr=learning_rate, decay=1e-4)

# Let's train the model using RMSprop
model.compile(loss={i.name: 'categorical_crossentropy' for i in model.output_layers},
              optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125,
                             horizontal_flip=True, vertical_flip=False)

tensorboard = TensorBoard()
logging_clb = logger()

def multiout_iterator(model, gen, x, y, batch_size):
    while True:
        batch_x, batch_y = next(gen.flow(x=x, y=y, batch_size=batch_size))
        yield (batch_x, {i.name: batch_y for i in model.output_layers})

model.fit_generator(multiout_iterator(model, datagen, x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]/batch_size,
                    epochs=epochs,
                    validation_data=(x_test, {i.name: y_test for i in model.output_layers}),
                    shuffle=True, callbacks=[tensorboard, step_decay(learning_rate), logging_clb],
                    verbose=1
                    )

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
