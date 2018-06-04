from __future__ import print_function

from keras import backend as K
from keras.layers import Conv2D, Concatenate, Dense, Flatten, BatchNormalization, Activation, AveragePooling2D
from keras.models import Model, Input, Sequential
from keras.utils import plot_model


def _convbnrelu(x, nb_filters, stride, kernel_size, name):
    """
    Convolution block of the first layer

    :param x: input tensor
    :param nb_filters: integer or tuple, number of filters
    :param stride: integer or tuple, stride of convolution
    :param kernel_size: integer or tuple, filter's kernel size
    :param name: string, block label

    :return: output tensor of a block
    """

    shape = K.int_shape(x)[1:]
    return Sequential(layers=
    [
        Conv2D(filters=nb_filters, strides=stride, kernel_size=kernel_size, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv2d",
               input_shape=shape),
        BatchNormalization(name=name + "_batch_norm"),
        Activation(activation='relu', name=name + '_relu')
    ],
        name=name + "_convbnrelu")(x)


def _bottleneck(x, growth_rate, stride, name):
    """
    DenseNet-like block for subsequent layers
    :param x: input tensor
    :param growth_rate: integer, number of output channels
    :param stride: integer, stride of 3x3 convolution
    :param name: string, block label

    :return: output tensor of a block
    """

    shape = K.int_shape(x)[1:]
    return Sequential(layers=
    [
        Conv2D(filters=4 * growth_rate, strides=1, kernel_size=1, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv2d_1x1",
               input_shape=shape),
        BatchNormalization(name=name + "_batch_norm_1"),
        Activation(activation='relu', name=name + "_relu_1"),
        Conv2D(filters=growth_rate, strides=stride, kernel_size=3,
               padding='same', kernel_initializer='he_normal', use_bias=False,
               name=name + "_conv2d_3x3"),
        BatchNormalization(name=name + "_batch_norm_2"),
        Activation(activation='relu', name=name + "_relu_2")
    ],
        name=name + "_bottleneck")(x)


def basic_block(x, l_growth_rate=None, scale=3, name="basic_block"):
    """
    Basic building block of MSDNet

    :param x: Input tensor or list of tensors
    :param l_growth_rate: list, numbers of output channels for each scale
    :param scale: Number of different scales features
    :param name:
    :return: list of different scales features listed from fine-grained to coarse
    """
    output_features = []

    try:
        is_tensor = K.is_keras_tensor(x)
        # check if not a tensor
        # if keras/tf class raise error instead of assign False
        if not is_tensor:
            raise TypeError("Tensor or list [] expected")

    except ValueError:
        # if not keras/tf class set False
        is_tensor = False

    if is_tensor:

        for i in range(scale):
            mult = 2 ** i
            x = _convbnrelu(x, nb_filters=32 * mult, stride=min(2, mult), kernel_size=3, name=name + "_" + str(i))
            output_features.append(x)

    else:

        assert len(l_growth_rate) == scale, "Must be equal: len(l_growth_rate)={0} scale={1}".format(len(l_growth_rate),
                                                                                                     scale)

        for i in range(scale):
            if i == 0:
                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv2d_" + str(i))

                conc = Concatenate(axis=3, name=name + "_concat_post_" + str(i))([conv, x[i]])
            else:
                strided_conv = _bottleneck(x[i - 1], growth_rate=l_growth_rate[i], stride=2,
                                           name=name + "_strided_conv2d_" + str(i))

                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv2d_" + str(i))

                conc = Concatenate(axis=3, name=name + "_concat_pre_" + str(i))([strided_conv, conv, x[i]])

            output_features.append(conc)

    return output_features


def transition_block(x, reduction, name):
    """
    Transition block for network reduction
    :param x: list, set of tensors
    :param reduction: float, fraction of output channels with respect to number of input channels
    :param name: string, block label

    :return: list of tensors
    """
    output_features = []
    for i, item in enumerate(x):
        conv = _convbnrelu(item, nb_filters=int(reduction * K.int_shape(item)[3]), stride=1, kernel_size=1,
                           name=name + "_transition_block_" + str(i))
        output_features.append(conv)

    return output_features


def classifier_block(x, nb_filters, nb_classes, activation, name):
    """
    Classifier block
    :param x: input tensor
    :param nb_filters: integer, number of filters
    :param nb_classes: integer, number of classes
    :param activation: string, activation function
    :param name: string, block label

    :return: block tensor
    """
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_1")
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_2")
    x = AveragePooling2D(pool_size=2, strides=2, padding='same', name=name + '_avg_pool2d')(x)
    x = Flatten(name=name + "_flatten")(x)
    out = Dense(units=nb_classes, activation=activation, name=name + "_dense")(x)
    return out


def build(input_size=(256, 256, 3), nb_classes=100, scale=3, depth=5, l_growth_rate=(6, 12, 24),
          transition_block_location=(12, 20), classifier_ch_nb=128, classifier_location=(5, )):
    """
    Function that builds MSDNet

    :param input_size: tuple of integers, 3x1, size of input image
    :param nb_classes: integer, number of classes
    :param scale: integer, number of network's scales
    :param depth: integer, network depth
    :param l_growth_rate: tuple of integers, scale x 1, growth rate of each scale
    :param transition_block_location: tuple of integer, array of block's numbers to place transition block after
    :param classifier_ch_nb: integer, output channel of conv blocks in classifier, if None than the same number as in
                                      an input tensor
    :param classifier_location: tuple of integers, array of block's numbers to place classifier after

    :return: MSDNet
    """

    inp = Input(shape=input_size)
    out = []

    for i in range(depth):

        if i == 0:
            x = basic_block(inp, l_growth_rate=[],
                            scale=scale, name="basic_block_" + str(i + 1))
        elif i in transition_block_location:
            x = transition_block(x, reduction=0.5, name="transition_block_" + str(i + 1))

            x = basic_block(x, l_growth_rate=l_growth_rate,
                            scale=scale, name="basic_block_" + str(i + 1))
            scale -= 1
            l_growth_rate = l_growth_rate[1:]
            x = x[1:]
        else:
            x = basic_block(x, l_growth_rate=l_growth_rate,
                            scale=scale, name="basic_block_" + str(i + 1))

        if i+1 in classifier_location:
            cls_ch = K.int_shape(x[-1])[3] if classifier_ch_nb is None else classifier_ch_nb
            out.append(classifier_block(x[-1], nb_filters=cls_ch, nb_classes=nb_classes, activation='sigmoid',
                                        name='classifier_' + str(i + 1)))

    return Model(inputs=inp, outputs=out)


def MSDNet_cifar(input_shape, nb_classes):
    transition_location = (12, 18)
    classifier_location = [2*(i+1) for i in range(1, 12)]

    return build(input_size=input_shape, nb_classes=nb_classes, scale=3,
                 depth=24, l_growth_rate=(6, 12, 24), transition_block_location=transition_location,
                 classifier_ch_nb=128, classifier_location=classifier_location)

# plot_model(model=model, to_file='model.png', show_shapes=True)
