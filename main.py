#!/usr/bin/env python3
#
# Trains an FCN on images and masks, uses pre-trained VGG network as a starting point
#
#

import os.path
import tensorflow as tf
import helper
import warnings
import argparse
from distutils.version import LooseVersion
import project_tests as tests


parser = argparse.ArgumentParser(description='Train FCN')
parser.add_argument('--data', type=str,
                    help='Location of data directory to import')
parser.add_argument('--vgg', type=str,
                    help='Location of vgg model')
parser.add_argument('--output', type=str,
                    help='Output directory for trained model')
parser.add_argument('--run', type=str,
                    help='Output directory for test images')
args = parser.parse_args()


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load frozen model into session
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get graph from session
    graph = tf.get_default_graph()

    # get tensors
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 convolution out 7
    conv_11_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # deconvolution x2
    upsample_1 = tf.layers.conv2d_transpose(conv_11_7, num_classes, 4, 2, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 convolution out 4
    conv_11_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # skip layer
    skip_1 = tf.add(conv_11_4, upsample_1)

    # deconvolution x2
    upsample_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 convolution out 3
    conv_11_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # skip layer 2
    skip_2 = tf.add(conv_11_3, upsample_2)

    # final deconvolution x8
    output = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # reshape (flatten) image output
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # reshape labels
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # softmax cross entropy loss with logits
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print("Start training for {} epochs".format(epochs))

    # TODO: Implement function
    for epoch_i in range(epochs):
        print("Epoch #{}".format(epoch_i))

        for image, label in get_batches_fn(batch_size):
            # image shape: (batch_size, rows, cols, depth)
            # labels shape: (batch_size, rows, cols, num_classes)

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={
                                   keep_prob: 0.5,
                                   input_image: image,
                                   correct_label: label,
                                   learning_rate: 0.001
                               })

            print("Loss: {}".format(loss))


tests.test_train_nn(train_nn)


def run(data_arg, vgg_arg, output_arg, run_arg):
    num_classes = 2
    image_shape = (256, 256)  # KITTI dataset uses 160x576 images, puzzle dataset uses 256x256
    data_dir = data_arg
    vgg_path = vgg_arg
    runs_dir = run_arg
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(vgg_path)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # own parameters
    n_epochs = 1
    batch_size = 4

    print("==========\nSession start.\n==========")
    with tf.Session() as sess:
        # Path to vgg model
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'training'), image_shape, puzzle=True)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        print("Building NN...")
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # create placeholders
        correct_label = tf.placeholder(tf.int32, (None, None, None, None), name="correct_label")
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        print("Training NN...")
        train_nn(sess, n_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        print("Saving examples")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        print("saving model...")

        # Save the variables to disk.
        saver = tf.train.Saver()
        save_path = saver.save(sess, output_arg)
        print("Model saved in path: %s" % save_path)

        # OPTIONAL: Apply the trained model to a video
        print("==========\nSession end.\n==========")


if __name__ == '__main__':
    # training and testing data directory
    data_arg = args.data
    if data_arg == "":
        data_arg = './puzzle_data'
    tests.test_data_source(data_arg)

    # vgg location directory
    vgg_arg = args.vgg
    if vgg_arg == "":
        vgg_arg = './data/vgg'

    # trained model output file location
    # create directory if not exists
    output_arg = args.output
    if output_arg == "":
        output_arg = "./model/puzzle_model.ckpt"
    output_dirname = os.path.dirname(output_arg)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # output of run over test images
    # create directory if not exists
    run_arg = args.run
    if run_arg == "":
        run_arg = './runs'
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    run(data_arg, vgg_arg, output_arg, run_arg)
