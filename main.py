import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import glob

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Global variables for paths
data_dir = './data'
runs_dir = './runs'
training_dir = data_dir + '/data_road/training'
training_size = len(glob.glob(training_dir + '/calib/*.*'))
vgg_dir = data_dir + '/vgg'
    

# Training constants
# Road and not Road
num_classes_ = 2

# Image shape (height, width) -> (rows,columns)
img_shape_ = (160, 576)

# Training epochs. Tuned empirically
epochs_ = 25

# Can only use 1 or max 2 with VRAM at 4 GB
batch_size_ = 2

# Learning rate is kept small because batch size is pretty minimal
learning_rate_ = 0.0001

# Dropout tuned empirically
dropout_ = 0.80

# Place_holders (_ph)
label_ph = tf.placeholder(tf.float32, [None, 
                                       img_shape_[0],
                                       img_shape_[1], 
                                       num_classes_])

# Learning Rate
learning_rate_ph = tf.placeholder(tf.float32)

# keep_prob
keep_prob = tf.placeholder(tf.float32)

# Initialize training losses to null
all_training_losses = [] 
    
    
def load_vgg(sess, vgg_path):
    
    """
    Load Pretrained VGG Model
    @param sess:  Tf Session
    @param vgg_dir: Directory containing vgg "variables/" and "saved_model.pb"
    return: VGG Tensor Tuple(image_input, keep_prob, layer3, layer4, layer7)
    """
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    # Load Model with Weights from vgg directory
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Setup tensors to get from graph ( vgg after loading)
    graph = tf.get_default_graph()

    # get image input
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)

    # get keep probability
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    # Get layer outputs
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # return as 5D list
    return image_input, keep_prob, layer_3, layer_4, layer_7

tests.test_load_vgg(load_vgg, tf)

def conv_1x1(layer, layer_name):
  """ convolve layer by (1x1) to preserve spatial information """
  return tf.layers.conv2d(inputs = layer,
                          filters =  num_classes_,
                          kernel_size = (1, 1),
                          strides = (1, 1),
                          name = layer_name)

def deconvolve(layer, k, s, layer_name):
  """ Transpose Convolve/ deconvolve a layer with arguments as params """
  return tf.layers.conv2d_transpose(inputs = layer,
                                    filters = num_classes_,
                                    kernel_size = (k, k),
                                    strides = (s, s),
                                    padding = 'same',
                                    name = layer_name)


def layers(vgg_layer_3_out, vgg_layer_4_out, vgg_layer_7_out, num_classes = num_classes_):
    """
    # Create layers for the FCN.
    vgg_layer_n_out: TF Tensor for VGG Layer n output
    num_classes: Number of classes to classify
    return: The Tensor for the last layer of output
    """
    
    # Apply a 1x1 convolution to all argument layers
    layer_3x = conv_1x1(layer = vgg_layer_3_out, layer_name = "layer3conv1x1")
    layer_4x = conv_1x1(layer = vgg_layer_4_out, layer_name = "layer4conv1x1")
    layer_7x = conv_1x1(layer = vgg_layer_7_out, layer_name = "layer7conv1x1")

    # Add decoder layers to the network with skip connections
    # Deconvolve
    decoder_layer_1 = deconvolve(layer = layer_7x, k = 4, s = 2, layer_name = "decoderlayer1")
    
    # Sum (skip connection)
    decoder_layer_2 = tf.add(decoder_layer_1, layer_4x, name = "decoderlayer2")
    
    # Deconvolve
    decoder_layer_3 = deconvolve(layer = decoder_layer_2, k = 4, s = 2, layer_name = "decoderlayer3")

    # Sum (skip connection)
    decoder_layer_4 = tf.add(decoder_layer_3, layer_3x, name = "decoderlayer4")
    
    # Deconvolve
    decoderlayer_output = deconvolve(layer = decoder_layer_4, k = 16, s = 8, layer_name = "decoderlayer_output")

    return decoderlayer_output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes = num_classes_):
    """
    TF loss and optimizer operations.
    nn_last_layer: last layer tensor
    correct_label: label image placeholder
    learning_rate: learning rate placeholder
    num_classes: Number of classes to classify
    return: logits, train_op, cross_entropy_loss as python list
    """
    # Flatten 4D tensors to 2D
    # (pixel,class)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    class_labels = tf.reshape(correct_label, (-1, num_classes))

    # The cross_entropy_loss is the cost heuristic
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                            labels = class_labels)
    # use the reduce mean method
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Use the standard Adam optimizer to minimize loss
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    # return logits, train_op, cross_entropy_loss as python list
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train the neural network and provide debug prints during training
    Arguments: 
    sess: TF Session
    epochs: Number of epochs
    batch_size: Batch size
    get_batches_fn: Function to get batches of training data
    train_op: training operation
    cross_entropy_loss: Loss Tensor
    input_image: TF Placeholder for input images
    correct_label: TF Placeholder for label images
    keep_prob: TF Placeholder for dropout keep probability
    learning_rate: TF Placeholder for learning rate
    """
    # For all epochs
    for epoch in range(epochs):
        #initialize losses and counter
        losses, i = [], 0
        
        # For all images in the batch
        for images, labels in get_batches_fn(batch_size_):
            
            # increment batch counter by 1
            i += 1
            
            # Create the feed by assigining values to placeholders
            feed = {input_image: images,
                    correct_label: labels,
                    keep_prob: dropout_,
                    learning_rate: learning_rate_ }

            # Run the training op with the created feed
            _, partial_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)

            # display output
            print("- - - - - >Iteration: ", i, "----->Partial loss:", partial_loss)
            
            # Add to list of losses
            losses.append(partial_loss)

        # After each batch compute net average loss
        training_loss = sum(losses) / len(losses)
        
        # Add to list of global training losses
        all_training_losses.append(training_loss)

        # Print Training loss at end of each Epoch
        print("***************")
        print("Epoch: ", epoch + 1, " of ", epochs_, "->Training loss: ", training_loss)
        print("***************")


tests.test_train_nn(train_nn)


def run():
    
    print("Training data size", training_size)
    
    # download vgg model if it doesnt exist
    helper.maybe_download_pretrained_vgg(data_dir)
    
    # use the get batches function from the helper.py provided
    get_batches_fn = helper.gen_batch_function(training_dir, img_shape_)
    
    # Using the default session
    with tf.Session() as session:
        
        # Returns the input dropout and output layers from vgg
        image_input, keep_prob, layer_3, layer_4, layer_7 = load_vgg(session, vgg_dir)

        # Create the layers and get the output
        model_output = layers(layer_3, layer_4, layer_7, num_classes_)

        # Get the logits, training op and the loss
        logits, train_op, cross_entropy_loss = optimize(model_output, label_ph, learning_rate_ph, num_classes_)

        # Initilize all variables
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Run the training step
        train_nn(session, epochs_, batch_size_, get_batches_fn, 
                 train_op, cross_entropy_loss, image_input,
                 label_ph, keep_prob, learning_rate_ph)

        # Save inference data
        helper.save_inference_samples(runs_dir, data_dir, session, img_shape_, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
