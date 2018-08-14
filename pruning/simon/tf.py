import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
tf.logging.set_verbosity(tf.logging.INFO)

LR_NAMES = ['exp0.2-1000-0.75sc', 'exp0.2-100-0.96sc', 'exp0.2-2000-0.5sc']
LR_NAME = '0.1'

def lenet5_model_fn(features, labels, mode):
    '''Model function for lenet5 model, based on how implemented in caffe examp,le model.'''
    # Input Layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=20,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=50,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

    pool2_flat_size = np.prod(pool2.shape.as_list()[1:])

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, pool2_flat_size])
    
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=500,
            activation=tf.nn.relu)

    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(
            inputs=dropout,
            units=10)

    return estimator_fn_complete(features, labels, mode, logits)

def estimator_fn_complete(features, labels, mode, logits):
    '''
    this method does the rest of the work needed to define the model_fn for 
    an estimator. It is here because lenet5 and lenet300, and other models,
    share much the details.
    '''
    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        acc = tf.reduce_mean(tf.cast(tf.equal(predictions['classes'], tf.cast(labels, tf.int64)), tf.float32))
        tf.summary.scalar('training_accuracy', acc)

        global_step = tf.train.get_global_step()
        learning_rate = 0.2
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:      
        accuracy = tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])
        eval_metric_ops = { 'accuracy': accuracy} 
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def dummy_fn(features, labels, mode):
    input_layer = features['x']
    
    logits = tf.layers.dense(
            inputs=input_layer,
            units=10,
            activation=tf.nn.relu)
 
    predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) 

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.no_op()
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def lenet300_100_model_fn(features, labels, mode):
    input_layer = features['x']
    
    dense1 = tf.layers.dense(
            inputs=input_layer,
            units=300,
            activation=tf.nn.relu)
    
    dense2 = tf.layers.dense(
            inputs=dense1,
            units=100,
            activation=tf.nn.relu)

    logits = tf.layers.dense(
            inputs=dense2,
            units=10)
    
    return estimator_fn_complete(features, labels, mode, logits)

def lenet5(model_dir='lenet5'):
    return tf.estimator.Estimator(
            model_fn=lenet5_model_fn,
            model_dir='/home/gridsan/salford/tf/saved/' + model_dir + '/')

def lenet300_100(model_dir='lenet300_100'):
    return tf.estimator.Estimator(
            model_fn=lenet300_100_model_fn,
            model_dir='/home/gridsan/salford/tf/saved/' + model_dir + '/')

def dummy(model_dir='dummy'):
    return tf.estimator.Estimator(
            model_fn=dummy_fn,
            model_dir='/home/gridsan/salford/tf/saved/' + model_dir + '/')

def train_mnist(model, num_steps, test_freq=None):
    mnist = input_data.read_data_sets('/home/gridsan/salford/mnist/data/') 
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    steps = 0
    if test_freq == None:
        test_freq = num_steps
        
    while steps < num_steps:
        to_go = min(num_steps - steps, test_freq)
        train(train_data, train_labels, model, to_go)
        test(eval_data, eval_labels, model)
        steps += to_go

def test_mnist(model):
    mnist = input_data.read_data_sets('/home/gridsan/salford/mnist/data/')
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    test(eval_data, eval_labels, model)

def train(train_data, train_labels, classifier, iters):
    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    classifier.train(
            input_fn=train_input_fn,
            steps=iters)

def test(eval_data, eval_labels, classifier):
    # test 
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)

if __name__=='__main__':
    train_mnist(lenet300_100('lenet300_100_2'), 20000, 500)
