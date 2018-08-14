from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import tensorflow as tf
sys.path.append('/home/gridsan/salford/tf/model_pruning/')
import retrain
from mnist_input import MnistData
tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR = retrain.MODEL_DIR
MODEL_FN = retrain.MODEL_FN

TRAIN_STEPS = retrain.TRAIN_STEPS
TEST_SET_SIZE = retrain.TEST_SET_SIZE
TEST_FREQ = retrain.TEST_FREQ

def monitor_test_mnist():
    with tf.Graph().as_default():
        # I do this right away so the other script can't pull a fast one on it
        latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
        if latest_ckpt is None:
            start_step = -1
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
        else:
            start_step = int(latest_ckpt.split('/')[-1].split('-')[-1])
        tested_ckpt_numbers = set([start_step])
        # we want the batch size for testing to be the entire dataset
        input_pipe = MnistData(TEST_SET_SIZE)
        test_features, test_labels = input_pipe.build_test_data_tensor()

        global_step = tf.contrib.framework.get_or_create_global_step()

        accuracy = test_graph(test_features, test_labels)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(MODEL_DIR + '/eval/')

        saver = tf.train.Saver()

        before = set(f for f in os.listdir(MODEL_DIR))
        ckpt_number = start_step
        while ckpt_number - start_step < TRAIN_STEPS:
            after = set(f for f in os.listdir(MODEL_DIR))
            added = [f for f in after if not f in before]
            ckpt_files = [f for f in added if 'model.ckpt' in f]
            # do now in case we 'continue' below
            before = after

            if ckpt_files:
                a = ckpt_files[0]
                ckpt_number = int(a[11:11+a[11:].index('.')])
                if ckpt_number < 10:
                    tested_ckpt_numbers.add(ckpt_number)
                    continue
                if ckpt_number in tested_ckpt_numbers:
                    continue
                checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    saver.restore(sess, checkpoint)
                    acc = accuracy.eval()
                    gs = global_step.eval()
                    summary_writer.add_summary(merged.eval(), global_step=gs)

                    coord.request_stop()
                    coord.join(threads)
                    sess.close()
                print('Model at step ' + str(ckpt_number) + ': acc = ' + str(acc))
                tested_ckpt_numbers.add(ckpt_number)
            time.sleep(1)


def test_graph(features, labels):
    labels = tf.cast(labels, tf.int64)

    logits = MODEL_FN(features)

    predictions = tf.argmax(input=logits, axis=1)

    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, labels), tf.float32))
    # because it gets put in the eval folder, this doesn't interfere with train
    tf.summary.scalar('accuracy', accuracy)

    return accuracy


if __name__ == '__main__':
    monitor_test_mnist()

