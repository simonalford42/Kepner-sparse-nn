import tensorflow as tf
import numpy as np


def inference(a):
    with tf.variable_scope('logits'):
        b = tf.get_variable(
            'b',shape=(10,1), initializer=tf.truncated_normal_initializer(
                stddev=0.1, dtype=tf.float32))
    return b


def train():
    with tf.Graph().as_default():
        INIT_DICT = {'logits':{'b': tf.convert_to_tensor(np.ones((10,1),dtype=np.float32))}}
        inp = tf.Variable(5)
        outp = inference(inp)
        assign_ops = []
        for scope in INIT_DICT:
            print('scope= ' + scope)
            var_dict = INIT_DICT[scope]
            with tf.variable_scope(scope, reuse=True):
                for var in INIT_DICT[scope]:
                    print('var = ' + var)
                    assign_op = tf.assign(tf.get_variable(var), var_dict[var])
                    assign_ops.append(assign_op)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if INIT_DICT is not None:
                sess.run(assign_ops)
                for scope in INIT_DICT.keys():
                    with tf.variable_scope(scope, reuse=True):
                        for var in INIT_DICT[scope].keys():
                            print('assigned ' + scope + '/' + var + ': '
                                  + str(sess.run(tf.get_variable(var))))
            print(sess.run(outp))


if __name__ == '__main__':
    train()
