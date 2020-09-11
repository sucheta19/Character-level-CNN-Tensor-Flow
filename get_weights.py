import os

import tensorflow.compat.v1 as tf
from tensorflow import keras


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('trained_models/char_level_cnn.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./trained_models'))

  vars = tf.trainable_variables()
  print(vars) #some infos about variables...
  # vars_vals = sess.run(vars)
  # for var, val in zip(vars, vars_vals):
  #   print("var: {}, value: {}".format(var.name, val))
