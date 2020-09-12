import os
import json
import tensorflow.compat.v1 as tf
from tensorflow import keras


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('trained_models/char_level_cnn.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./trained_models'))

  # vars = tf.trainable_variables()
  # print(vars) #some infos about variables...
  # vars_vals = sess.run(vars)
  # for var, val in zip(vars, vars_vals):

  tvars = tf.trainable_variables()
  tvars_vals = sess.run(tvars)

  output_dict  = {}

  for var, val in zip(tvars, tvars_vals):
      output_dict[var.name] = val #.tolist()

  print (output_dict)
  # app_json = json.dumps(output_dict)
  # with open('data.json', 'w', encoding='utf-8') as f:
  #   json.dump(output_dict, f, ensure_ascii=False, indent=4)
