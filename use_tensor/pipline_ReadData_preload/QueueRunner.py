#coding:utf-8
'''
Created on Aug 22, 2018

@author: root
'''
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [['null'], ['1null'], ['1null'], ['1null'], ['1null']]#[['csv1_1', 'csv1_2', 'csv1_3', 'csv1_4', 'csv1_5']]#

print (record_defaults)

col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(2):
    # Retrieve a single instance:
    fl,val,example, label = sess.run([reader,value,features, col5])
    print fl,"-->",val,"-->",example,"-->",label
  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
    pass