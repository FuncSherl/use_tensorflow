# coding:utf-8
'''
Created on Aug 22, 2018

@author: root
'''
import tensorflow as tf



filenames=["file0.csv", "file1.csv"]

def someread(filename_queue):
    reader = tf.TextLineReader()#
    key, value = reader.read(filename_queue)
    
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [['null'], ['1null'], ['1null'], ['1null'], ['1null']]  # [['csv1_1', 'csv1_2', 'csv1_3', 'csv1_4', 'csv1_5']]#
    
    #print (record_defaults)
    
    col1, col2, col3, col4, col5 = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.stack([col1, col2, col3, col4])
    return features, col5

def fill_queue(batch_size=2, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
    
    example, label=someread(filename_queue)
    
    min_after_dequeue = 4
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], 
      batch_size=batch_size, 
      capacity=capacity,
      num_threads=3,
      min_after_dequeue=min_after_dequeue)
      
    with tf.Session() as sess:
        #!~~~~!!!!if 'num_epochs' is not None,the below must be done
        init_op = tf.local_variables_initializer() #here maust be this but not global_variables_initializer()
        sess.run(init_op)
        # Start populating the filename queue.
        
        coord = tf.train.Coordinator()#this helps manage the threads,but without it ,it still works
        threads = tf.train.start_queue_runners(coord=coord)#
    
        try:
            while not coord.should_stop():
                # Run training steps or whatever!!!!!!!!
                example, label = sess.run([ example_batch, label_batch])
                print example, "-->", label
        
        except tf.errors.OutOfRangeError:
            print 'Done training -- epoch limit reached'
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        
        
        coord.join(threads)
        

if __name__ == '__main__':
    fill_queue(5,2)
