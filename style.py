import tensorflow as tf
import numpy as np
from imageio import imread, imsave
import os
import time

def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

style = 'wave'
model = 'samples_%s' % style
content_image = 'assets/suzzallo-library.jpg'
result_image = 'suzzalo_%s.jpg' % style
X_image = imread(content_image)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join(model, 'fast_style_transfer.meta'))
saver.restore(sess, tf.train.latest_checkpoint(model))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
g = graph.get_tensor_by_name('transformer/g:0')

the_current_time()

gen_img = sess.run(g, feed_dict={X: [X_image]})[0]
gen_img = np.clip(gen_img, 0, 255) / 255.
imsave(result_image, gen_img)

the_current_time()