import tensorflow as tf
import utils
import numpy as np

with open("vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 16, 16, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"

graph = tf.get_default_graph()
for op in graph.get_operations():
    print op.name
cat = utils.load_image("cat.jpg")

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  #batch = cat.reshape((1, 224, 224, 3))
  batch = np.ones((1,16,16,3))
  #assert batch.shape == (1, 224, 224, 3)


  feed_dict = { images: batch }

  prob_tensor = graph.get_operation_by_name("import/pool5").outputs[0]
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

print prob.shape

#utils.print_prob(prob[0])


