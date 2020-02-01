import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

tf_model_path = '../model/mapillary_selected/deeplabv3_resnet50_mapillary.pb'

# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('../model/mapillary_selected/deeplabv3_resnet50_mapillary.onnx')

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph(tf_model_path)


# def load_pb(path_to_pb):
#     with tf.gfile.GFile(path_to_pb, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph

# tf_graph = load_pb(tf_model_path)
# sess = tf.Session(graph=tf_graph)

# # Show tensor names in graph
# for op in tf_graph.get_operations():
#   print(op.values())

# output_tensor = tf_graph.get_tensor_by_name('seg_result:0')
# input_tensor = tf_graph.get_tensor_by_name('input_image:0')
# dummy_input = np.ones((1, 3, 320, 640))
# output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
# print(output)