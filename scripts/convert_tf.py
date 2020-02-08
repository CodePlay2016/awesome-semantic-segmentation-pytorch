import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
import cv2
from vis_utils import visualization


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# tf_model_path = '../model/mapillary_selected/deeplabv3_resnet50_mapillary.pb'
tf_model_path = "../model/mapillary_test_1/deeplabv3_mobilenetv2_mapillary.pb"
tf_model_final_path = "../model/mapillary_test_1/deeplabv3_mobilenetv2_mapillary_final.pb"

def convert_model():
    # Load ONNX model and convert to TensorFlow format
    # model_onnx = onnx.load('../model/mapillary_selected/deeplabv3_resnet50_mapillary_2.onnx')
    model_onnx = onnx.load("../model/mapillary_test_1/deeplabv3_mobilenetv2_mapillary_2.onnx")

    tf_rep = prepare(model_onnx, strict=False, opset_version=11)

    # Export model as .pb file
    tf_rep.export_graph(tf_model_path)


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def complete_model():
    tf_graph = load_pb(tf_model_path)
    tf.disable_eager_execution()

    dummy_input = np.ones((1, 3, 320, 640))
    # Show tensor names in graph
    # for op in tf_graph.get_operations():
    #     print(op.values())
    with tf_graph.as_default(): # add op to graph
        input_tensor = tf_graph.get_tensor_by_name('input_image:0')
        output_tensor = tf_graph.get_tensor_by_name('seg_result:0')
        output_tensor = tf.compat.v1.transpose(output_tensor, perm=[0,2,3,1])
        img_shape = input_tensor.get_shape().as_list()[2:]
        pred_result = tf.image.resize_bilinear(output_tensor,
                                                img_shape,
                                                align_corners=True)
        prediction = tf.argmax(pred_result, 3, name='final_seg_result')

        with tf.Session(graph=tf_graph) as sess:
            output = sess.run(prediction, feed_dict={input_tensor: dummy_input})
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['final_seg_result'])
            with tf.gfile.GFile(tf_model_final_path, "wb") as f:
                f.write(constant_graph.SerializeToString())

    print(output)
    
    print(input_tensor.get_shape().as_list())
    print(output_tensor.get_shape().as_list())
    print(pred_result.get_shape().as_list())
    print(prediction.get_shape().as_list())

    
def test_model():
    image = cv2.imread("../tests/ori.png")
    image_raw = cv2.resize(image, (640, 320))
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))*127
    std  = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))*255
    image = np.transpose(image_raw, (2,0,1))
    image = np.expand_dims(image, 0)
    image = image.astype(np.float)
    image = (image - mean) / std

    tf_graph = load_pb(tf_model_final_path)
    sess = tf.Session(graph=tf_graph)
    input_tensor = tf_graph.get_tensor_by_name('input_image:0')
    output_tensor = tf_graph.get_tensor_by_name('final_seg_result:0')
    output = sess.run(output_tensor, feed_dict={input_tensor: image})
    output = np.squeeze(output)
    added_image = visualization(image_raw, output.astype(np.int), 'mapillary_partial', add_on=True)
    cv2.imwrite("tmp.png", added_image)
    print(output)

if __name__ == "__main__":
    # convert_model()
    # complete_model()
    test_model()