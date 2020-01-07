#coding:utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util

import cv2
import numpy as np

def freezegraph(input_checkpoint, output_pb):
    output_node_names = "MobilenetV1/Predictions/Reshape_1"
    saver = tf.train.import_meta_graph(input_checkpoint+".meta",clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess,input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants( # 模型持久化将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(",")
        )
        with tf.gfile.GFile(output_pb,'wb') as f:
            f.write(output_graph_def.SerializeToString())   # 序列化输出

        for op in sess.graph.get_operations():
            print(op.name, op.values())
        print("%d ops in the final graph." % len(output_graph_def.node))


def test_picture(pb_path, image_path):

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path,"rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def,name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            input_tensor_name = sess.graph.get_tensor_by_name("input:0")
            output_tensor_name = sess.graph.get_tensor_by_name("MobilenetV1/Predictions/Reshape_1:0")


            # 读取测试图片
            im=cv2.imread(image_path,0)
            res_im = cv2.resize(im,(24,24),0,0)

            res_im = res_im/255-0.5
            res_im  =res_im.reshape([1,32,32,1])
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor_name, feed_dict={input_tensor_name: res_im
                                                     })
            print("out:{}".format(out))
            # score = tf.nn.softmax(out, name='pre')
            # class_id = tf.argmax(score, 1)
            # print("pre class_id:{}".format(sess.run(class_id)))
    pass


if __name__ == '__main__':

    input_checkpoint="./testpics/model.ckpt-203539"
    output_pb="./testpics/frozen_graph.pb"

    # freezegraph(input_checkpoint,output_pb)

    image_path="./testpics/1709.jpg"
    test_picture(output_pb,image_path)