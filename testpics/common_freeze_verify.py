#coding:utf-8
import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.python.framework import graph_util

#固话成pb文件
def freeze_graph():
    output_node_names = "MobilenetV1/Predictions/Reshape_1" # 输出节点
    meta_path = "E:\\Classify_Data\\Models\\20190830_poker_3232_gray_cls_14_mobivenet\\model.ckpt-3188828"
    saver = tf.train.import_meta_graph(meta_path+".meta",clear_devices=True) #计算图

    with tf.Session() as sess:
        saver.restore(sess,meta_path)  # 加载参数和训练项 到会话中
        output_graph_def = graph_util.convert_variables_to_constants( # 将变量固定化
            sess=sess, # 哪一个会话
            input_graph_def=sess.graph_def, # 会话中的计算图
            output_node_names = output_node_names.split(",")
        )

    # 写入pb文件中
    pb_name ="xxx/xxx/frozen_graph.pb"
    with tf.gfile.GFile(pb_name,"wb") as f:
        f.write(output_graph_def.SerializeToString())  # 序列化的方式写入

    # 查看会话中计算图的节点信息
    for op in sess.graph.get_operations():
        print(op.name,op.values())


def src_thresold(dst):
    # dst=cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)

    if 0:
        expend_dst = cv2.resize(dst,(dst.shape[0]*3,dst.shape[1]*3),0,0)
        contours, hierarchy = cv2.findContours(expend_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp = np.ones(expend_dst.shape, np.uint8) * 255
        print(type(contours))
        print(len(contours))
        # for hire in range(len(contours)):
        cv2.drawContours(temp,contours,-1,(255,255,0),1,8)

        cv2.imshow("dst",temp)
        cv2.waitKey(0)

    print(dst.shape[0]/2,dst.shape[1]/2)
    M2 = cv2.getRotationMatrix2D((int(dst.shape[0]/2),int(dst.shape[1]/2)),60,1)
    rot_mat = cv2.warpAffine(dst,M2,(dst.shape[0],dst.shape[1]))
    cv2.imshow("rot_mat", rot_mat)
    cv2.waitKey(0)
    return rot_mat


# 检测一张图片
def interence_one_image(num_pb_path):
    with tf.gfile.GFile(num_pb_path,'rb') as f:
        graph_def = tf.GraphDef() #
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,    # 计算图
                            input_map=None,
                            return_elements=None,
                            name="",
                            op_dict=None,
                            producer_op_list=None
                            )


    image_batch = graph.get_tensor_by_name("input:0") # 输入节点
    softmax = graph.get_tensor_by_name("MobilenetV1/Predictions/Reshape_1:0") # 输出节点

    read_src = cv2.imread("C:\\Users\\admin_user\\Desktop\\2019-09-26-17-01-15_8324.jpg",0)
    cv2.imshow("原始",read_src)
    src=src_thresold(read_src)
    dst=cv2.resize(src,(32,32),0,0)
    dst = dst/255-0.5   # [-0.5,0.5]
    dst = dst.reshape([1,32,32,1]) # 调正大小到合适的比列



    with tf.Session(graph=graph) as sess:
        results = sess.run(softmax,feed_dict={image_batch:dst})
        results = np.squeeze(results)  # 删除单一维度

        top_k=results.argsort()[-1:]  # 排序获取数值最大的下标  [3]
        print("result:",top_k[0],results[top_k[0]])  # 打印最大的分类结果

if __name__ == '__main__':

    # pb文件路径
    num_pb_path = "E:\\Android_Data\\PokerDealer\\AndroidEyesDealer\\app\\src\\main\\assets\\jni\\20191112_quantized_graph_num.pb"
    # src = cv2.imread("C:\\Users\\admin_user\\Desktop\\123.jpg",0)


    switch_o = "t"
    if switch_o=='t':
        interence_one_image(num_pb_path)
    elif switch_o=='g':
        freeze_graph()
