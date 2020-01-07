import cv2
import os, argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



#read an image as a tensor form image file
def read_tensor_from_image_file(file_name, input_height=128, input_width=128,
				input_mean=255, input_std=0.5):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 1,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 1,
                                        name='jpeg_reader')

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.subtract(tf.divide(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result






def inference(frozen_graph_filename,image_path_in,image_path_out,image_width,image_height,label_offset):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )

    #for op in graph.get_operations():
    #    print(op.name)

    #lenet_5#####################################################
    '''
    image_batch = graph.get_tensor_by_name('image_batch:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predict_val = graph.get_tensor_by_name('TopKV2:0')
    predict_idx = graph.get_tensor_by_name('TopKV2:1'),
    '''
    #other = graph.get_tensor_by_name('Softmax:0')
    ###############################################################

    #mobilenet_ofc
    input_node_name=graph.get_tensor_by_name('input:0')#prefetch_queue/fifo_queue:0
    output_node_name=graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')#MobilenetV1/Predictions/Softmax:0

    # We launch a Session
    with tf.Session(graph=graph) as sess:

        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        testAtrainDir = os.listdir(image_path_in)

        for tt in testAtrainDir:#func file:validation_images,train_images
            print("file_path:",tt)
            if os.path.isdir(image_path_in + "/" + tt):
                allTrue = 0
                allSum = 0
                ttOutpath = image_path_out + "/" + tt

                os.makedirs(ttOutpath)

                pathDir = os.listdir(image_path_in + "/" + tt)

                for allDir in pathDir:  # label list:1,2,3,4...
                    setTrue = 0
                    setSum = 0

                    print(allDir + '\n')

                    label = allDir  # 0,1,2,3,4,5,6,7
                    os.makedirs(ttOutpath + "/" + allDir)

                    eveTongZi = os.path.join(image_path_in + "/" + tt + '/' + allDir)
                    print(eveTongZi + "\n")
                    pathImg = os.listdir(eveTongZi)
                    eveImg_list = []
                    for allImg in pathImg:
                        eveImg_list.append(allImg)

                    for ind, item in enumerate(eveImg_list):
                        if 1:#ind < 2000:#TODO:只使用前2000
                            allSum += 1
                            setSum += 1

                            evePath = eveTongZi + "/" + item
                            srcImg = cv2.imread(evePath, 0)
                            dstImg = cv2.resize(srcImg, (image_width, image_height), None, 0, 0)
                            srcImg_f = dstImg / 255 - 0.5
                            dst = srcImg_f.reshape([-1, image_height, image_width, 1])
                            results = sess.run(output_node_name,
                                               feed_dict={input_node_name: dst})  # ,keep_prob: 1
                            results = np.squeeze(results)
                            top_1 = results.argsort()[-1:][::-1]
                            # print(top_1[0], results[top_1[0]])
                            # if label == str(top_1[0] + 1):
                            if label == str(top_1[
                                                0] + label_offset):  # label都是从0开始的，label是文件夹中的名字；新的label值为真实的label，老版本因为有背景类，要减一：top_1[0] - 1
                                setTrue += 1
                                allTrue += 1
                            cv2.imwrite(
                                # 1_(0.99)_
                                # ttOutpath + "/" + allDir + "/" +str(top_1[0]+1)+'('+str(results[top_1[0]])[:4]+')_'+str(setSum) + ".jpeg",srcImg)

                                ttOutpath + "/" + allDir + "/" + str(top_1[0] + label_offset) + '(' + str(
                                    results[top_1[0]])[:4] + ')_' + item, srcImg)


                            # ttOutpath + "/" + allDir + "/c(" + str(top_1[0]) +
                            # ')-p(' + str(results[top_1[0]])[:4] + ')-' + item, srcImg)


                            # lenet_5#####################################################################################
                            # pre_val,pre_indx=sess.run([predict_val,predict_indx],feed_dict={image_batch:dst,keep_prob:1})
                            # print('result：', pre_val[0][0], pre_indx[0][0][0])
                            # if label == str(pre_indx[0][0][0] - 1):
                            #    setTrue += 1
                            #    allTrue += 1
                            # cv2.imwrite(
                            #    ttOutpath + "/" + allDir + "/" +str(pre_indx[0][0][0] - 1)+'('+str(pre_val[0][0])[:4]+')_'+str(setSum) + ".jpeg",srcImg)

                    if(setSum==0):
                        print('%s:zero'%(label))
                        newSetName = ttOutpath + "/" + allDir + '_[' + str(setTrue) + ':' + str(setSum) + ']_(null)'
                        os.renames(ttOutpath + "/" + allDir, newSetName)
                    else:
                        print('%s:%0.3f'%(label,setTrue/setSum))
                        newSetName=ttOutpath + "/" + allDir+'_['+str(setTrue)+':'+str(setSum)+']_('+str(setTrue/setSum)[:5]+')'
                        os.renames(ttOutpath + "/" + allDir,newSetName)
                print('total:',str(allTrue/allSum)[:5])
                newAllName=ttOutpath+'_['+str(allTrue)+':'+str(allSum)+']_('+str(allTrue/allSum)[:5]+')'
                os.renames(ttOutpath,newAllName)








def inference_one_image(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def

    with tf.gfile.GFile(frozen_graph_filename,'rb') as f:


        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )


    #for op in graph.get_operations():
    #    print(op.name)

    #lenet_5############################################################
    #image_batch = graph.get_tensor_by_name('image_batch:0')
    #keep_prob = graph.get_tensor_by_name('keep_prob:0')
    #predict_val = graph.get_tensor_by_name('TopKV2:0')
    #predict_idx = graph.get_tensor_by_name('TopKV2:1'),
    # other = graph.get_tensor_by_name('Softmax:0')
    #####################################################################

    #mobilenet@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    image_batch = graph.get_tensor_by_name('input:0')#prefetch_queue/fifo_queue
    softmax = graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    main_path = "E:\\cut_pic\\20190821\\restore_cls_5\\2new"
    picname=os.listdir(main_path)
    pic_path=os.path.join(main_path,picname[20])
    srcImg = cv2.imread('C:\\Users\\admin_user\\Desktop\\scripts\\testpics\\9.jpg', 0)
    # srcImg = cv2.imread(pic_path, 0)
    cv2.imshow('src', srcImg)
    cv2.waitKey(100)

    dst = cv2.resize(srcImg, (32, 32), None, 0, 0)

    dst = dst / 255 - 0.5
    # dst = dst.reshape([1, 128, 128, 3])
    dst = dst.reshape([1, 32, 32, 1])
    #ret=read_tensor_from_image_file('/home/tnc/PycharmProjects/DATA/tongzi_image_128d/validation_images/3/3(2_0).jpeg')
    # We launch a Session
    with tf.Session(graph=graph) as sess:

        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        #result = sess.run([predict_val, predict_idx], feed_dict={image_batch: dst})#, keep_prob: 1
        #print('result：[', result[1][0][0], '', result[0][0][0], ']')
        # print('other:', result[2])
        results = sess.run(softmax, feed_dict={image_batch: dst})#, keep_prob: 1
        results = np.squeeze(results)

        print(results)
        top_k = results.argsort()[-1:][::-1]##no bg

        print("result:",top_k[0],results[top_k[0]])


if __name__ == '__main__':

    #model_folder='/home/tnc/PycharmProjects/mobilenet/model_tongzi_ofc_img_80d/save'#model_tongzi_025'#gzNet/model_tongzi
    model_folder='.'

    switch='t'#f or t


    if switch=='t':#frozen_graph
        inference_one_image(model_folder + '/frozen_graph.pb')#frozen_model.pb
    elif switch=='i':

        model_path=model_folder + '/frozen_graph.pb'#frozen_model.pb
        #image_in:../poker_side/poker_side_jpeg/train_images/1/1.jpeg,,path:../poer_side_jpeg/
        image_in='/home/tnc/PycharmProjects/DATA/tongzi/d'
        image_out='/home/tnc/PycharmProjects/DATA/tongzi/test_tongzi_part_doubledragon_jpeg'
        inference(model_path,image_in,image_out,78,52,1)
    else:
        print('please input correct key word')





