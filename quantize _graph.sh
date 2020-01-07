quantize graph



python quantize_graph.py --input=E:/cls_14_20190824.pb --output_node_names="MobilenetV1/Predictions/Reshape_1" --output=E:/quantized_graph.pb --mode=eightbit

echo "Quantizing weights to ${MODEL_FOLDER}/quantized_graph.pb"
bazel run tensorflow/tools/graph_transforms:transform_graph -- \
  --in_graph=${MODEL_FOLDER}/frozen_graph.pb \
  --out_graph=${MODEL_FOLDER}/quantized_graph.pb \
  --inputs=input --outputs=MobilenetV1/Predictions/Reshape_1 \
  --transforms='fold_constants fold_batch_norms quantize_weights'




echo "mobilenet-------------------------------------------------------------------------------------------"
cd tensorflow-master
bazel build tensorflow/tools/graph_transforms:transform_graph
=============================================================
 tensorflow-master/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph="/home/tnc/PycharmProjects/classify/model_tongzi_gray_c10_s78x52_mbnetv1_2019032901/frozen_graph.pb" \
  --out_graph="/home/tnc/PycharmProjects/classify/model_tongzi_gray_c10_s78x52_mbnetv1_2019032901/quantized_graph.pb" \
  --inputs="input" \
  --outputs="MobilenetV1/Predictions/Reshape_1" \
  --transforms="
    fold_batch_norms
    fold_constants
    quantize_weights"

echo "--------------------------------------------------------------------------------------------"


echo "lenet-------------------------------------------------------------------------------------------"
cd tensorflow-master
bazel build tensorflow/tools/graph_transforms:transform_graph
=============================================================
 tensorflow-master/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph="/home/tnc/PycharmProjects/classify/model_lenet_poker_patn_rgb_c4_s32_010602/frozen_graph.pb" \
  --out_graph="/home/tnc/PycharmProjects/classify/model_lenet_poker_patn_rgb_c4_s32_010602/quantized_graph.pb" \
  --inputs="input" \
  --outputs="Predictions/Reshape_1" \
  --transforms="
    fold_batch_norms
    fold_constants
    quantize_weights"

echo "--------------------------------------------------------------------------------------------"









echo "object detection------------------------------------------------------------------------------"
 tensorflow-master/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph="/home/tnc/PycharmProjects/object_detection/tongzi_ssd_mobilenet_img_0375_complete/output_12035/frozen_inference_graph.pb" \
  --out_graph="/home/tnc/PycharmProjects/object_detection/tongzi_ssd_mobilenet_img_0375_complete/output_12035/quantized_graph.pb" \
  --inputs="image_tensor" \
  --outputs="detection_boxes,detection_scores,detection_classes,num_detections" \
  --transforms="
    fold_batch_norms
    fold_constants
    quantize_weights"
echo "-----------------------------------------------------------------------------------------------"


