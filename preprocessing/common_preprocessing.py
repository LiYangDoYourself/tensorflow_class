#right 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def preprocess_image(image_org, output_height, output_width, is_training):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  #image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)

  image_exdim = tf.expand_dims(image_org, 0)

  #常规############################################################################
  image_crop = tf.image.resize_bilinear(image_exdim, [output_height, output_width],align_corners=False)
  ################################################################################################



  #筒子,截取高度的1/4######################################################################
  #bbox = tf.constant([0.25, 0.0, 1.0, 1.0],dtype=tf.float32,shape=[1, 4])#y1,x1,y2,x2
  #image_crop = tf.image.crop_and_resize(image_exdim,bbox,[0],[output_height,output_width])
  #print("crop_img:",image_crop)
  ############################################################################

  # tf.summary.image('image', image_crop)
  image = tf.squeeze(image_crop, [0])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5


  #image = tf.to_float(image)
  #image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
  #image = tf.subtract(image, 128.0)
  #image = tf.div(image, 128.0)
  return image
