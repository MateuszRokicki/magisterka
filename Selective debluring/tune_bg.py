import tensorflow as tf
import numpy as np
from PIL import Image
from pixellib.semantic.deeplab import Deeplab_xcep_pascal
from pixellib.semantic import obtain_segmentation
import cv2
import time
from datetime import datetime


class alter_bg():
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, model_type="h5"):
        global model_file
        self.model_type = model_type
        model_file = model_type

        self.model = Deeplab_xcep_pascal()

    def load_pascalvoc_model(self, model_path):
        if model_file == "pb":
            self.graph = tf.Graph()

            graph_def = None

            with tf.compat.v1.gfile.GFile(model_path, 'rb') as file_handle:
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())

            if graph_def is None:
                raise RuntimeError('Cannot find inference graph')

            with self.graph.as_default():
                tf.graph_util.import_graph_def(graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=self.graph)

        else:
            self.model.load_weights(model_path)

    def change_bg_img(self, f_image, b_image, output_image_name=None, verbose=None, detect=None):
        if verbose is not None:
            print("processing image......")

        # ori_img = f_image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        # bg_img = b_image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        ori_img = f_image
        bg_img = b_image
        seg_image = self.segmentAsPascalvoc(ori_img)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        w, h, _ = ori_img.shape
        bg_img = cv2.resize(bg_img, (h, w))

        result = np.where(seg_image[1], ori_img, bg_img)
        if output_image_name is not None:
            cv2.imwrite(output_image_name, result)

        return result

    def segmentAsPascalvoc(self, image, process_frame=False):
        if model_file == "pb":

            # if process_frame == True:
            #     image = image_path
            # else:
            #     image = cv2.imread(image_path)

            h, w, n = image.shape

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
            target_size = (int(resize_ratio * w), int(resize_ratio * h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

            seg_image = batch_seg_map[0]
            raw_labels = seg_image
            labels = obtain_segmentation(seg_image)
            labels = np.array(Image.fromarray(labels.astype('uint8')).resize((w, h)))
            labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            return raw_labels, labels

        else:
            trained_image_width = 512
            mean_subtraction_value = 127.5

            if process_frame == True:
                image = image_path

            else:
                image = np.array(Image.open(image_path))

                # resize to max dimension of images from training dataset
            w, h, n = image.shape

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            ratio = float(trained_image_width) / np.max([w, h])
            resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
            resized_image = (resized_image / mean_subtraction_value) - 1

            # pad array to square image to match training images
            pad_x = int(trained_image_width - resized_image.shape[0])
            pad_y = int(trained_image_width - resized_image.shape[1])
            resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

            # run prediction
            res = self.model.predict(np.expand_dims(resized_image, 0))

            labels = np.argmax(res.squeeze(), -1)
            # remove padding and resize back to original image
            if pad_x > 0:
                labels = labels[:-pad_x]
            if pad_y > 0:
                labels = labels[:, :-pad_y]

            raw_labels = labels

            # Apply segmentation color map
            labels = obtain_segmentation(labels)
            labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

            new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            return raw_labels, new_img