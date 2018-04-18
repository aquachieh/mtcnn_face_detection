
import tensorflow as tf
import numpy as np
import detect_face

import cv2
from PIL import Image,ImageDraw,ImageFont

with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False,gpu_options.allow_growth=True))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

image = cv2.imread("/.../s0.jpg")
cv2pil = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
imagePIL = Image.fromarray(cv2pil)
imagePILDraw = ImageDraw.Draw(imagePIL)
#img = facenet.to_rgb(img)
bounding_boxes, _ = detect_face.detect_face(cv2pil, minsize, pnet, rnet, onet, threshold, factor)
print(bounding_boxes)
print(len(bounding_boxes))
for j, face in enumerate(bounding_boxes):
    print ("---",j)
    print (face)
    imagePILDraw.line([(face[0],face[1]),(face[2],face[1]),(face[2],face[3]),(face[0],face[3]),(face[0],face[1])], fill=(255,0,0),width=4)
imagePIL.save("/media/sophiewang/DATA/mtcnn/sample/s0_b.jpg")
