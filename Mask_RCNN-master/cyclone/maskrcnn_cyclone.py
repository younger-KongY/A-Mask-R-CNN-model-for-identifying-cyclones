"""
Mask R-CNN
Train on the SUN dataset.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Mask RCNN is Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 sun.py train --dataset=/Users/ekaterina/Desktop/diploma/mask_rcnn/datasets/SUNRGBD/train --weights=coco
    # Resume training a model that you had trained earlier
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=last
    # Train a new model starting from ImageNet weights
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=imagenet
    # Apply color splash to an image
    python3 sun.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 sun.py splash --weights=last --video=<URL or path to file>?
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
import struct 

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SunConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cyclone"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    #IMAGES_PER_GPU = 1
    GPU_COUNT = 2
    IMAGES_PER_GPU = 4
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 20

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8
    IMAGE_MIN_DIM = 192
    IMAGE_MAX_DIM = 192
    TRAIN_ROIS_PER_IMAGE    =       32
    RPN_ANCHOR_SCALES       =      (8, 16, 32, 64, 128)    
############################################################
#  Dataset
############################################################

class SunDataset(utils.Dataset):

    def load_sun(self, count, height, width, img_floder, mask_floder, imglist,dataset_root_path):
            """Generate the requested number of synthetic images.
            count: number of images to generate.
            height, width: the size of the generated images.
            """
            # Add classes
            self.add_class("shapes", 1, "cyclone")
            for i in range(count):
                filestr = imglist[i].split(".")[1]
                mask_path = os.path.join(mask_floder, 'mask.'+ filestr + ".grd")
                self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                               width=width, height=height, mask_path=mask_path)  
                               
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        #shapes = info['shapes']
        #count = len(shapes)     
        with open(info['mask_path'], 'rb') as fp:
          data_raw2 = struct.unpack('f'*36864,fp.read(4*36864))
          fp.close()                 
        data_np2=np.array(data_raw2, dtype='uint8')
        mask=np.zeros(shape=(data_np2.shape[0],np.max(data_np2)), dtype=np.uint8)
        mask_size=np.max(data_np2)
        for i in range(np.max(data_np2)):
          for j in range(data_np2.shape[0]):
            if data_np2[j]==i+1:
              mask[j,i]=1
        mask=mask.reshape((192,192,mask.shape[1])) 
        # Handle occlusions--to avoid overlay
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(mask_size-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.--- all belong to cyclone class
        class_ids = np.ones(shape=(np.max(data_np2)), dtype=np.int32)
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sun":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img_path=info['path']
        #print(img_path)
        im = Image.open(img_path)
        image=np.array(im,dtype='uint8')
        return image

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_root_path="/media/webdisk/disk1/tmp_data/kongy/review/git_public/Mask_RCNN-master/cyclone"
    img_floder  = os.path.join(dataset_root_path, "z850_png/2000")
    mask_floder = os.path.join(dataset_root_path, "mask_label/2000")
    imglist = os.listdir(img_floder)
    count = len(imglist)
    width = 192
    height = 192
    dataset_train = SunDataset()
    dataset_train.load_sun(count, 192, 192, img_floder, mask_floder, imglist,dataset_root_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SunDataset()
    dataset_val.load_sun(count, 192, 192, img_floder, mask_floder, imglist,dataset_root_path)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='all')

                
#def color_splash(image, mask):
#    """Apply color splash effect.
#    image: RGB image [height, width, 3]
#    mask: instance segmentation mask [height, width, instance count]
#    Returns result image.
#    """
#    # Make a grayscale copy of the image. The grayscale copy still
#    # has 3 RGB channels, though.
#    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#    # Copy color pixels from the original color image where mask is set
#    if mask.shape[-1] > 0:
#        # We're treating all instances as one, so collapse the mask into one layer
#        mask = (np.sum(mask, -1, keepdims=True) >= 1)
#        splash = np.where(mask, image, gray).astype(np.uint8)
#    else:
#        splash = gray.astype(np.uint8)
#    return splash
#
#
def detect_and_evaluation(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
#        # Run model detection and generate the color splash effect
#        print("Running on {}".format(args.image))
#        # Read image
         
        #Read image filename        
        #batch processing
        z850_path      = args.image
        image_filename = os.lisdir(z850_path)
        image_filename.sort()
        image_num      = len(image_filename)
        
        for i in range(image_num): 
          if image_filename[i].endswith('.png') :
            masks_date = image_filename[i].split('.')[1]
            args.image = os.path.join(z850_path, image_filename[i])
            image = skimage.io.imread(args.image)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            masks = r['masks']
            masks = masks.astype(np.float)
            print(masks.shape)
            data_vector = np.reshape(masks, [-1])
            #number of cyclone
            num_cyclone = masks.shape[2]
            #print(num_cyclone)
            if os.path.exists('./maskrcnn_out/masks') is not True :
            	os.makedirs('./maskrcnn_out/masks')
            	
            with open('./maskrcnn_out/masks/masks_' + masks_date + '_' + str(100+masks.shape[2])[1:3] + '.grd', 'wb')as fp:
              data_out = struct.pack('f'*192*192*num_cyclone,*data_vector)
              fp.write(data_out)
          
            print("Saved to ", './maskrcnn_out/masks/masks_' + masks_date + '_' + str(100+masks.shape[2])[1:3] + '.grd')
#    elif video_path:
#        import cv2
#        # Video capture
#        vcapture = cv2.VideoCapture(video_path)
#        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#        fps = vcapture.get(cv2.CAP_PROP_FPS)
#
#        # Define codec and create video writer
#        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#        vwriter = cv2.VideoWriter(file_name,
#                                  cv2.VideoWriter_fourcc(*'MJPG'),
#                                  fps, (width, height))
#
#        count = 0
#        success = True
#        while success:
#            print("frame: ", count)
#            # Read next image
#            success, image = vcapture.read()
#            if success:
#                # OpenCV returns images as BGR, convert to RGB
#                image = image[..., ::-1]
#                # Detect objects
#                r = model.detect([image], verbose=0)[0]
#                # Color splash
#                splash = color_splash(image, r['masks'])
#                # RGB -> BGR to save image to video
#                splash = splash[..., ::-1]
#                # Add image to video writer
#                vwriter.write(splash)
#                count += 1
#        vwriter.release()
    #print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
#    parser.add_argument('--dataset', required=False,
#                        metavar="/path/to/balloon/dataset/",
#                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

#    # Validate arguments
#    if args.command == "train":
#        assert args.dataset, "Argument --dataset is required for training"
#    elif args.command == "splash":
#        assert args.image or args.video,\
#               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
#    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SunConfig()
    else:
        class InferenceConfig(SunConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    #input()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        detect_and_evaluation(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
