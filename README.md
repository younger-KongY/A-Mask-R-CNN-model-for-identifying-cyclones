# Mask R-CNN for Identifying Cyclones

This is an application of [Mask R-CNN](https://arxiv.org/abs/1703.06870) . The source code is packed into a compressed package of Mask_RCNN-master_source_code.zip. More details can be obtained here, https://github.com/matterport/Mask_RCNN. 

We use the Mask R-CNN model for identifying extratropical cyclones.


# Train a new model starting from pre-trained COCO weights
python3 Mask_RCNN-master/cyclone/maskrcnn_cyclone.py train  --weight=coco

# Continue training a model that you had trained earlier
python3 Mask_RCNN-master/cyclone/maskrcnn_cyclone.py train  --weight=last

# Run trained Mask R-CNN model 
python3 Mask_RCNN-master/cyclone/maskrcnn_cyclone.py test --image=[path] --weight=last

