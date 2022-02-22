# TA-CycleGAN
TA-CycleGAN

Folder STEDB_BB contains dendritic images translated into new STED images. These train/valid splits were used to train a UNet for the segmentation of F-actin rings and fibers in new STED images. To test the segmentation model, run the following line:

```
python3 test.py --dataroot=STEDB_BB --dataset_mode=two_masks_channels --phase=valid --preprocess=none --model=segmentation --netS=unet_128 --name=STEDB_segmentation200 --epoch=best
```

Change parameters ```--dataroot``` (folder), ```--dataset-model``` (dataloader) and ```--phase``` (subfolder) to test on other images.

To test in STED_Julia2022 images:

```
python3 test.py --dataroot=STED_Julia2022 --dataset_mode=single --phase=test --preprocess=none --model=segmentation --netS=unet_128 --name=STEDB_segmentation200 --epoch=best
```
