import os, time
from pathlib import Path
from train_recipes import train, VOC2012_DEFAULT, CITYSCAPES_DEFAULT, CITYSCAPES_DAD_STAGE1, CITYSCAPES_DAD_STAGE2, CITYSCAPES_DADV2_STAGE1, CITYSCAPES_DADV2_STAGE2

"sample"
model_dir = r'train_cache/built_models/voc2012_models/r18'
checkpoint_dir = r'results/voc2012/r18'
bsize, nw = 256, 1
epoch = 60
train(model_dir, checkpoint_dir, bsize, nw, epoch, VOC2012_DEFAULT)


