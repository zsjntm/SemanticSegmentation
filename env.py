from pathlib import Path

PROGRAM_DIR = Path(r'D:\DLWS\PROGRAMS\SemanticSegmentation')  # 该项目的当前位置，该项目使用绝对路径
VOC2012_DIR = PROGRAM_DIR / r'data/voc2012'  # voc数据集所在位置，绝对路径
CITYSCAPES_DIR = PROGRAM_DIR / r'data/cityscapes'  # cityscapes数据集所在位置，绝对路径
TRAIN_PY_PATH = PROGRAM_DIR / 'train.py'  # train.py的路径

if __name__ == '__main__':
    "test code"
    # print(PROGRAM_DIR)
    # print(VOC2012_DIR)
    # print(CITYSCAPES_DIR)

    # print(Path.cwd())


