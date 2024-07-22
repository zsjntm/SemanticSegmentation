# Semantic Segmentation
a good project to implement all kinds of lab of semantic segmentation

usage:

1.put data files to data directory of this program, specific explanation in readme.md of data directory.

2.revise the variables of env.py to be correct.

3.create a directory which contains a model.py, and model.py contains a build() function, and it can return a torch model. such as train_cache/built_models/r18.

4.in start.py, revise model_dir to directory path created just, and revise checkpoint_dir to the path which you want the training results in.

5.use 'python start.py' in cmd.