from evaluate.evaluator import SemanticSegmentationEvaluator
from train_cache.built_datasets import cityscapes_val
from train_cache.built_losses import cce

def build_evaluator(bsize=25, num_workers=1):
    dataset = cityscapes_val.build()
    lf = cce.build_loss(dataset.border_index, 1, 0, 0, 0, -1, 0)
    return SemanticSegmentationEvaluator(dataset, lf, bsize, num_workers)

if __name__ == '__main__':
    evaluator = build_evaluator()

    from tools.model_tools import load_model

    model_dir = r'../built_models/cityscapes_models/esnet_upgrade/esnet_u+ddr_revise2'
    state_dict_path = r'../../results/cityscapes/esnet_u+ddr_revise2/model.pth'
    model = load_model(model_dir, state_dict_path)

    evaluator.evaluate(model, 'cuda', 1)
