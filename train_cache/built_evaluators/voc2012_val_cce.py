from evaluate.evaluator import SemanticSegmentationEvaluator
from train_cache.built_datasets import voc2012_val
from train_cache.built_losses import cce


def build_evaluator(bsize=300, num_workers=1):
    dataset = voc2012_val.build()
    lf = cce.build_loss(dataset.border_index, 1, 0, 0, 0, -1, 0)

    evaluator = SemanticSegmentationEvaluator(dataset, lf, bsize, num_workers)
    return evaluator


if __name__ == '__main__':
    evaluator = build_evaluator()

    from tools.model_tools import load_model

    model_dir = r'../built_models/voc2012_models/r18'
    state_dict_path = r'../../results/voc2012/r18_large_bsize_hyper_search/lr=0.01/model.pth'
    model = load_model(model_dir, state_dict_path)

    evaluator.evaluate(model, 'cuda', 1)
