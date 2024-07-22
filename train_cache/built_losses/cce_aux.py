from train_cache.built_losses import cce
from losses.loss_functions import SSCrossEntropy

class Loss:
    def __init__(self, lf_final, lf_aux, aux_weight):
        self.lf_final = lf_final
        self.lf_aux = lf_aux
        self.aux_weight = aux_weight

    def __call__(self, outputs, targets):
        """
        将final_outputs, aux_output分别与targets运算，然后将两个损失加权和
        :param outputs: (final_outputs, aux_outputs)
        :param targets: targets
        """

        loss_final = self.lf_final(outputs[0], targets)
        loss_aux = self.lf_aux(outputs[1], targets)
        return loss_final + self.aux_weight * loss_aux


def build_loss(*args):
    """
    :param args[0~6]: cce的build_loss所需的参数
    :param args[7]: aux_loss的ignore_idx
    :param args[8]: aux_loss的reduction: 0->'mean' 1->'sum'
    :param args[9]: aux_loss的aux_weight
    """


    lf_final = new_cce.build_loss(*args[:7])

    ignore_idx = int(args[7])
    reduction = 'mean' if args[8] == 0 else 'sum'
    aux_weight = args[9]
    lf_aux = SSCrossEntropy(ignore_idx, reduction)

    lf = Loss(lf_final, lf_aux, aux_weight)
    return lf


