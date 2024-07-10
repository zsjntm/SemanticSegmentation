from train_cache.built_losses import cce


def build_loss(*args):
    """
    args[0]: ignore_index
    args[1]: reduction 0->'mean', 1->'sum'
    args[2]: aux_weight
    """

    class Loss:
        def __init__(self):
            self.cce = cce.build_loss(args[0], args[1])
            self.aux_weight = args[2]

        def __call__(self, outputs, targets):
            """
            :param outputs: (seg, seg_aux) 两个都是(b, c, h, w)
            :param targets: (b, 1, h, w)
            """

            return self.cce(outputs[0], targets) + self.aux_weight * self.cce(outputs[1], targets)

    return Loss()
