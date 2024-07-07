# from train_cache.built_losses import cce
#
#
# class Lab3_Cross_Entropy:
#     def __init__(self, b1_weight, b2_weight, b3_weight, b4_weight, b5_weight, ignore_index=255, reduction=0):
#         self.weights = [b1_weight, b2_weight, b3_weight, b4_weight, b5_weight]  # 每个尺度的损失的权重
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.ce = cce.build_loss(self.ignore_index, reduction)
#
#     def __call__(self, outputs, targets):
#         """
#         :param outputs: [(b, cls_n, h, w),
#                          (b, cls_n, h, w),
#                          (b, cls_n, h, w),
#                          (b, cls_n, h, w),
#                          (b, cls_n, h, w)]
#         :param targets: (b, 1, h, w)
#         """
#
#         return (
#                 self.ce(outputs[0], targets) * self.weights[0] + \
#                 self.ce(outputs[1], targets) * self.weights[1] + \
#                 self.ce(outputs[2], targets) * self.weights[2] + \
#                 self.ce(outputs[3], targets) * self.weights[3] + \
#                 self.ce(outputs[4], targets) * self.weights[4]
#         )
#
#
# def build_loss(*args):
#     ignore_index = int(args[5])
#     loss = Lab3_Cross_Entropy(args[0], args[1], args[2], args[3], args[4], ignore_index, args[6])
#     return loss
