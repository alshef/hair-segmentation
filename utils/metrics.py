import torch


class IoUMetric:
    def __init__(self, activation=torch.sigmoid, threshold: float = 0.5, SMOOTH: float = 1e-6) -> None:
        self.thresh = threshold
        self.activation = activation
        self.SMOOTH = SMOOTH

    def compute(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        n = logits.size(0)
        pred_mask = self.activation(logits)

        y_pred = (pred_mask.view(n, -1, 1) > self.thresh).byte()
        y = masks.byte().view(n, -1, 1)

        tp_mask = y_pred * y == 1
        tn_mask = y_pred + y == 0
        fp_mask = y_pred - y == 1
        fn_mask = y - y_pred == 1

        tp = torch.sum(tp_mask, dim=[1]).float()
        tn = torch.sum(tn_mask, dim=[1]).float()
        fp = torch.sum(fp_mask, dim=[1]).float()
        fn = torch.sum(fn_mask, dim=[1]).float()

        iou = (tp + self.SMOOTH) / (tp + fp + fn + self.SMOOTH)

        return iou.mean()



