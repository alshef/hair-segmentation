class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.mean = self.sum / self.count

    def value(self):
        return self.mean
