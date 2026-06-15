class MetricAccumulator:
    def __init__(self, names=("psnr", "ssim")):
        self.count = 0
        self._names = tuple(names)
        self._totals = {name: 0.0 for name in self._names}

    def update(self, **metrics):
        self.count += 1
        for name in self._names:
            self._totals[name] += float(metrics[name])

    def averages(self):
        if self.count == 0:
            return None
        return {name: value / self.count for name, value in self._totals.items()}
