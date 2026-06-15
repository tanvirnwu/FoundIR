class MetricAccumulator:
    def __init__(self):
        self.count = 0
        self._totals = {"psnr": 0.0, "ssim": 0.0}

    def update(self, psnr, ssim):
        self.count += 1
        self._totals["psnr"] += float(psnr)
        self._totals["ssim"] += float(ssim)

    def averages(self):
        if self.count == 0:
            return None
        return {name: value / self.count for name, value in self._totals.items()}
