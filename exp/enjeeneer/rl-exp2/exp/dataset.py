from data.collector import DataCollector


# TODO: remove this file
class Collector:
    def __init__(self, cfg):
        self.collector = DataCollector(cfg=cfg)

    def dataset(self):
        dataset = self.collector.run()

        return dataset

