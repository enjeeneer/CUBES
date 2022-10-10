from data.collector import DataCollector

import sys
from utils.utils import Cfg

Cfg = Cfg()
cfg = Cfg.parse(model='dt')

collector = DataCollector(cfg=cfg)
collector.run()
