from collector import DataCollector

import sys
sys.path.append('../utils')
from utils import Cfg

Cfg = Cfg()
cfg = Cfg.parse(model='dt')

collector = DataCollector(cfg=cfg)
collector.run()
