# experiments/shared/yaml_loader.py

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExperimentConfig:
    """Configuration for a thermostat experiment loaded from YAML."""
    name: str
    description: str
    base: Path
    thermostat_control: str
    always_include: List[Dict[str, str]]
    modes: Dict[str, Dict[str, str]]
    run_periods: Dict[str, str]
    boiler: Dict[str, Dict]
    defaults: Dict[str, str]

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """Load experiment configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Resolve base path relative to YAML file location
        yaml_dir = yaml_path.parent
        base_path = yaml_dir / data['base']

        return cls(
            name=data['name'],
            description=data['description'],
            base=base_path,
            thermostat_control=data['thermostat_control'],
            always_include=data['always_include'],
            modes=data['modes'],
            run_periods=data['run_periods'],
            boiler=data['boiler'],
            defaults=data['defaults']
        )

    def get_run_config(self, mode=None, run_period=None, boiler_part_load=None, boiler_efficiency=None):
        """Get resolved run configuration with defaults applied."""
        return {
            'mode': mode or self.defaults['mode'],
            'run_period': run_period or self.defaults['run_period'],
            'boiler_part_load': boiler_part_load or self.defaults['boiler_part_load'],
            'boiler_efficiency': boiler_efficiency or self.defaults['boiler_efficiency']
        }

    def resolve_file_path(self, relative_path: str, yaml_dir: Path) -> Path:
        """Resolve a file path from YAML relative to the experiment directory."""
        # If it starts with "shared/", resolve from base
        if relative_path.startswith("shared/"):
            return self.base / relative_path.replace("shared/", "", 1)
        else:
            # Otherwise, resolve from YAML directory
            return yaml_dir / relative_path
