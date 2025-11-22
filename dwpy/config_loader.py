"""
Configuration file loader for experiment-based workflows.

Supports loading experiment configurations from YAML or JSON files.
"""

import json
from pathlib import Path
from typing import Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .config_schema import ExperimentConfig


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from YAML or JSON file.

    Parameters
    ----------
    config_path : str or Path
        Path to configuration file (.yaml, .yml, or .json)

    Returns
    -------
    ExperimentConfig
        Loaded and validated configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ImportError
        If YAML file specified but PyYAML not installed
    ValueError
        If config file is invalid

    Examples
    --------
    >>> config = load_experiment_config('configs/dapi_60x_oil.yaml')
    >>> print(config.name)
    DAPI 60x Oil Immersion
    >>> print(config.microscope.NA)
    1.45
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Determine file type
    suffix = config_path.suffix.lower()

    if suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(config_path) as f:
            data = yaml.safe_load(f)

    elif suffix == '.json':
        with open(config_path) as f:
            data = json.load(f)

    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            "Use .yaml, .yml, or .json"
        )

    # Validate and create config
    try:
        config = ExperimentConfig.from_dict(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid config file format: {e}") from e

    return config


def save_experiment_config(
    config: ExperimentConfig,
    output_path: Union[str, Path],
    format: str = 'yaml'
):
    """
    Save experiment configuration to file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save
    output_path : str or Path
        Output file path
    format : str
        File format: 'yaml' or 'json' (default: 'yaml')

    Examples
    --------
    >>> config = ExperimentConfig(...)
    >>> save_experiment_config(config, 'my_experiment.yaml')
    """
    output_path = Path(output_path)
    data = config.to_dict()

    if format == 'yaml':
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML output")
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")


__all__ = [
    'load_experiment_config',
    'save_experiment_config',
]
