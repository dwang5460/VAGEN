"""
Environment package for bounding box chart tasks
"""
from .env import BoundingBoxChartEnv
from .env_config import BoundingBoxChartEnvConfig


# Environment registration information
BOUNDING_BOX_ENV_INFO = {
    "env_cls": BoundingBoxChartEnv,
    "config_cls": BoundingBoxChartEnvConfig,
    "description": "Environment for training agents to add bounding boxes on charts"
}


__all__ = [
    'BoundingBoxChartEnv',
    'BoundingBoxChartEnvConfig',
    'BOUNDING_BOX_ENV_INFO',
]

