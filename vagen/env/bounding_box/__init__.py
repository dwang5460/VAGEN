"""
Environment package for bounding box chart tasks
"""
from .env import BoundingBoxChartEnv
from .env_config import BoundingBoxChartEnvConfig
from .service import BoundingBoxService
from .service_config import BoundingBoxServiceConfig


# Environment registration information
BOUNDING_BOX_ENV_INFO = {
    "env_cls": BoundingBoxChartEnv,
    "config_cls": BoundingBoxChartEnvConfig,
    "service_cls": BoundingBoxService,
    "service_config_cls": BoundingBoxServiceConfig,
    "description": "Environment for training agents to add bounding boxes on charts"
}


__all__ = [
    'BoundingBoxChartEnv',
    'BoundingBoxChartEnvConfig',
    'BoundingBoxService',
    'BoundingBoxServiceConfig',
    'BOUNDING_BOX_ENV_INFO',
]

