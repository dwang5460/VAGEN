"""
BoundingBox Chart Service Configuration
Configuration for the BoundingBox service that manages multiple environments
"""
from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass


@dataclass
class BoundingBoxServiceConfig(BaseServiceConfig):
    """Configuration for BoundingBoxService"""
    
    # Inherits max_workers from BaseServiceConfig
    # Can add BoundingBox-specific service-level configuration here if needed
    pass

