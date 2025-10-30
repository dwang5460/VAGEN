"""
BoundingBox Chart Environment Configuration
Defines configuration parameters for the bounding box chart environment
"""
from dataclasses import dataclass
from typing import Optional
from vagen.env.base.base_env_config import BaseEnvConfig


@dataclass
class BoundingBoxChartEnvConfig(BaseEnvConfig):
    """Configuration class for bounding box chart environment"""
    
    # Basic environment information
    env_name: str = "bounding_box"
    
    # Render mode: "vision" uses images, "text" uses text descriptions
    render_mode: str = "vision"
    
    # Maximum actions per step (bounding box environment only needs one bbox per step)
    max_actions_per_step: int = 1
    
    # Prompt format type: supports "free_think", "grounding", "direct"
    prompt_format: str = "free_think"
    
    # Input/Output settings
    input_image_path: Optional[str] = "./tests/data/chart.png"
    """Path to input chart image.
    
    Default uses a test image. Can be overridden with full or relative path.
    """

    output_image_path: Optional[str] = "./tests/data/output_with_bbox.png"
    """Where to save annotated image with predicted bounding box.
    
    Overwrite for custom output locations. Supports absolute/relative paths.
    """
    
    # Normalize bounding box coordinates (if True, range is 0-1; otherwise actual pixel values)
    normalize_coords: bool = True
    
    # Visualization settings
    draw_bbox: bool = True                # Whether to draw bounding box on image
    bbox_color: str = "red"               # Color for bounding box
    bbox_width: int = 3                   # Line width for bounding box
    
    def config_id(self) -> str:
        """Return unique identifier for the configuration"""
        return (f"BoundingBoxChartEnvConfig("
                f"mode={self.render_mode},"
                f"format={self.prompt_format})")

