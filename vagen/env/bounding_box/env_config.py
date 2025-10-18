"""
BoundingBox Chart Environment Configuration
Defines configuration parameters for the bounding box chart environment
"""
from dataclasses import dataclass
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
    
    # Prompt format type: supports "free_think", "grounding", etc.
    prompt_format: str = "free_think"
    
    # Chart image dimensions
    image_width: int = 640
    image_height: int = 480
    
    # Normalize bounding box coordinates (if True, range is 0-1; otherwise actual pixel values)
    normalize_coords: bool = True
    
    # IoU threshold: used to determine if bounding box is valid
    iou_threshold: float = 0.5
    
    # Reward scaling factor
    reward_scale: float = 1.0
    
    # Whether to draw bounding boxes on the image
    draw_bbox: bool = True
    
    def config_id(self) -> str:
        """Return unique identifier for the configuration"""
        return (f"BoundingBoxChartEnvConfig("
                f"mode={self.render_mode},"
                f"format={self.prompt_format},"
                f"iou_thresh={self.iou_threshold})")

