"""
BoundingBox Chart Environment - VAGEN Wrapper
Reinforcement learning environment for training agents to add bounding boxes on charts
This is a VAGEN wrapper that bridges LLM responses to the core gym environment
"""
import re
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any

from vagen.env.base.base_env import BaseEnv
from .env_config import BoundingBoxChartEnvConfig
from . import prompt as prompt_module
from .bounding_box import BoundingBoxChartEnv as GymBoundingBoxEnv


class BoundingBoxChartEnv(BaseEnv):
    """
    Bounding Box Chart Environment
    
    Task: LLM needs to analyze chart images and output a bounding box to frame the main plot area
    
    Action Space: LLM outputs bounding box coordinates [x_min, y_min, x_max, y_max], normalized to [0, 1]
    Observation Space: Contains chart images and text descriptions
    Reward: Based on IoU (Intersection over Union) between predicted and ground truth bounding boxes
    """
    
    def __init__(self, config: BoundingBoxChartEnvConfig):
        """
        Initialize the bounding box chart environment
        
        Args:
            config: Environment configuration object
        """
        BaseEnv.__init__(self)
        self.config = config
        
        # Initialize core gym environment
        self.gym_env = GymBoundingBoxEnv(
            image_width=config.image_width,
            image_height=config.image_height,
            normalize_coords=config.normalize_coords,
            render_mode="rgb_array"
        )
        
        # Store episode state
        self.step_count = 0
        self.total_reward = 0
        
        # Get prompt template functions
        self.system_prompt_func = prompt_module.system_prompt
        self.init_obs_template = prompt_module.init_observation_template
        self.obs_template = prompt_module.observation_template
        self.format_prompt_func = prompt_module.format_prompt.get(
            config.prompt_format, 
            prompt_module.format_prompt["free_think"]
        )
        
        # Regular expression for parsing bounding boxes
        # Matches format: BBox[x_min, y_min, x_max, y_max]
        self.bbox_pattern = re.compile(
            r'BBox\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]',
            re.IGNORECASE
        )
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and start a new episode
        
        Args:
            seed: Random seed for generating reproducible charts and bounding boxes
            
        Returns:
            observation: Initial observation (containing chart image)
            info: Additional information dictionary
        """
        # Reset gym environment
        gym_obs, gym_info = self.gym_env.reset(seed=seed)
        
        # Reset state
        self.step_count = 0
        self.total_reward = 0
        
        # Generate initial observation
        observation = self._create_observation(gym_obs, is_initial=True)
        
        info = gym_info.copy()
        
        return observation, info
    
    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step of environment interaction
        
        Args:
            llm_raw_response: LLM's raw text response
            
        Returns:
            observation: Next observation
            reward: Reward value
            done: Whether the episode is complete
            info: Dictionary containing metrics and other information
        """
        self.step_count += 1
        
        # 1. Parse LLM response and extract bounding box coordinates
        parsed_result = self._parse_llm_response(llm_raw_response)
        bbox = parsed_result.get('bbox', None)
        is_valid = parsed_result.get('is_valid', False)
        
        # 2. Execute action
        if is_valid and bbox is not None:
            # Convert bounding box to numpy array
            action = np.array(bbox, dtype=np.float32)
            # Execute in gym environment
            gym_obs, reward, terminated, truncated, gym_info = self.gym_env.step(action)
            done = terminated or truncated
            iou = gym_info.get('iou', 0.0)
            is_effective = True
            success = iou >= self.config.iou_threshold
        else:
            # Invalid output, create a dummy observation
            gym_obs = self.gym_env.render()
            reward = -0.5
            done = True
            iou = 0.0
            is_effective = False
            success = False
            gym_info = {"iou": 0.0, "error": parsed_result.get('error', 'Invalid output')}
        
        # Apply reward scaling
        reward = reward * self.config.reward_scale
        self.total_reward += reward
        
        # 3. Create metrics
        metrics = {
            "turn_metrics": {
                "action_is_valid": is_valid,
                "action_is_effective": is_effective,
            },
            "traj_metrics": {
                "success": success,
                "iou": float(iou),
                "reward": float(reward),
            }
        }
        
        # 4. Generate next observation
        observation = self._create_observation(
            gym_obs,
            is_initial=False, 
            feedback=f"IoU: {iou:.3f}, Reward: {reward:.3f}"
        )
        
        # 5. Build info dictionary
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response,
            "llm_response": parsed_result,
        }
        info.update(gym_info)
        
        return observation, reward, done, info
    
    def system_prompt(self) -> str:
        """
        Return the system prompt
        
        Returns:
            System prompt string that tells the LLM its task and rules
        """
        base_prompt = self.system_prompt_func()
        format_instruction = self.format_prompt_func()
        return base_prompt + "\n\n" + format_instruction
    
    def close(self):
        """Clean up resources"""
        self.gym_env.close()
    
    def compute_reward(self) -> float:
        """
        Calculate additional reward at episode end
        
        Returns:
            Additional reward value (usually 0, as reward is already calculated in step)
        """
        return 0.0
    
    # ==================== Helper Methods ====================
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract bounding box coordinates
        
        Args:
            response: LLM's raw response text
            
        Returns:
            Dictionary containing bbox and is_valid
        """
        # Find BBox[x_min, y_min, x_max, y_max] format
        match = self.bbox_pattern.search(response)
        
        if match:
            try:
                x_min = float(match.group(1))
                y_min = float(match.group(2))
                x_max = float(match.group(3))
                y_max = float(match.group(4))
                
                # Validate coordinate validity
                if (0 <= x_min < x_max <= 1 and 
                    0 <= y_min < y_max <= 1):
                    return {
                        'bbox': [x_min, y_min, x_max, y_max],
                        'is_valid': True,
                        'raw_match': match.group(0)
                    }
                else:
                    return {
                        'bbox': None,
                        'is_valid': False,
                        'error': 'Coordinates out of range or invalid order'
                    }
            except (ValueError, IndexError) as e:
                return {
                    'bbox': None,
                    'is_valid': False,
                    'error': f'Parse error: {str(e)}'
                }
        else:
            return {
                'bbox': None,
                'is_valid': False,
                'error': 'No BBox pattern found in response'
            }
    
    def _create_observation(self, gym_obs: np.ndarray, is_initial: bool = True, 
                          feedback: str = "") -> Dict:
        """
        Create observation dictionary, converting gym observation to VAGEN format
        
        Args:
            gym_obs: Observation returned by gym environment (numpy array)
            is_initial: Whether this is the initial observation
            feedback: Feedback information
            
        Returns:
            Observation dictionary containing obs_str and multi_modal_data
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(gym_obs)
        
        if self.config.render_mode == 'vision':
            if is_initial:
                obs_str = self.init_obs_template(
                    observation="Please view the image <image> and analyze its structure."
                )
            else:
                obs_str = self.obs_template(
                    observation=f"<image>\n{feedback}"
                )
            
            return {
                'obs_str': obs_str,
                'multi_modal_data': {
                    '<image>': [pil_image]
                }
            }
        else:
            # Text mode: describe image dimensions and task
            text_desc = f"Image dimensions: {self.config.image_width}x{self.config.image_height}"
            if not is_initial:
                text_desc += f"\n{feedback}"
            
            return {
                'obs_str': text_desc,
                'multi_modal_data': None
            }

