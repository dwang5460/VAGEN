"""
BoundingBox Chart Environment - VAGEN Wrapper
VAGEN wrapper for bounding box chart task that bridges LLM responses to the gym environment
Supports both synthetic chart generation (training) and user-provided images (inference)
"""
import re
import os
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from .env_config import BoundingBoxChartEnvConfig
from . import prompt as prompt_module
from .bounding_box import BoundingBoxChartEnv as GymBoundingBoxEnv


class BoundingBoxChartEnv(BaseEnv):
    """
    Bounding Box Chart Environment - VAGEN Wrapper
    
    This is the VAGEN wrapper that handles LLM interactions for the bounding box chart task.
    It uses the underlying GymBoundingBoxEnv for core game logic.
    
    Workflow:
    1. reset() - Initialize environment (load/generate chart image)
    2. step() - Parse LLM response, execute action in gym env, return observation
    
    Action Space: LLM outputs bounding box coordinates [x_min, y_min, x_max, y_max], normalized to [0, 1]
    Observation Space: Contains chart images (synthetic or user-provided)
    Reward: Based on IoU for synthetic charts, 0.0 for user images
    """
    
    def __init__(self, config: BoundingBoxChartEnvConfig):
        """
        Initialize the bounding box chart environment
        
        Args:
            config: Environment configuration object
        """
        BaseEnv.__init__(self)
        self.config = config
        
        # Initialize the underlying Gymnasium environment
        # For inference mode with user images, use use_synthetic=False
        use_synthetic = config.input_image_path is None
        self.gym_env = GymBoundingBoxEnv(
            image_width=640,
            image_height=480,
            normalize_coords=config.normalize_coords,
            render_mode="rgb_array",
            use_synthetic=use_synthetic
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
        
    def reset(self, seed: Optional[int] = None, image_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and load a new chart image
        
        Args:
            seed: Random seed for synthetic chart generation
            image_path: Path to user-provided chart image. If None, uses config.input_image_path
            
        Returns:
            observation: Dict containing chart image and prompt
            info: Dict with image metadata
        """
        # Determine image path
        if image_path is None:
            image_path = self.config.input_image_path
        
        # Reset the gym environment
        with NoLoggerWarnings():
            with set_seed(seed):
                if image_path:
                    # Load user-provided image
                    gym_obs, gym_info = self.gym_env.reset(seed=seed, options={'image_path': image_path})
                else:
                    # Generate synthetic chart
                    gym_obs, gym_info = self.gym_env.reset(seed=seed)
        
        # Reset episode state
        self.step_count = 0
        self.total_reward = 0
        
        # Create VAGEN-formatted observation
        observation = self._render(gym_obs, is_initial=True)
        
        info = {
            "image_path": gym_info.get("image_path"),
            "image_size": gym_info.get("image_size"),
            "ground_truth_bbox": gym_info.get("ground_truth_bbox")
        }
        
        return observation, info
    
    def step(self, llm_raw_response: str, output_path: Optional[str] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step: parse LLM response and apply bounding box
        
        Args:
            llm_raw_response: LLM's raw text response containing bbox
            output_path: Path to save annotated image (optional)
            
        Returns:
            observation: Dict containing chart image with bbox drawn
            reward: Reward value (IoU-based for synthetic, 0.0 for user images)
            done: Always True (single-step task)
            info: Dict with metrics and bbox information
        """
        self.step_count += 1
        
        # Parse LLM response to extract bounding box coordinates
        parsed_result = self._parse_llm_response(llm_raw_response)
        bbox = parsed_result.get('bbox', None)
        is_valid = parsed_result.get('is_valid', False)
        
        # Initialize metrics
        metrics = {
            "turn_metrics": {
                "action_is_valid": is_valid,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
                "bbox": bbox if bbox else None,
            }
        }
        
        reward = 0.0
        done = False
        info = {"llm_raw_response": llm_raw_response, "llm_response": parsed_result}
        
        # Execute action in gym environment if bbox is valid
        if is_valid and bbox is not None:
            try:
                # Convert bbox list to numpy array for gym env
                action_array = np.array(bbox, dtype=np.float32)
                
                # Step in gym environment
                gym_obs, gym_reward, terminated, truncated, gym_info = self.gym_env.step(action_array)
                
                reward = gym_reward
                done = terminated or truncated
                metrics["turn_metrics"]["action_is_effective"] = True
                
                # Success if reward is positive (good IoU for synthetic charts)
                if reward > 0:
                    metrics["traj_metrics"]["success"] = True
                
                # Update info with gym info
                info.update(gym_info)
                
                # Save annotated image if requested
                if output_path is not False and self.config.draw_bbox:
                    save_path = self._save_annotated_image(gym_obs, output_path)
                    info["output_path"] = save_path
                
            except Exception as e:
                # Handle execution errors
                metrics["turn_metrics"]["action_is_valid"] = False
                reward = -0.5
                done = True
                info["error"] = str(e)
        else:
            # Invalid action - small penalty
            reward = -0.5 if self.gym_env.use_synthetic else 0.0
            done = True
            info["error"] = parsed_result.get('error', 'Invalid bbox format')
        
        # Add metrics to info
        info["metrics"] = metrics
        info["predicted_bbox"] = bbox
        info["success"] = metrics["traj_metrics"]["success"]
        
        self.total_reward += reward
        
        # Render observation
        gym_obs = self.gym_env.render()
        observation = self._render(gym_obs, is_initial=False)
        
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
        In this environment, all rewards are calculated in step(), so return 0.0
        
        Returns:
            0.0
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
    
    def _render(self, gym_obs: np.ndarray, is_initial: bool = True) -> Dict:
        """
        Render the environment observation in VAGEN format
        
        Args:
            gym_obs: Numpy array from gym environment
            is_initial: Whether this is the initial observation
            
        Returns:
            Observation dictionary containing obs_str and multi_modal_data
        """
        multi_modal_data = None
        
        # Get format prompt
        format_prompt_text = self.format_prompt_func()
        
        if self.config.render_mode == 'vision':
            # Convert numpy image to PIL Image
            img_placeholder = self.config.image_placeholder if hasattr(self.config, 'image_placeholder') else '<image>'
            pil_image = convert_numpy_to_PIL(gym_obs)
            
            multi_modal_data = {
                img_placeholder: [pil_image]
            }
            
            if is_initial:
                observation_text = "Please view the image " + img_placeholder + " and analyze its structure."
                obs_str = self.init_obs_template(observation=observation_text)
            else:
                obs_str = self.obs_template(observation=img_placeholder)
            
            obs_str += "\n" + format_prompt_text
            
            return {
                'obs_str': obs_str,
                'multi_modal_data': multi_modal_data
            }
        else:
            # Text mode: describe image dimensions
            height, width = gym_obs.shape[:2]
            text_desc = f"Image dimensions: {width}x{height}"
            
            if is_initial:
                obs_str = self.init_obs_template(observation=text_desc)
            else:
                obs_str = self.obs_template(observation=text_desc)
            
            obs_str += "\n" + format_prompt_text
            
            return {
                'obs_str': obs_str
            }
    
    def _save_annotated_image(self, image: np.ndarray, output_path: Optional[str] = None) -> str:
        """
        Save annotated image to file
        
        Args:
            image: Numpy array image to save
            output_path: Output file path. If None, uses config.output_image_path
            
        Returns:
            Path where image was saved
        """
        # Determine output path
        if output_path is None:
            output_path = self.config.output_image_path
        
        if output_path is None:
            # Generate default output path
            output_path = "output_with_bbox.png"
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy array to PIL Image and save
        pil_image = convert_numpy_to_PIL(image)
        pil_image.save(output_path)
        print(f"✓ Annotated image saved to: {output_path}")
        
        return output_path

