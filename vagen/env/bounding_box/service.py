"""
BoundingBox Chart Service
Service layer for managing multiple BoundingBox environments in batch operations
Used for efficient parallel processing during training
"""
from typing import Dict, List, Tuple, Optional, Any
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.server.serial import serialize_observation

from .env import BoundingBoxChartEnv
from .env_config import BoundingBoxChartEnvConfig


class BoundingBoxService(BaseService):
    """
    Service for managing multiple BoundingBox Chart environments in batch.
    
    This service provides batch operations for creating, resetting, stepping,
    and closing multiple environment instances efficiently.
    """
    
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize the BoundingBox service
        
        Args:
            config: Service configuration object
        """
        self.environments = {}
        self.env_configs = {}
        self.config = config
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple BoundingBox environments in batch
        
        Args:
            ids2configs: Dictionary mapping environment IDs to their configurations
                         Each config should contain 'env_config' with BoundingBoxChartEnvConfig parameters
        """
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_config = BoundingBoxChartEnvConfig(**env_config_dict)
            env = BoundingBoxChartEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple environments in batch
        
        Args:
            ids2seeds: Dictionary mapping environment IDs to seed values
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        results = {}
        
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Step through multiple environments in batch
        
        Args:
            ids2actions: Dictionary mapping environment IDs to LLM raw response actions
            
        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        results = {}
        
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute final rewards for multiple environments in batch
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to their computed rewards
        """
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple environments in batch
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to their system prompts
        """
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources
        
        Args:
            env_ids: List of environment IDs to close. If None, closes all environments
        """
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            env = self.environments[env_id]
            env.close()
            
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

