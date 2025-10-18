# BoundingBox Chart Environment

## Overview

`BoundingBoxChartEnv` is a reinforcement learning environment for training LLM agents to accurately add bounding boxes on chart images. The environment follows the `BaseEnv` interface specification of the VAGEN project.

## Environment Description

### Task Objective
The agent needs to analyze chart images, identify the main plot area, and output a precise bounding box to frame it. The bounding box should exclude peripheral elements such as axis labels and titles.

### Action Space
LLM outputs bounding box coordinates in the format: `BBox[x_min, y_min, x_max, y_max]`
- All coordinates are normalized (range 0-1)
- `(x_min, y_min)`: Top-left corner of the bounding box
- `(x_max, y_max)`: Bottom-right corner of the bounding box
- Coordinate origin `(0, 0)` is at the top-left corner of the image

### Observation Space
- **Vision mode** (`render_mode="vision"`): Multi-modal observation containing chart images
- **Text mode** (`render_mode="text"`): Text description containing only image dimensions

### Reward Logic
Rewards are calculated based on **IoU (Intersection over Union)** between predicted and ground truth bounding boxes:

```python
if iou >= 0.9:
    reward = 1.0    # Excellent
elif iou >= 0.7:
    reward = 0.7    # Good
elif iou >= 0.5:
    reward = 0.5    # Pass
elif iou >= 0.3:
    reward = 0.2    # Poor
else:
    reward = 0.0    # Very Poor

# Invalid output receives negative reward
invalid_output_reward = -0.5
```

**Success criterion**: IoU >= `iou_threshold` (default 0.5)

## File Structure

```
vagen/env/bounding_box/
├── env.py                # BoundingBoxChartEnv - Main environment class
├── env_config.py         # BoundingBoxChartEnvConfig - Configuration class
├── prompt.py             # Prompt templates and format configurations
├── __init__.py           # Environment registration
└── README.md             # This documentation
```

## Usage

### 1. Basic Usage

```python
from vagen.env.bounding_box import BoundingBoxChartEnv, BoundingBoxChartEnvConfig

# Create configuration
config = BoundingBoxChartEnvConfig(
    render_mode="vision",
    prompt_format="free_think",
    iou_threshold=0.5
)

# Create environment
env = BoundingBoxChartEnv(config)

# Reset environment
obs, info = env.reset(seed=42)

# Get system prompt
system_prompt = env.system_prompt()

# Simulate LLM response
llm_response = """<think>
This chart has a title, axes, and a main plot area.
The plot area is approximately from (0.15, 0.1) to (0.9, 0.85)
</think>
<answer>BBox[0.15, 0.1, 0.9, 0.85]</answer>"""

# Execute one step
next_obs, reward, done, info = env.step(llm_response)

# View results
print(f"IoU: {info['iou']:.3f}")
print(f"Reward: {reward:.3f}")
print(f"Success: {info['metrics']['traj_metrics']['success']}")

# Close environment
env.close()
```

### 2. Configuration Options

```python
@dataclass
class BoundingBoxChartEnvConfig(BaseEnvConfig):
    env_name: str = "bounding_box"
    render_mode: str = "vision"           # "vision" or "text"
    max_actions_per_step: int = 1
    prompt_format: str = "free_think"     # "free_think", "grounding", "direct"
    
    # Image settings
    image_width: int = 640
    image_height: int = 480
    
    # Evaluation settings
    normalize_coords: bool = True          # Coordinate normalization
    iou_threshold: float = 0.5             # IoU threshold for success
    
    # Visualization settings
    draw_bbox: bool = True                 # Draw bounding boxes on image
    reward_scale: float = 1.0              # Reward scaling factor
```

### 3. Prompt Formats

The environment supports three prompt formats:

**free_think** (Free thinking):
```
<think>Thinking process...</think>
<answer>BBox[x_min, y_min, x_max, y_max]</answer>
```

**grounding** (Grounding reasoning):
```
<think>
  <observation>Observed features...</observation>
  <reasoning>Reasoning process...</reasoning>
</think>
<answer>BBox[x_min, y_min, x_max, y_max]</answer>
```

**direct** (Direct output):
```
BBox[x_min, y_min, x_max, y_max]
```

## Environment Registration

To register this environment in VAGEN, follow these steps:

### Method 1: Modify `vagen/env/__init__.py`

Add to the `vagen/env/__init__.py` file:

```python
from .bounding_box import BOUNDING_BOX_ENV_INFO

# Add to REGISTERED_ENV dictionary
REGISTERED_ENV = {
    # ... other environments ...
    "bounding_box": BOUNDING_BOX_ENV_INFO,
}
```

### Method 2: Dynamic Registration (if supported)

```python
from vagen.env import register_env
from vagen.env.bounding_box import BOUNDING_BOX_ENV_INFO

register_env("bounding_box", BOUNDING_BOX_ENV_INFO)
```

## Metrics Description

The metrics returned by the environment contain the following fields:

```python
{
    "turn_metrics": {
        "action_is_valid": bool,        # Whether output format is correct
        "action_is_effective": bool,    # Whether bounding box was successfully parsed
    },
    "traj_metrics": {
        "success": bool,                # Whether IoU reaches threshold
        "iou": float,                   # IoU value
        "reward": float,                # Reward received
    }
}
```

## Testing

Run the test script:

```bash
python examples/test_bounding_box_env.py
```

Tests include:
- Basic functionality testing
- Different prompt format testing
- IoU calculation correctness verification
- Image generation testing

## Core Method Implementation

### `__init__(config)`
Initialize the environment, set configuration parameters, prepare prompt templates and parsers.

### `reset(seed=None)`
Reset environment to initial state:
- Generate new chart image
- Generate corresponding ground truth bounding box
- Return initial observation

### `step(llm_raw_response)`
Execute one step of interaction:
1. Parse LLM response and extract bounding box coordinates
2. Validate coordinate validity
3. Calculate IoU and reward
4. Generate annotated observation image
5. Return (observation, reward, done, info)

### `system_prompt()`
Return system prompt, including:
- Task description
- Bounding box format
- Evaluation criteria
- Output format requirements

### `close()`
Clean up resources and release image memory.

### `compute_reward()`
Return additional reward at episode end (0.0 in this environment).

## Extension Suggestions

This environment can be extended based on requirements:

1. **Multi-object detection**: Support annotating multiple regions in one chart
2. **More complex charts**: Add more chart types (scatter plots, pie charts, etc.)
3. **Real data**: Use real chart datasets instead of generated images
4. **Step-by-step annotation**: Allow agents to adjust bounding boxes over multiple steps
5. **Additional tasks**: Simultaneously annotate and classify chart types

## Dependencies

- `numpy`: Numerical computation
- `PIL (Pillow)`: Image processing
- `vagen`: VAGEN framework base components

## Author

Created according to VAGEN environment development guidelines
