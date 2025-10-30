"""
BoundingBox Chart Environment Prompts
Defines prompt templates and format configurations for the bounding box chart environment
"""


def system_prompt(**kwargs):
    """System prompt: defines the role and task of the LLM"""
    return """You are a professional chart analysis assistant. Your task is to accurately add bounding boxes to frame important chart elements in chart images.

**Task Objective:**
Analyze the given chart image, locate the main plot area, and output a bounding box to precisely frame it.

**Bounding Box Format:**
A bounding box is defined by four normalized coordinates: [x_min, y_min, x_max, y_max]
- x_min, y_min: Coordinates of the top-left corner of the bounding box (normalized values between 0-1)
- x_max, y_max: Coordinates of the bottom-right corner of the bounding box (normalized values between 0-1)
- The origin (0,0) is at the top-left corner of the image, (1,1) is at the bottom-right corner

**Available Actions:**
- BBox[x_min, y_min, x_max, y_max]: Define a bounding box
  Example: BBox[0.1, 0.2, 0.9, 0.8]

**Important Notes:**
1. Carefully observe the axes, title, and legend positions in the chart
2. The bounding box should tightly frame the main plot area, excluding axis labels and title
3. Ensure coordinate values are within [0, 1] range
4. Coordinate order must be [x_min, y_min, x_max, y_max]
"""


def init_observation_template(observation="", **kwargs):
    """Initial observation prompt template"""
    return f"""[Task Start]
Please analyze the following chart image and output a bounding box to frame the main plot area.

{observation}

Please carefully observe the chart structure, then output your bounding box coordinates.
"""


def observation_template(observation="", **kwargs):
    """Subsequent observation prompt template"""
    return f"""[Feedback]
{observation}

If needed, please adjust your bounding box.
"""


# Configuration for different formats
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>BBox[x_min, y_min, x_max, y_max]</answer>",
        "example": "<think>This is a line chart with axes on the left and bottom. The main plot area occupies roughly the central part of the image, excluding axis labels.</think><answer>BBox[0.15, 0.1, 0.9, 0.85]</answer>",
        "description": "Free thinking mode: first analyze the chart structure, then provide the bounding box"
    },
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>BBox[x_min, y_min, x_max, y_max]</answer>",
        "example": "<think><observation>The chart shows a bar chart with the x-axis at the bottom displaying categories and the y-axis on the left showing values. The chart title is at the top.</observation><reasoning>The plot area should start above the x-axis labels and end to the right of the y-axis labels, leaving space for the title at the top. Estimated area is x: 0.2-0.95, y: 0.15-0.8.</reasoning></think><answer>BBox[0.2, 0.15, 0.95, 0.8]</answer>",
        "description": "Grounding reasoning mode: first observe chart features, then reason, and finally provide the bounding box"
    },
    "direct": {
        "format": "BBox[x_min, y_min, x_max, y_max]",
        "example": "BBox[0.1, 0.15, 0.9, 0.85]",
        "description": "Direct mode: directly output the bounding box without thinking process"
    }
}


def format_prompt_generator(format_type):
    """Generate prompt function for specific format"""
    def prompt_function(**kwargs):
        config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["free_think"])
        return f"""
**Output Format Requirements:**
{config['description']}

Format: {config['format']}

Example: {config['example']}

Please strictly follow this format for your answer.
"""
    return prompt_function


# Generate corresponding prompt functions for each format
format_prompt = {fmt: format_prompt_generator(fmt) for fmt in FORMAT_CONFIGS}


def get_format_instruction(format_type="free_think"):
    """Get format instruction"""
    config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["free_think"])
    return config["format"]

