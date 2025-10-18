"""
Core Gymnasium Environment for Bounding Box Chart Task
Core Gym environment - implements the standard gymnasium interface
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Dict, Any


class BoundingBoxChartEnv(gym.Env):
    """
    Standard Gymnasium environment for bounding box chart task
    
    ## Task Description
    Given a chart image, the agent needs to output a bounding box to frame the main plot area.
    
    ## Action Space
    Box(4,) - Four normalized coordinate values [x_min, y_min, x_max, y_max], range [0, 1]
    
    ## Observation Space
    Box(height, width, 3) - RGB image
    
    ## Reward
    Based on IoU (Intersection over Union):
    - IoU >= 0.9: +1.0
    - IoU >= 0.7: +0.7
    - IoU >= 0.5: +0.5
    - IoU >= 0.3: +0.2
    - IoU < 0.3: 0.0
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        normalize_coords: bool = True,
        render_mode: Optional[str] = "rgb_array"
    ):
        """
        Initialize the environment
        
        Args:
            image_width: Image width
            image_height: Image height
            normalize_coords: Whether to normalize coordinates
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.image_width = image_width
        self.image_height = image_height
        self.normalize_coords = normalize_coords
        self.render_mode = render_mode
        
        # Define action space: [x_min, y_min, x_max, y_max], normalized to [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Define observation space: RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_height, image_width, 3),
            dtype=np.uint8
        )
        
        # State variables
        self.current_image = None  # PIL Image
        self.ground_truth_bbox = None  # [x_min, y_min, x_max, y_max] normalized
        self.predicted_bbox = None
        self.np_random = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation (numpy array)
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Generate new chart and ground truth bounding box
        self.current_image, self.ground_truth_bbox = self._generate_chart_with_bbox()
        self.predicted_bbox = None
        
        # Convert to numpy array for observation
        obs = np.array(self.current_image, dtype=np.uint8)
        
        info = {
            "ground_truth_bbox": self.ground_truth_bbox.copy()
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step
        
        Args:
            action: Bounding box coordinates [x_min, y_min, x_max, y_max]
            
        Returns:
            observation: Next observation
            reward: Reward
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Validate action format
        if len(action) != 4:
            # Invalid action
            obs = np.array(self.current_image, dtype=np.uint8)
            return obs, -0.5, True, False, {
                "error": "Invalid action shape",
                "iou": 0.0
            }
        
        # Ensure coordinates are within valid range
        x_min, y_min, x_max, y_max = action
        
        if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
            # Invalid coordinates
            obs = np.array(self.current_image, dtype=np.uint8)
            return obs, -0.5, True, False, {
                "error": "Coordinates out of range",
                "iou": 0.0
            }
        
        # Save prediction
        self.predicted_bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
        
        # Calculate IoU
        iou = self._calculate_iou(self.predicted_bbox, self.ground_truth_bbox)
        
        # Calculate reward
        reward = self._compute_reward(iou)
        
        # Generate observation image with bounding boxes
        annotated_image = self._draw_bboxes(
            self.current_image.copy(),
            predicted=self.predicted_bbox,
            ground_truth=self.ground_truth_bbox
        )
        obs = np.array(annotated_image, dtype=np.uint8)
        
        # Bounding box task is a single-step task
        terminated = True
        truncated = False
        
        info = {
            "iou": float(iou),
            "predicted_bbox": self.predicted_bbox,
            "ground_truth_bbox": self.ground_truth_bbox
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state
        
        Returns:
            RGB image array
        """
        if self.render_mode == "rgb_array":
            if self.current_image is None:
                return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            
            # Draw bounding boxes if prediction exists
            if self.predicted_bbox is not None:
                img = self._draw_bboxes(
                    self.current_image.copy(),
                    predicted=self.predicted_bbox,
                    ground_truth=self.ground_truth_bbox
                )
            else:
                img = self.current_image.copy()
            
            return np.array(img, dtype=np.uint8)
        
        return None
    
    def close(self):
        """Clean up resources"""
        self.current_image = None
        self.ground_truth_bbox = None
        self.predicted_bbox = None
    
    # ==================== Helper Methods ====================
    
    def _calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """Calculate IoU"""
        x_min_inter = max(bbox1[0], bbox2[0])
        y_min_inter = max(bbox1[1], bbox2[1])
        x_max_inter = min(bbox1[2], bbox2[2])
        y_max_inter = min(bbox1[3], bbox2[3])
        
        if x_max_inter > x_min_inter and y_max_inter > y_min_inter:
            intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        else:
            intersection = 0.0
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_reward(self, iou: float) -> float:
        """Calculate reward based on IoU"""
        if iou >= 0.9:
            return 1.0
        elif iou >= 0.7:
            return 0.7
        elif iou >= 0.5:
            return 0.5
        elif iou >= 0.3:
            return 0.2
        else:
            return 0.0
    
    def _generate_chart_with_bbox(self) -> Tuple[Image.Image, list]:
        """
        Generate a simulated chart and corresponding ground truth bounding box
        
        Returns:
            image: PIL image
            bbox: Ground truth bounding box [x_min, y_min, x_max, y_max]
        """
        # Create blank image
        img = Image.new('RGB', (self.image_width, self.image_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Define chart component positions (pixels)
        title_height = int(self.image_height * 0.12)
        y_axis_width = int(self.image_width * 0.12)
        x_axis_height = int(self.image_height * 0.12)
        right_margin = int(self.image_width * 0.05)
        
        # Calculate plot area (normalized coordinates)
        plot_x_min = y_axis_width / self.image_width
        plot_y_min = title_height / self.image_height
        plot_x_max = (self.image_width - right_margin) / self.image_width
        plot_y_max = (self.image_height - x_axis_height) / self.image_height
        
        # Add random perturbation (±3%)
        plot_x_min += self.np_random.uniform(-0.03, 0.03)
        plot_y_min += self.np_random.uniform(-0.03, 0.03)
        plot_x_max += self.np_random.uniform(-0.03, 0.03)
        plot_y_max += self.np_random.uniform(-0.03, 0.03)
        
        # Limit range
        plot_x_min = np.clip(plot_x_min, 0.05, 0.25)
        plot_y_min = np.clip(plot_y_min, 0.05, 0.25)
        plot_x_max = np.clip(plot_x_max, 0.75, 0.98)
        plot_y_max = np.clip(plot_y_max, 0.75, 0.95)
        
        # Convert to pixel coordinates for drawing
        plot_x_min_px = int(plot_x_min * self.image_width)
        plot_y_min_px = int(plot_y_min * self.image_height)
        plot_x_max_px = int(plot_x_max * self.image_width)
        plot_y_max_px = int(plot_y_max * self.image_height)
        
        # Draw chart components
        # Title area
        draw.rectangle([0, 0, self.image_width, title_height],
                      fill='lightgray', outline='gray')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((self.image_width // 2, title_height // 2),
                 "Chart Title", fill='black', anchor='mm', font=font)
        
        # Y-axis area
        draw.rectangle([0, title_height, y_axis_width, 
                       self.image_height - x_axis_height],
                      fill='lightblue', outline='gray')
        
        # X-axis area
        draw.rectangle([y_axis_width, self.image_height - x_axis_height,
                       self.image_width, self.image_height],
                      fill='lightgreen', outline='gray')
        
        # Plot area
        draw.rectangle([plot_x_min_px, plot_y_min_px, plot_x_max_px, plot_y_max_px],
                      fill='lightyellow', outline='black', width=2)
        
        # Draw random chart in plot area
        chart_type = self.np_random.choice(['line', 'bar', 'scatter'])
        if chart_type == 'line':
            self._draw_line_chart(draw, plot_x_min_px, plot_y_min_px,
                                 plot_x_max_px, plot_y_max_px)
        elif chart_type == 'bar':
            self._draw_bar_chart(draw, plot_x_min_px, plot_y_min_px,
                                plot_x_max_px, plot_y_max_px)
        else:
            self._draw_scatter_chart(draw, plot_x_min_px, plot_y_min_px,
                                    plot_x_max_px, plot_y_max_px)
        
        bbox = [plot_x_min, plot_y_min, plot_x_max, plot_y_max]
        
        return img, bbox
    
    def _draw_line_chart(self, draw, x_min, y_min, x_max, y_max):
        """Draw line chart"""
        num_points = self.np_random.integers(5, 12)
        x_coords = np.linspace(x_min + 10, x_max - 10, num_points)
        y_coords = self.np_random.uniform(y_min + 10, y_max - 10, num_points)
        
        points = list(zip(x_coords, y_coords))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='blue', width=2)
            draw.ellipse([points[i][0] - 3, points[i][1] - 3,
                         points[i][0] + 3, points[i][1] + 3], fill='red')
    
    def _draw_bar_chart(self, draw, x_min, y_min, x_max, y_max):
        """Draw bar chart"""
        num_bars = self.np_random.integers(4, 8)
        bar_width = (x_max - x_min) / (num_bars * 1.5)
        spacing = bar_width * 0.5
        
        for i in range(num_bars):
            x = x_min + spacing + i * (bar_width + spacing)
            height = self.np_random.uniform(0.3, 0.9) * (y_max - y_min)
            y = y_max - height
            
            draw.rectangle([x, y, x + bar_width, y_max - 5],
                          fill='steelblue', outline='navy')
    
    def _draw_scatter_chart(self, draw, x_min, y_min, x_max, y_max):
        """Draw scatter chart"""
        num_points = self.np_random.integers(15, 30)
        
        for _ in range(num_points):
            x = self.np_random.uniform(x_min + 10, x_max - 10)
            y = self.np_random.uniform(y_min + 10, y_max - 10)
            radius = 3
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        fill='darkgreen', outline='black')
    
    def _draw_bboxes(
        self,
        image: Image.Image,
        predicted: Optional[list] = None,
        ground_truth: Optional[list] = None
    ) -> Image.Image:
        """Draw bounding boxes on image"""
        draw = ImageDraw.Draw(image)
        
        # Draw ground truth bounding box (green)
        if ground_truth is not None:
            x_min = int(ground_truth[0] * self.image_width)
            y_min = int(ground_truth[1] * self.image_height)
            x_max = int(ground_truth[2] * self.image_width)
            y_max = int(ground_truth[3] * self.image_height)
            
            draw.rectangle([x_min, y_min, x_max, y_max],
                          outline='green', width=3)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
            draw.text((x_min, y_min - 18), "Ground Truth", fill='green', font=font)
        
        # Draw predicted bounding box (red)
        if predicted is not None:
            x_min = int(predicted[0] * self.image_width)
            y_min = int(predicted[1] * self.image_height)
            x_max = int(predicted[2] * self.image_width)
            y_max = int(predicted[3] * self.image_height)
            
            draw.rectangle([x_min, y_min, x_max, y_max],
                          outline='red', width=3)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
            draw.text((x_min, y_max + 5), "Predicted", fill='red', font=font)
        
        return image

