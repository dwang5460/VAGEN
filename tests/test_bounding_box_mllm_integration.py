"""
BoundingBox Environment - MLLM Integration Test

This test validates the complete workflow:
1. Accept user-provided chart image
2. Use real MLLM agent to predict bounding box
3. Environment returns chart image with applied bounding box

Requirements:
    - VLLM model (Qwen2-VL-2B-Instruct or similar)
    - GPU with sufficient memory
    - Input chart image

Usage:
    # Basic test with default image
    python tests/test_bounding_box_mllm_integration.py
    
    # Test with custom image
    python tests/test_bounding_box_mllm_integration.py --image_path your_chart.png
    
    # Use different model
    python tests/test_bounding_box_mllm_integration.py --model_name Qwen/Qwen2-VL-7B-Instruct
"""
import sys
import os
import argparse
from pathlib import Path
os.environ["OPENROUTER_API_KEY"] = "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"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
import numpy as np
from vagen.env.bounding_box import BoundingBoxChartEnv, BoundingBoxChartEnvConfig
from vagen.inference.model_interface.factory_model import ModelFactory


class MLLMIntegrationTester:
    """Integration test for BoundingBox environment with real MLLM"""
    
    def __init__(self, verbose=True):
        self.passed = 0
        self.failed = 0
        self.verbose = verbose
        self.test_results = []
    
    def log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def assert_true(self, condition, test_name, error_msg=""):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            result = f"✅ {test_name}"
            self.test_results.append(result)
            self.log(result)
            return True
        else:
            self.failed += 1
            result = f"❌ {test_name}"
            if error_msg:
                result += f": {error_msg}"
            self.test_results.append(result)
            self.log(result)
            return False
    
    def assert_isinstance(self, obj, cls, test_name):
        """Assert object is instance of class"""
        return self.assert_true(
            isinstance(obj, cls),
            test_name,
            f"Expected {cls}, got {type(obj)}"
        )
    
    def assert_in(self, item, container, test_name):
        """Assert item in container"""
        return self.assert_true(
            item in container,
            test_name,
            f"{item} not in {container}"
        )
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print("=" * 80)
        
        if self.failed > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if "❌" in result:
                    print(f"  {result}")


def test_1_environment_setup(tester, image_path):
    """Test 1: Environment setup with user image"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 1: Environment Setup with User Image")
    tester.log("=" * 80)
    
    # Check if image exists
    tester.assert_true(
        os.path.exists(image_path),
        "Input image file exists",
        f"Image not found: {image_path}"
    )
    
    # Load image to verify it's valid
    try:
        img = Image.open(image_path)
        img_valid = True
    except Exception as e:
        img_valid = False
        tester.log(f"   Error loading image: {e}")
    
    tester.assert_true(
        img_valid,
        "Input image is valid and loadable"
    )
    
    if img_valid:
        tester.log(f"   Image size: {img.size}")
        tester.log(f"   Image mode: {img.mode}")
    
    # Create environment configuration
    config = BoundingBoxChartEnvConfig(
        input_image_path=image_path,
        output_image_path="test_output_mllm.png",
        bbox_color="red",
        bbox_width=3,
        prompt_format="free_think"
        # prompt_format="direct"
    )
    
    # Create environment
    try:
        env = BoundingBoxChartEnv(config)
        env_created = True
    except Exception as e:
        env_created = False
        tester.log(f"   Error creating environment: {e}")
    
    tester.assert_true(
        env_created,
        "BoundingBox environment created successfully"
    )
    
    return env if env_created else None


def test_2_reset_and_observation(tester, env):
    """Test 2: Reset environment and get observation"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 2: Reset and Observation")
    tester.log("=" * 80)
    
    if env is None:
        tester.assert_true(False, "Environment is None, skipping test")
        return None, None
    
    # Reset environment
    try:
        observation, info = env.reset()
        reset_success = True
    except Exception as e:
        reset_success = False
        tester.log(f"   Error during reset: {e}")
        return None, None
    
    tester.assert_true(
        reset_success,
        "Environment reset successful"
    )
    
    # Check observation structure
    tester.assert_isinstance(
        observation,
        dict,
        "Observation is a dictionary"
    )
    
    tester.assert_in(
        'obs_str',
        observation,
        "Observation contains 'obs_str' (agent question)"
    )
    
    tester.assert_in(
        'multi_modal_data',
        observation,
        "Observation contains 'multi_modal_data' (chart image)"
    )
    
    # Check multi-modal data contains image
    if 'multi_modal_data' in observation and observation['multi_modal_data']:
        has_image = '<image>' in observation['multi_modal_data']
        tester.assert_true(
            has_image,
            "Multi-modal data contains image"
        )
        
        if has_image:
            image_list = observation['multi_modal_data']['<image>']
            tester.assert_true(
                len(image_list) > 0,
                "Image list is not empty"
            )
            
            if len(image_list) > 0:
                img = image_list[0]
                tester.assert_isinstance(
                    img,
                    Image.Image,
                    "Image is PIL Image object"
                )
                tester.log(f"   Image size: {img.size}")
    
    # Check info structure
    tester.assert_isinstance(
        info,
        dict,
        "Info is a dictionary"
    )
    
    tester.assert_in(
        'image_path',
        info,
        "Info contains image_path"
    )
    
    tester.assert_in(
        'image_size',
        info,
        "Info contains image_size"
    )
    
    tester.log(f"   Agent question preview: {observation['obs_str'][:150]}...")
    
    return observation, info

def test_2_1_multiple_resets(tester, env):
    """Test 2.1: Multiple resets should work consistently"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 2.1: Multiple Resets")
    tester.log("=" * 80)
    
    if env is None:
        tester.assert_true(False, "Environment is None, skipping test")
        return
    
    # First reset
    obs1, info1 = env.reset()
    image_path1 = info1.get('image_path')
    
    # Second reset
    obs2, info2 = env.reset()
    image_path2 = info2.get('image_path')
    
    # Should return same structure
    tester.assert_true(
        'obs_str' in obs1 and 'obs_str' in obs2,
        "Both resets return valid observations"
    )
    
    tester.assert_true(
        image_path1 == image_path2,
        f"Image path consistent across resets"
    )
    
    tester.log(f"   Reset 1 image: {image_path1}")
    tester.log(f"   Reset 2 image: {image_path2}")

def test_2_2_reset_after_step(tester, env):
    """Test 2.2: Reset after step should clear state"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 2.2: Reset After Step")
    tester.log("=" * 80)
    
    if env is None:
        tester.assert_true(False, "Environment is None, skipping test")
        return
    
    # Initial reset
    obs1, info1 = env.reset()
    
    # Take a step
    obs2, reward, done, info2 = env.step("BBox[0.1, 0.1, 0.9, 0.9]")
    
    tester.log(f"   After step - done: {done}, reward: {reward}")
    
    # Reset again
    obs3, info3 = env.reset()
    
    # Should be able to step again after reset
    tester.assert_true(
        'obs_str' in obs3,
        "Reset after step returns valid observation"
    )
    
    # Verify environment state was cleared
    tester.assert_true(
        env.step_count == 0,
        "Step count reset to 0"
    )
    
    tester.assert_true(
        env.total_reward == 0,
        "Total reward reset to 0"
    )

def test_2_3_verify_loaded_image(tester, env, expected_image_path):
    """Test 2.3: Verify correct image was loaded"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 2.3: Verify Loaded Image")
    tester.log("=" * 80)
    
    if env is None:
        tester.assert_true(False, "Environment is None, skipping test")
        return
    
    # Reset and get observation
    obs, info = env.reset()
    
    # Check image path matches expected
    actual_path = info.get('image_path')
    tester.assert_true(
        actual_path == expected_image_path,
        f"Loaded correct image",
        f"Expected: {expected_image_path}, Got: {actual_path}"
    )
    
    # Check image dimensions
    if 'multi_modal_data' in obs and obs['multi_modal_data']:
        img = obs['multi_modal_data']['<image>'][0]
        expected_img = Image.open(expected_image_path)
        
        tester.assert_true(
            img.size == expected_img.size,
            f"Image dimensions match: {img.size}"
        )
        
        tester.log(f"   Loaded image: {actual_path}")
        tester.log(f"   Image size: {img.size}")

def test_2_4_invalid_image_handling(tester):
    """Test 2.4: Handle invalid image paths"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 2.4: Invalid Image Handling")
    tester.log("=" * 80)
    
    # Test with non-existent image
    config = BoundingBoxChartEnvConfig(
        input_image_path="nonexistent_image.png"
    )
    
    try:
        env = BoundingBoxChartEnv(config)
        obs, info = env.reset()
        reset_succeeded = True
    except FileNotFoundError:
        reset_succeeded = False
        tester.log("   Correctly raised FileNotFoundError")
    except Exception as e:
        reset_succeeded = False
        tester.log(f"   Raised exception: {type(e).__name__}: {e}")
    
    tester.assert_true(
        not reset_succeeded,
        "Reset with invalid image should fail appropriately"
    )

def test_3_mllm_inference(tester, env, observation, model_name, model_config):
    """Test 3: MLLM inference to predict bounding box"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 3: MLLM Inference")
    tester.log("=" * 80)
    
    if observation is None:
        tester.assert_true(False, "Observation is None, skipping test")
        return None
    
    # Create MLLM model interface
    tester.log(f"   Initializing MLLM: {model_name}")
    try:
        model_interface = ModelFactory.create(model_config)
        model_created = True
    except Exception as e:
        model_created = False
        tester.log(f"   Error creating model: {e}")
        tester.log(f"   Make sure you have GPU and VLLM installed")
    
    tester.assert_true(
        model_created,
        f"MLLM model interface created ({model_name})"
    )
    
    if not model_created:
        return None
    
    # Get system prompt
    system_prompt = env.system_prompt()
    tester.log(f"   System prompt length: {len(system_prompt)} chars")
    
    # Prepare prompt for MLLM
    image = observation['multi_modal_data']['<image>'][0]
    agent_question = observation['obs_str']
    #NOTE： It was openai format, now it is routerapi format, which is qwen format
    # prompts = [{
    #     "role": "system",
    #     "content": system_prompt
    # }, {
    #     "role": "user",
    #     "content": [
    #         {
    #             "type": "image",
    #             "image": image
    #         },
    #         {
    #             "type": "text",
    #             "text": agent_question
    #         }
    #     ]
    # }]

    # ✅ 修改这部分：使用 Qwen 格式
    prompts = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": f"<image>\n{agent_question}",  # 字符串格式，包含 <image> 占位符
        "multi_modal_data": {                      # 单独的多模态数据
            "<image>": [image]                     # 图像列表
        }
    }]
    
    # Generate response with MLLM
    tester.log("   Generating MLLM response...")
    try:
        responses = model_interface.generate(
            prompts=[prompts],
            max_tokens=model_config.get('max_tokens', 256),
            temperature=model_config.get('temperature', 0.1)
        )
        inference_success = True
    except Exception as e:
        inference_success = False
        tester.log(f"   Error during inference: {e}")
        return None
    
    tester.assert_true(
        inference_success,
        "MLLM inference completed"
    )
    
    if not inference_success or not responses:
        return None
    
    mllm_response = responses[0]["text"]
    
    # NOTE: Add these debug outputs
    tester.log("\n" + "=" * 80)
    tester.log("FULL MLLM RESPONSE:")
    tester.log("=" * 80)
    tester.log(mllm_response)  # 打印完整响应
    tester.log("=" * 80)

    tester.assert_true(
        len(mllm_response) > 0,
        "MLLM response is not empty"
    )
    
    # Check if response contains BBox pattern
    has_bbox = "BBox[" in mllm_response or "bbox[" in mllm_response.lower()
    tester.assert_true(
        has_bbox,
        "MLLM response contains BBox pattern"
    )
    
    tester.log(f"   MLLM response length: {len(mllm_response)} chars")
    tester.log(f"   MLLM response preview: {mllm_response[:200]}...")
    
    return mllm_response


def test_4_step_and_bbox_application(tester, env, mllm_response):
    """Test 4: Apply bounding box and get annotated image"""
    tester.log("\n" + "=" * 80)
    tester.log("TEST 4: Step and BBox Application")
    tester.log("=" * 80)
    
    if mllm_response is None:
        tester.assert_true(False, "MLLM response is None, skipping test")
        return None
    
    # Execute step with MLLM response
    try:
        obs, reward, done, info = env.step(mllm_response)
        step_success = True
    except Exception as e:
        step_success = False
        tester.log(f"   Error during step: {e}")
        return None
    
    tester.assert_true(
        step_success,
        "Environment step executed successfully"
    )
    
    # Check return values
    tester.assert_isinstance(
        obs,
        dict,
        "Step returns observation as dict"
    )
    
    tester.assert_isinstance(
        reward,
        float,
        "Step returns reward as float"
    )
    
    tester.assert_isinstance(
        done,
        bool,
        "Step returns done as bool"
    )
    
    tester.assert_isinstance(
        info,
        dict,
        "Step returns info as dict"
    )
    
    # Check if bbox was successfully parsed
    tester.assert_in(
        'success',
        info,
        "Info contains 'success' flag"
    )
    
    tester.assert_in(
        'predicted_bbox',
        info,
        "Info contains 'predicted_bbox'"
    )
    
    bbox_success = info.get('success', False)
    tester.assert_true(
        bbox_success,
        "Bounding box was successfully parsed and applied"
    )
    
    if bbox_success:
        bbox = info['predicted_bbox']
        tester.log(f"   Predicted BBox: {bbox}")
        
        # Validate bbox coordinates
        if bbox:
            tester.assert_true(
                len(bbox) == 4,
                "BBox has 4 coordinates"
            )
            
            if len(bbox) == 4:
                tester.assert_true(
                    all(0 <= x <= 1 for x in bbox),
                    "BBox coordinates are in [0, 1] range"
                )
                
                tester.assert_true(
                    bbox[0] < bbox[2],
                    "BBox x_min < x_max"
                )
                
                tester.assert_true(
                    bbox[1] < bbox[3],
                    "BBox y_min < y_max"
                )
    
    # Check if output image was saved
    tester.assert_in(
        'output_path',
        info,
        "Info contains 'output_path'"
    )
    
    output_path = info.get('output_path')
    if output_path:
        output_exists = os.path.exists(output_path)
        tester.assert_true(
            output_exists,
            f"Output image saved to {output_path}"
        )
        
        if output_exists:
            # Verify output image is valid
            try:
                output_img = Image.open(output_path)
                tester.log(f"   Output image size: {output_img.size}")
                tester.assert_true(
                    True,
                    "Output image is valid and loadable"
                )
            except Exception as e:
                tester.assert_true(
                    False,
                    "Output image is valid and loadable",
                    f"Error: {e}"
                )
    
    # Check observation contains annotated image
    if 'multi_modal_data' in obs and obs['multi_modal_data']:
        has_image = '<image>' in obs['multi_modal_data']
        if has_image:
            annotated_img = obs['multi_modal_data']['<image>'][0]
            tester.assert_isinstance(
                annotated_img,
                Image.Image,
                "Step observation contains annotated image"
            )
            tester.log(f"   Annotated image size: {annotated_img.size}")
    
    # Check done flag
    tester.assert_true(
        done,
        "Episode is marked as done (single-step environment)"
    )
    
    tester.log(f"   Reward: {reward}")
    
    return info


def create_default_test_chart(output_path="test_chart_mllm.png"):
    """Create a default test chart if no image provided"""
    from examples.create_test_chart import create_test_chart
    create_test_chart(output_path, width=800, height=600)
    return output_path


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="BoundingBox MLLM Integration Test")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input chart image (default: create test chart)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="Qwen/Qwen2.5-VL-3B-Instruct",
        default="qwen/qwen2.5-vl-32b-instruct:free",
        help="MLLM model name (default: Qwen2.5-VL-3B-Instruct)"
    )
    parser.add_argument(
        "--provider",  # 新增参数
        type=str,
        default="routerapi",
        choices=["vllm", "openai", "claude", "gemini", "together", "routerapi"],
        help="Model provider (default: routerapi)"
    )
    # parser.add_argument(
    #     "--tensor_parallel_size",
    #     type=int,
    #     default=1,
    #     help="GPU tensor parallelism (default: 1)"
    # )
    # parser.add_argument(
    #     "--gpu_memory_utilization",
    #     type=float,
    #     default=0.85,
    #     help="GPU memory utilization (default: 0.85)"
    # )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = MLLMIntegrationTester(verbose=not args.quiet)
    
    print("=" * 80)
    print("BOUNDING BOX ENVIRONMENT - MLLM INTEGRATION TEST")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    # print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    
    # Prepare image
    if args.image_path is None:
        tester.log("\nNo image provided, creating default test chart...")
        # image_path = create_default_test_chart()
        image_path = "./tests/data/chart.png"
        tester.log(f"Test chart created: {image_path}")
    else:
        image_path = args.image_path
    
    print(f"Input Image: {image_path}")
    print("=" * 80)
    
    # Model configuration
    # model_config = {
    #     "provider": "vllm",
    #     "model_name": args.model_name,
    #     "tensor_parallel_size": args.tensor_parallel_size,
    #     "gpu_memory_utilization": args.gpu_memory_utilization,
    #     "dtype": "bfloat16",
    #     "temperature": 0.1,
    #     "max_tokens": 256,
    #     "trust_remote_code": True,
    #     "top_p": 0.95,
    #     "top_k": 50,
    #     # Explicitly set to None for newer vLLM versions
    #     # "image_input_type": None,
    # }
    # API-based 模型配置
    model_config = {
        "provider": "routerapi",
        "model_name": args.model_name,
        "temperature": 0.1,
        "max_tokens": 3000,
    }
    
    # 根据不同 provider 添加特定配置
    if args.provider == "routerapi":
        model_config.update({
            "api_key": os.getenv("OPENROUTER_API_KEY")
        })
    # Run tests
    env = test_1_environment_setup(tester, image_path)
    observation, info = test_2_reset_and_observation(tester, env)
    mllm_response = test_3_mllm_inference(tester, env, observation, args.model_name, model_config)
    result_info = test_4_step_and_bbox_application(tester, env, mllm_response)


    test_2_1_multiple_resets(tester, env)
    test_2_2_reset_after_step(tester, env)
    test_2_3_verify_loaded_image(tester, env, image_path)
    test_2_4_invalid_image_handling(tester)
    
    # Cleanup
    if env:
        env.close()
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    return 0 if tester.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

