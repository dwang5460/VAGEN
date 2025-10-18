"""
BoundingBox Chart Environment - Unit Tests
Focused on testing environment input/output correctness

Test scope:
1. reset() method output format
2. step() method various input situations and corresponding outputs
3. system_prompt() method output
4. Boundary conditions and error handling

Run:
    python examples/test_bounding_box_env_unit.py
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from vagen.env.bounding_box.env import BoundingBoxChartEnv
from vagen.env.bounding_box.env_config import BoundingBoxChartEnvConfig


class TestBoundingBoxEnv:
    """Unit tests for BoundingBox environment"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []
    
    def assert_equal(self, actual, expected, test_name):
        """Assert equality"""
        if actual == expected:
            self.passed += 1
            self.test_results.append(f"✅ {test_name}")
            return True
        else:
            self.failed += 1
            self.test_results.append(f"❌ {test_name}: Expected {expected}, got {actual}")
            return False
    
    def assert_true(self, condition, test_name):
        """Assert true"""
        if condition:
            self.passed += 1
            self.test_results.append(f"✅ {test_name}")
            return True
        else:
            self.failed += 1
            self.test_results.append(f"❌ {test_name}: Condition is false")
            return False
    
    def assert_in(self, item, container, test_name):
        """Assert contains"""
        if item in container:
            self.passed += 1
            self.test_results.append(f"✅ {test_name}")
            return True
        else:
            self.failed += 1
            self.test_results.append(f"❌ {test_name}: {item} not in {container}")
            return False
    
    def assert_isinstance(self, obj, cls, test_name):
        """Assert type"""
        if isinstance(obj, cls):
            self.passed += 1
            self.test_results.append(f"✅ {test_name}")
            return True
        else:
            self.failed += 1
            self.test_results.append(f"❌ {test_name}: {type(obj)} is not {cls}")
            return False


def test_reset_output_format(tester):
    """Test reset() method output format"""
    print("\n" + "=" * 70)
    print("Test 1: reset() Output Format")
    print("=" * 70)
    
    config = BoundingBoxChartEnvConfig(
        render_mode="vision",
        image_width=640,
        image_height=480
    )
    env = BoundingBoxChartEnv(config)
    
    # Test reset return values
    obs, info = env.reset(seed=42)
    
    # 1. Check return value types
    tester.assert_isinstance(obs, dict, "reset() returns obs as dict")
    tester.assert_isinstance(info, dict, "reset() returns info as dict")
    
    # 2. Check observation dictionary keys
    tester.assert_in('obs_str', obs, "obs contains 'obs_str' key")
    tester.assert_in('multi_modal_data', obs, "obs contains 'multi_modal_data' key")
    
    # 3. Check observation string type
    tester.assert_isinstance(obs['obs_str'], str, "obs_str is string")
    tester.assert_true(len(obs['obs_str']) > 0, "obs_str is not empty")
    tester.assert_in('<image>', obs['obs_str'], "obs_str contains <image> placeholder")
    
    # 4. Check multi-modal data
    if obs['multi_modal_data'] is not None:
        tester.assert_isinstance(obs['multi_modal_data'], dict, "multi_modal_data is dict")
        tester.assert_in('<image>', obs['multi_modal_data'], "multi_modal_data contains '<image>' key")
        
        images = obs['multi_modal_data']['<image>']
        tester.assert_isinstance(images, list, "Images is list")
        tester.assert_true(len(images) > 0, "Images list is not empty")
        
        # Check image placeholder count matches
        image_count_in_str = obs['obs_str'].count('<image>')
        tester.assert_equal(len(images), image_count_in_str, 
                          "Image count matches <image> count in obs_str")
    
    # 5. Check info content
    tester.assert_in('ground_truth_bbox', info, "info contains 'ground_truth_bbox'")
    
    gt_bbox = info['ground_truth_bbox']
    tester.assert_isinstance(gt_bbox, list, "ground_truth_bbox is list")
    tester.assert_equal(len(gt_bbox), 4, "ground_truth_bbox has 4 elements")
    
    # 6. Check bounding box coordinate range
    x_min, y_min, x_max, y_max = gt_bbox
    tester.assert_true(0 <= x_min < 1, "x_min in [0, 1) range")
    tester.assert_true(0 <= y_min < 1, "y_min in [0, 1) range")
    tester.assert_true(0 < x_max <= 1, "x_max in (0, 1] range")
    tester.assert_true(0 < y_max <= 1, "y_max in (0, 1] range")
    tester.assert_true(x_min < x_max, "x_min < x_max")
    tester.assert_true(y_min < y_max, "y_min < y_max")
    
    # 7. Test reproducibility (same seed should give same result)
    obs2, info2 = env.reset(seed=42)
    tester.assert_equal(info['ground_truth_bbox'], info2['ground_truth_bbox'],
                       "Same seed generates same ground_truth_bbox")
    
    env.close()
    print(f"\nSubtotal: {tester.passed - len([r for r in tester.test_results if '❌' in r])} passed")


def test_step_valid_inputs(tester):
    """Test step() method with valid inputs"""
    print("\n" + "=" * 70)
    print("Test 2: step() Valid Inputs")
    print("=" * 70)
    
    config = BoundingBoxChartEnvConfig()
    env = BoundingBoxChartEnv(config)
    
    # === Test 2.1: Standard format input ===
    print("\n2.1 Standard BBox format")
    obs, info = env.reset(seed=42)
    llm_response = "<think>Analysis</think><answer>BBox[0.1, 0.2, 0.9, 0.8]</answer>"
    
    next_obs, reward, done, step_info = env.step(llm_response)
    
    # Check return value types
    tester.assert_isinstance(next_obs, dict, "step() returns obs as dict")
    tester.assert_isinstance(reward, (int, float), "step() returns reward as number")
    tester.assert_isinstance(done, bool, "step() returns done as boolean")
    tester.assert_isinstance(step_info, dict, "step() returns info as dict")
    
    # Check info structure
    tester.assert_in('metrics', step_info, "info contains 'metrics'")
    tester.assert_in('llm_raw_response', step_info, "info contains 'llm_raw_response'")
    tester.assert_in('llm_response', step_info, "info contains 'llm_response'")
    tester.assert_in('iou', step_info, "info contains 'iou'")
    
    # Check metrics structure
    metrics = step_info['metrics']
    tester.assert_in('turn_metrics', metrics, "metrics contains 'turn_metrics'")
    tester.assert_in('traj_metrics', metrics, "metrics contains 'traj_metrics'")
    
    turn_metrics = metrics['turn_metrics']
    tester.assert_in('action_is_valid', turn_metrics, "turn_metrics contains 'action_is_valid'")
    tester.assert_in('action_is_effective', turn_metrics, "turn_metrics contains 'action_is_effective'")
    
    traj_metrics = metrics['traj_metrics']
    tester.assert_in('success', traj_metrics, "traj_metrics contains 'success'")
    tester.assert_in('iou', traj_metrics, "traj_metrics contains 'iou'")
    
    # Check valid action flags
    tester.assert_equal(turn_metrics['action_is_valid'], True, "Valid format marked as valid")
    tester.assert_equal(turn_metrics['action_is_effective'], True, "Valid action marked as effective")
    
    # === Test 2.2: Format with different spacing ===
    print("\n2.2 BBox format with spaces")
    env.reset(seed=123)
    llm_response = "<answer>BBox[ 0.15 , 0.25 , 0.85 , 0.75 ]</answer>"
    _, _, _, step_info = env.step(llm_response)
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Format with spaces still valid")
    
    # === Test 2.3: Direct output format (no think tags) ===
    print("\n2.3 Direct output format")
    env.reset(seed=456)
    llm_response = "BBox[0.2, 0.3, 0.8, 0.7]"
    _, _, _, step_info = env.step(llm_response)
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Direct BBox format valid")
    
    # === Test 2.4: Perfect prediction (IoU = 1.0) ===
    print("\n2.4 Perfect prediction")
    obs, info = env.reset(seed=789)
    gt = info['ground_truth_bbox']
    llm_response = f"<answer>BBox[{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}, {gt[3]:.3f}]</answer>"
    _, reward, _, step_info = env.step(llm_response)
    
    tester.assert_true(step_info['iou'] > 0.99, "Perfect prediction IoU close to 1.0")
    tester.assert_true(reward > 0.9, "Perfect prediction reward > 0.9")
    tester.assert_equal(step_info['metrics']['traj_metrics']['success'], True,
                       "Perfect prediction marked as success")
    
    env.close()
    print(f"\nSubtotal: {tester.passed - sum(1 for r in tester.test_results[:-20] if '✅' in r)} new tests passed")


def test_step_invalid_inputs(tester):
    """Test step() method with invalid inputs"""
    print("\n" + "=" * 70)
    print("Test 3: step() Invalid Inputs")
    print("=" * 70)
    
    config = BoundingBoxChartEnvConfig()
    env = BoundingBoxChartEnv(config)
    
    # === Test 3.1: Empty string ===
    print("\n3.1 Empty string input")
    env.reset(seed=42)
    _, reward, _, step_info = env.step("")
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Empty string marked as invalid")
    tester.assert_true(reward < 0, "Invalid input gets negative reward")
    
    # === Test 3.2: No BBox format ===
    print("\n3.2 Missing BBox keyword")
    env.reset(seed=123)
    _, reward, _, step_info = env.step("<answer>[0.1, 0.2, 0.9, 0.8]</answer>")
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Missing BBox keyword marked as invalid")
    tester.assert_true(reward < 0, "Invalid input gets negative reward")
    
    # === Test 3.3: Wrong coordinate count ===
    print("\n3.3 Wrong coordinate count")
    env.reset(seed=456)
    _, _, _, step_info = env.step("BBox[0.1, 0.2, 0.9]")  # Only 3 coordinates
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Wrong coordinate count marked as invalid")
    
    # === Test 3.4: Coordinates out of bounds ===
    print("\n3.4 Coordinates out of bounds")
    env.reset(seed=789)
    _, _, _, step_info = env.step("BBox[0.5, 0.5, 1.5, 1.5]")  # x_max, y_max > 1
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Out of bounds coordinates marked as invalid")
    tester.assert_in('error', step_info['llm_response'],
                    "Invalid input contains error message")
    
    # === Test 3.5: Wrong coordinate order ===
    print("\n3.5 Wrong coordinate order")
    env.reset(seed=111)
    _, _, _, step_info = env.step("BBox[0.9, 0.8, 0.1, 0.2]")  # x_min > x_max
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Wrong coordinate order marked as invalid")
    
    # === Test 3.6: Non-numeric coordinates ===
    print("\n3.6 Non-numeric coordinates")
    env.reset(seed=222)
    _, _, _, step_info = env.step("BBox[a, b, c, d]")
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Non-numeric coordinates marked as invalid")
    
    # === Test 3.7: Special characters ===
    print("\n3.7 Special character input")
    env.reset(seed=333)
    _, _, _, step_info = env.step("!@#$%^&*()")
    
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], False,
                       "Special character input marked as invalid")
    
    # === Test 3.8: Multiple BBoxes (should parse first one) ===
    print("\n3.8 Multiple BBoxes")
    env.reset(seed=444)
    _, _, _, step_info = env.step("BBox[0.1, 0.2, 0.9, 0.8] BBox[0.2, 0.3, 0.8, 0.7]")
    
    # Should parse the first one
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Multiple BBoxes parses first one")
    
    env.close()
    print(f"\nSubtotal: New tests passed")


def test_step_boundary_conditions(tester):
    """Test step() method boundary conditions"""
    print("\n" + "=" * 70)
    print("Test 4: step() Boundary Conditions")
    print("=" * 70)
    
    config = BoundingBoxChartEnvConfig()
    env = BoundingBoxChartEnv(config)
    
    # === Test 4.1: Minimum bounding box ===
    print("\n4.1 Minimum bounding box (0, 0, 0.01, 0.01)")
    env.reset(seed=42)
    _, _, _, step_info = env.step("BBox[0, 0, 0.01, 0.01]")
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Minimum bounding box valid")
    
    # === Test 4.2: Maximum bounding box ===
    print("\n4.2 Maximum bounding box (0, 0, 1, 1)")
    env.reset(seed=123)
    _, _, _, step_info = env.step("BBox[0, 0, 1, 1]")
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Maximum bounding box valid")
    
    # === Test 4.3: Boundary value 0 ===
    print("\n4.3 Boundary value containing 0")
    env.reset(seed=456)
    _, _, _, step_info = env.step("BBox[0, 0.1, 0.5, 0.9]")
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Coordinates containing 0 valid")
    
    # === Test 4.4: Boundary value 1 ===
    print("\n4.4 Boundary value containing 1")
    env.reset(seed=789)
    _, _, _, step_info = env.step("BBox[0.1, 0.1, 1, 1]")
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Coordinates containing 1 valid")
    
    # === Test 4.5: Very small difference ===
    print("\n4.5 Very small difference")
    env.reset(seed=111)
    _, _, _, step_info = env.step("BBox[0.5, 0.5, 0.500001, 0.500001]")
    tester.assert_equal(step_info['metrics']['turn_metrics']['action_is_valid'], True,
                       "Very small difference valid")
    
    env.close()
    print(f"\nSubtotal: Boundary condition tests complete")


def test_system_prompt_output(tester):
    """Test system_prompt() output"""
    print("\n" + "=" * 70)
    print("Test 5: system_prompt() Output")
    print("=" * 70)
    
    # === Test 5.1: Basic output ===
    config = BoundingBoxChartEnvConfig()
    env = BoundingBoxChartEnv(config)
    
    prompt = env.system_prompt()
    
    tester.assert_isinstance(prompt, str, "system_prompt() returns string")
    tester.assert_true(len(prompt) > 100, "system_prompt() length sufficient (>100 chars)")
    
    # === Test 5.2: Contains key information ===
    tester.assert_in('BBox', prompt, "system_prompt contains 'BBox' format description")
    tester.assert_in('[', prompt, "system_prompt contains brackets")
    
    # === Test 5.3: Different format prompts ===
    formats = ["free_think", "grounding", "direct"]
    for fmt in formats:
        config = BoundingBoxChartEnvConfig(prompt_format=fmt)
        env = BoundingBoxChartEnv(config)
        prompt = env.system_prompt()
        
        tester.assert_isinstance(prompt, str, f"Format '{fmt}' system_prompt is string")
        tester.assert_true(len(prompt) > 50, f"Format '{fmt}' system_prompt not empty")
    
    env.close()
    print(f"\nSubtotal: system_prompt tests complete")


def test_reward_calculation(tester):
    """Test reward calculation correctness"""
    print("\n" + "=" * 70)
    print("Test 6: Reward Calculation")
    print("=" * 70)
    
    config = BoundingBoxChartEnvConfig(reward_scale=1.0)
    env = BoundingBoxChartEnv(config)
    
    # === Test 6.1: IoU = 1.0 -> high reward ===
    print("\n6.1 Perfect match reward")
    obs, info = env.reset(seed=42)
    gt = info['ground_truth_bbox']
    llm_response = f"BBox[{gt[0]}, {gt[1]}, {gt[2]}, {gt[3]}]"
    _, reward, _, step_info = env.step(llm_response)
    
    tester.assert_true(reward >= 0.9, "IoU=1.0 reward >= 0.9")
    tester.assert_true(step_info['iou'] > 0.99, "Perfect match IoU > 0.99")
    
    # === Test 6.2: Invalid input -> negative reward ===
    print("\n6.2 Invalid input negative reward")
    env.reset(seed=123)
    _, reward, _, _ = env.step("Invalid input")
    
    tester.assert_true(reward < 0, "Invalid input gets negative reward")
    
    # === Test 6.3: Reward range ===
    print("\n6.3 Reward in reasonable range")
    env.reset(seed=456)
    _, reward, _, _ = env.step("BBox[0.1, 0.1, 0.5, 0.5]")
    
    tester.assert_true(-1 <= reward <= 1.5, "Reward in reasonable range [-1, 1.5]")
    
    env.close()
    print(f"\nSubtotal: Reward calculation tests complete")


def test_config_variations(tester):
    """Test different configuration inputs/outputs"""
    print("\n" + "=" * 70)
    print("Test 7: Different Configurations")
    print("=" * 70)
    
    # === Test 7.1: Different image sizes ===
    print("\n7.1 Different image sizes")
    for width, height in [(320, 240), (640, 480), (1024, 768)]:
        config = BoundingBoxChartEnvConfig(image_width=width, image_height=height)
        env = BoundingBoxChartEnv(config)
        obs, _ = env.reset(seed=42)
        
        if obs.get('multi_modal_data') and obs['multi_modal_data'].get('<image>'):
            img = obs['multi_modal_data']['<image>'][0]
            tester.assert_equal(img.size, (width, height),
                               f"Image size correct {width}x{height}")
        env.close()
    
    # === Test 7.2: Text mode ===
    print("\n7.2 Text render mode")
    config = BoundingBoxChartEnvConfig(render_mode="text")
    env = BoundingBoxChartEnv(config)
    obs, _ = env.reset(seed=42)
    
    tester.assert_in('obs_str', obs, "Text mode has obs_str")
    tester.assert_isinstance(obs['obs_str'], str, "obs_str is string")
    # Text mode may not have multi_modal_data
    
    env.close()
    print(f"\nSubtotal: Configuration tests complete")


def main():
    """Run all unit tests"""
    print("\n" + "🧪" * 35)
    print("BoundingBox Environment - Unit Test Suite")
    print("Focus on: Input formats, output formats, boundary conditions, error handling")
    print("🧪" * 35)
    
    tester = TestBoundingBoxEnv()
    
    try:
        # Test 1: reset() output format
        test_reset_output_format(tester)
        
        # Test 2: step() valid inputs
        test_step_valid_inputs(tester)
        
        # Test 3: step() invalid inputs
        test_step_invalid_inputs(tester)
        
        # Test 4: step() boundary conditions
        test_step_boundary_conditions(tester)
        
        # Test 5: system_prompt() output
        test_system_prompt_output(tester)
        
        # Test 6: Reward calculation
        test_reward_calculation(tester)
        
        # Test 7: Different configurations
        test_config_variations(tester)
        
        # Summary
        print("\n" + "=" * 70)
        print("Test Results Summary")
        print("=" * 70)
        print(f"\n✅ Passed: {tester.passed}")
        print(f"❌ Failed: {tester.failed}")
        print(f"📊 Total: {tester.passed + tester.failed}")
        
        if tester.failed > 0:
            print("\nFailed tests:")
            for result in tester.test_results:
                if "❌" in result:
                    print(f"  {result}")
        
        if tester.failed == 0:
            print("\n" + "🎉" * 35)
            print("All unit tests passed! Environment input/output correct!")
            print("🎉" * 35)
            return 0
        else:
            print("\n⚠️ Some tests failed, please check implementation")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
