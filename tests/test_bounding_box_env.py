"""
BoundingBox Chart Environment Tests
Tests the corrected two-layer architecture (bounding_box.py + env.py)

Ensure dependencies are installed before running:
    pip install numpy Pillow gymnasium

Usage:
    python examples/test_bounding_box_env.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from vagen.env.bounding_box.bounding_box import BoundingBoxChartEnv as GymBoundingBoxEnv
from vagen.env.bounding_box.env import BoundingBoxChartEnv as VAGENBoundingBoxEnv
from vagen.env.bounding_box.env_config import BoundingBoxChartEnvConfig


def test_gym_environment():
    """Test core Gym environment (bounding_box.py)"""
    print("=" * 70)
    print("Test 1: Core Gym Environment (bounding_box.py)")
    print("=" * 70)
    
    # 1. Create Gym environment
    gym_env = GymBoundingBoxEnv(image_width=640, image_height=480)
    print("\n✅ Gym environment created successfully")
    print(f"   - Action space: {gym_env.action_space}")
    print(f"   - Observation space: {gym_env.observation_space}")
    
    # Assertions: check environment attributes
    assert hasattr(gym_env, 'action_space'), "Environment should have action_space"
    assert hasattr(gym_env, 'observation_space'), "Environment should have observation_space"
    
    # 2. Reset
    obs, info = gym_env.reset(seed=42)
    print("\n✅ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Observation dtype: {obs.dtype}")
    gt_bbox = info['ground_truth_bbox']
    print(f"   - Ground truth bbox: [{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
    
    # Assertions: check observation and info
    assert obs.shape == (480, 640, 3), f"Observation shape should be (480, 640, 3), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Observation dtype should be uint8, got {obs.dtype}"
    assert 'ground_truth_bbox' in info, "info should contain ground_truth_bbox"
    assert len(gt_bbox) == 4, f"Bounding box should have 4 values, got {len(gt_bbox)}"
    assert all(0 <= x <= 1 for x in gt_bbox), f"Bounding box coordinates should be in [0, 1] range, got {gt_bbox}"
    assert gt_bbox[0] < gt_bbox[2], "x_min should be less than x_max"
    assert gt_bbox[1] < gt_bbox[3], "y_min should be less than y_max"
    
    # 3. Step with perfect action
    action = np.array(gt_bbox, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = gym_env.step(action)
    print("\n✅ Step successful (perfect prediction)")
    print(f"   - IoU: {info['iou']:.4f}")
    print(f"   - Reward: {reward:.4f}")
    print(f"   - Terminated: {terminated}")
    print(f"   - Truncated: {truncated}")
    
    # Assertions: perfect prediction should have high IoU and reward
    assert info['iou'] >= 0.99, f"Perfect prediction IoU should be close to 1.0, got {info['iou']:.4f}"
    assert reward > 0, f"Perfect prediction should have positive reward, got {reward:.4f}"
    assert terminated == True, "Perfect prediction should terminate episode"
    assert next_obs.shape == obs.shape, "Observation shape should remain consistent"
    
    # 4. Step with imperfect action
    obs, info = gym_env.reset(seed=123)
    gt_bbox_new = info['ground_truth_bbox']
    imperfect_action = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = gym_env.step(imperfect_action)
    print("\n✅ Step successful (imperfect prediction)")
    print(f"   - IoU: {info['iou']:.4f}")
    print(f"   - Reward: {reward:.4f}")
    
    # Assertions: imperfect prediction should have lower IoU
    assert 0 <= info['iou'] < 1.0, f"Imperfect prediction IoU should be in [0, 1) range, got {info['iou']:.4f}"
    assert 'iou' in info, "info should contain iou"
    
    # 5. Step with invalid action
    gym_env.reset(seed=456)
    invalid_action = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float32)  # Out of bounds
    next_obs, reward, terminated, truncated, info = gym_env.step(invalid_action)
    print("\n✅ Invalid action handled correctly")
    print(f"   - Reward (negative): {reward:.4f}")
    print(f"   - Error message: {info.get('error', 'N/A')}")
    
    # Assertions: invalid action should have negative reward
    assert reward < 0, f"Invalid action should have negative reward, got {reward:.4f}"
    assert 'error' in info or info.get('iou', 1) < 1, "Invalid action should contain error in info or low IoU"
    
    # 6. Render
    rendered = gym_env.render()
    if rendered is not None:
        print(f"\n✅ Render successful")
        print(f"   - Render shape: {rendered.shape}")
        assert rendered.shape == (480, 640, 3), f"Render shape should be (480, 640, 3), got {rendered.shape}"
    
    # 7. Close
    gym_env.close()
    print("\n✅ Gym environment closed successfully")
    
    return True


def test_vagen_wrapper():
    """Test VAGEN wrapper (env.py)"""
    print("\n" + "=" * 70)
    print("Test 2: VAGEN Wrapper (env.py)")
    print("=" * 70)
    
    # 1. Create configuration
    config = BoundingBoxChartEnvConfig(
        render_mode="vision",
        prompt_format="free_think",
        iou_threshold=0.5,
        image_width=640,
        image_height=480,
        draw_bbox=True
    )
    print(f"\n✅ Configuration created successfully")
    print(f"   - {config.config_id()}")
    
    # Assertions: check configuration
    assert config.render_mode == "vision", "Render mode should be vision"
    assert config.prompt_format == "free_think", "Prompt format should be free_think"
    assert config.iou_threshold == 0.5, "IoU threshold should be 0.5"
    
    # 2. Create VAGEN environment
    env = VAGENBoundingBoxEnv(config)
    print("\n✅ VAGEN environment created successfully")
    print(f"   - Internal Gym environment: {type(env.gym_env).__name__}")
    
    # Assertions: check environment structure
    assert hasattr(env, 'gym_env'), "VAGEN environment should have gym_env attribute"
    assert hasattr(env, 'system_prompt'), "VAGEN environment should have system_prompt method"
    assert hasattr(env, 'reset'), "VAGEN environment should have reset method"
    assert hasattr(env, 'step'), "VAGEN environment should have step method"
    
    # 3. Reset
    obs, info = env.reset(seed=42)
    print("\n✅ Reset successful")
    print(f"   - Observation keys: {list(obs.keys())}")
    print(f"   - Observation string length: {len(obs['obs_str'])}")
    if obs.get('multi_modal_data'):
        images = obs['multi_modal_data'].get('<image>', [])
        if images:
            print(f"   - Image count: {len(images)}")
            print(f"   - Image size: {images[0].size}")
    gt_bbox = info['ground_truth_bbox']
    print(f"   - Ground truth bbox: [{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
    
    # Assertions: check observation structure
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert 'obs_str' in obs, "Observation should contain obs_str"
    assert isinstance(obs['obs_str'], str), "obs_str should be a string"
    assert len(obs['obs_str']) > 0, "obs_str should not be empty"
    if config.render_mode == "vision":
        assert 'multi_modal_data' in obs, "vision mode should contain multi_modal_data"
        assert '<image>' in obs['multi_modal_data'], "multi_modal_data should contain images"
        images = obs['multi_modal_data']['<image>']
        assert len(images) > 0, "Should have at least one image"
    
    # 4. System Prompt
    sys_prompt = env.system_prompt()
    print(f"\n✅ System Prompt successful")
    print(f"   - Length: {len(sys_prompt)} characters")
    print(f"   - First 100 characters: {sys_prompt[:100]}...")
    
    # Assertions: check system prompt
    assert isinstance(sys_prompt, str), "System prompt should be a string"
    assert len(sys_prompt) > 0, "System prompt should not be empty"
    
    # 5. Step with perfect LLM response
    llm_response = f"""<think>
I analyzed the chart structure and identified the main plot area.
Axes are on the left and bottom, title is at the top.
</think>
<answer>BBox[{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]</answer>"""
    
    next_obs, reward, done, info = env.step(llm_response)
    print("\n✅ Step successful (perfect LLM response)")
    print(f"   - Action valid: {info['metrics']['turn_metrics']['action_is_valid']}")
    print(f"   - Action effective: {info['metrics']['turn_metrics']['action_is_effective']}")
    print(f"   - Task success: {info['metrics']['traj_metrics']['success']}")
    print(f"   - IoU: {info['iou']:.4f}")
    print(f"   - Reward: {reward:.4f}")
    print(f"   - Episode done: {done}")
    
    # Assertions: perfect response should succeed
    assert info['metrics']['turn_metrics']['action_is_valid'], "Perfect response should be valid"
    assert info['metrics']['turn_metrics']['action_is_effective'], "Perfect response should be effective"
    assert info['metrics']['traj_metrics']['success'], "Perfect response should succeed"
    assert info['iou'] >= 0.99, f"Perfect response IoU should be close to 1.0, got {info['iou']:.4f}"
    assert reward > 0, f"Perfect response should have positive reward, got {reward:.4f}"
    assert done, "Perfect response should complete episode"
    
    # 6. Step with imperfect LLM response
    obs, info = env.reset(seed=123)
    gt_bbox = info['ground_truth_bbox']
    
    pred_bbox = [
        max(0, gt_bbox[0] - 0.08),
        max(0, gt_bbox[1] - 0.08),
        min(1, gt_bbox[2] + 0.05),
        min(1, gt_bbox[3] + 0.05)
    ]
    
    llm_response = f"""<think>Attempting prediction</think>
<answer>BBox[{pred_bbox[0]:.3f}, {pred_bbox[1]:.3f}, {pred_bbox[2]:.3f}, {pred_bbox[3]:.3f}]</answer>"""
    
    next_obs, reward, done, info = env.step(llm_response)
    print("\n✅ Step successful (imperfect LLM response)")
    print(f"   - IoU: {info['iou']:.4f}")
    print(f"   - Reward: {reward:.4f}")
    print(f"   - Success: {info['metrics']['traj_metrics']['success']}")
    
    # Assertions: imperfect response
    assert info['metrics']['turn_metrics']['action_is_valid'], "Correctly formatted response should be valid"
    assert 0 <= info['iou'] < 1.0, f"Imperfect response IoU should be in [0, 1) range, got {info['iou']:.4f}"
    
    # 7. Step with invalid LLM response (format error)
    env.reset(seed=456)
    invalid_response = "<think>Test</think><answer>No bounding box format</answer>"
    
    next_obs, reward, done, info = env.step(invalid_response)
    print("\n✅ Invalid LLM response handled correctly")
    print(f"   - Action valid: {info['metrics']['turn_metrics']['action_is_valid']}")
    print(f"   - Reward (negative): {reward:.4f}")
    print(f"   - Parse error: {info['llm_response'].get('error', 'N/A')}")
    
    # Assertions: invalid format should be handled correctly
    assert not info['metrics']['turn_metrics']['action_is_valid'], "Incorrectly formatted response should be invalid"
    assert reward < 0, f"Invalid response should have negative reward, got {reward:.4f}"
    assert 'error' in info['llm_response'], "Invalid response should contain error in llm_response"
    
    # 8. Step with invalid coordinates
    env.reset(seed=789)
    invalid_coords = "<think>Out of bounds test</think><answer>BBox[0.5, 0.5, 1.5, 1.5]</answer>"
    
    next_obs, reward, done, info = env.step(invalid_coords)
    print("\n✅ Out of bounds coordinates handled correctly")
    print(f"   - Action valid: {info['metrics']['turn_metrics']['action_is_valid']}")
    print(f"   - Error: {info['llm_response'].get('error', 'N/A')}")
    
    # Assertions: out of bounds coordinates should be handled correctly
    assert not info['metrics']['turn_metrics']['action_is_valid'], "Out of bounds coordinates should be invalid"
    assert reward < 0, f"Out of bounds coordinates should have negative reward, got {reward:.4f}"
    
    # 9. Close
    env.close()
    print("\n✅ VAGEN environment closed successfully")
    
    return True


def test_different_prompt_formats():
    """Test different prompt formats"""
    print("\n" + "=" * 70)
    print("Test 3: Different Prompt Formats")
    print("=" * 70)
    
    formats = ["free_think", "grounding", "direct"]
    
    for fmt in formats:
        config = BoundingBoxChartEnvConfig(
            render_mode="vision",
            prompt_format=fmt
        )
        env = VAGENBoundingBoxEnv(config)
        
        obs, info = env.reset(seed=42)
        sys_prompt = env.system_prompt()
        
        print(f"\n✅ Format '{fmt}' test successful")
        print(f"   - System prompt length: {len(sys_prompt)} characters")
        
        # Assertions: check each format
        assert isinstance(sys_prompt, str), f"Format '{fmt}' system prompt should be a string"
        assert len(sys_prompt) > 0, f"Format '{fmt}' system prompt should not be empty"
        assert isinstance(obs, dict), f"Format '{fmt}' observation should be a dictionary"
        assert 'obs_str' in obs, f"Format '{fmt}' observation should contain obs_str"
        assert 'ground_truth_bbox' in info, f"Format '{fmt}' info should contain ground_truth_bbox"
        
        env.close()
    
    return True


def test_architecture_integration():
    """Test two-layer architecture integration"""
    print("\n" + "=" * 70)
    print("Test 4: Two-Layer Architecture Integration")
    print("=" * 70)
    
    # Create VAGEN environment
    config = BoundingBoxChartEnvConfig()
    vagen_env = VAGENBoundingBoxEnv(config)
    
    # Verify internal structure
    print("\n✅ Architecture verification")
    print(f"   - VAGEN wrapper class: {type(vagen_env).__name__}")
    print(f"   - Internal Gym environment class: {type(vagen_env.gym_env).__name__}")
    print(f"   - Gym environment is gymnasium.Env: {hasattr(vagen_env.gym_env, 'action_space')}")
    print(f"   - VAGEN environment is BaseEnv: {hasattr(vagen_env, 'system_prompt')}")
    
    # Assertions: check architecture
    assert type(vagen_env).__name__ == "BoundingBoxChartEnv", "Should be VAGEN BoundingBoxChartEnv"
    assert type(vagen_env.gym_env).__name__ == "BoundingBoxChartEnv", "Internal should be Gym BoundingBoxChartEnv"
    assert hasattr(vagen_env.gym_env, 'action_space'), "Gym environment should have action_space"
    assert hasattr(vagen_env.gym_env, 'observation_space'), "Gym environment should have observation_space"
    assert hasattr(vagen_env, 'system_prompt'), "VAGEN environment should have system_prompt method"
    assert hasattr(vagen_env, 'reset'), "VAGEN environment should have reset method"
    assert hasattr(vagen_env, 'step'), "VAGEN environment should have step method"
    
    # Test data flow
    obs, info = vagen_env.reset(seed=42)
    gt_bbox = info['ground_truth_bbox']
    
    # Assertions: Reset returns correct data
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert isinstance(info, dict), "info should be a dictionary"
    assert 'ground_truth_bbox' in info, "info should contain ground_truth_bbox"
    
    # LLM response -> VAGEN wrapper -> Gym environment
    llm_response = f"<think>Test</think><answer>BBox[{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]</answer>"
    next_obs, reward, done, info = vagen_env.step(llm_response)
    
    print("\n✅ Data flow verification")
    print(f"   - LLM text response -> BBox parsing -> Numpy action")
    print(f"   - Gym environment execution -> IoU calculation -> Reward")
    print(f"   - Numpy observation -> Multi-modal format conversion")
    print(f"   - Metrics generation complete")
    
    # Assertions: Step returns correct data
    assert isinstance(next_obs, dict), "Next observation should be a dictionary"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "done should be boolean"
    assert isinstance(info, dict), "info should be a dictionary"
    assert 'metrics' in info, "info should contain metrics"
    assert 'turn_metrics' in info['metrics'], "metrics should contain turn_metrics"
    assert 'traj_metrics' in info['metrics'], "metrics should contain traj_metrics"
    assert 'iou' in info, "info should contain iou"
    assert 'llm_response' in info, "info should contain llm_response"
    
    # Verify perfect prediction results
    assert info['iou'] >= 0.99, f"Perfect prediction IoU should be close to 1.0, got {info['iou']:.4f}"
    assert reward > 0, f"Perfect prediction should have positive reward, got {reward:.4f}"
    assert done, "Perfect prediction should complete episode"
    assert info['metrics']['traj_metrics']['success'], "Perfect prediction should be marked as success"
    
    vagen_env.close()
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🎯" * 35)
    print("BoundingBox Chart Environment - Complete Test Suite")
    print("Testing two-layer architecture: bounding_box.py (Gym) + env.py (VAGEN)")
    print("🎯" * 35)
    
    try:
        # Test 1: Core Gym environment
        test_gym_environment()
        
        # Test 2: VAGEN wrapper
        test_vagen_wrapper()
        
        # Test 3: Different prompt formats
        test_different_prompt_formats()
        
        # Test 4: Architecture integration
        test_architecture_integration()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉🎉🎉 All tests passed! Two-layer architecture working properly! 🎉🎉🎉")
        print("=" * 70)
        
        print("\n✅ Architecture verification:")
        print("   ✓ Core Gym environment (bounding_box.py) working properly")
        print("   ✓ VAGEN wrapper (env.py) working properly")
        print("   ✓ LLM response parsing correct")
        print("   ✓ Multi-modal observation generation correct")
        print("   ✓ Metrics tracking complete")
        print("   ✓ Error handling robust")
        
        print("\n📚 Environment is ready!")
        print("\nNext steps:")
        print("   1. Register environment in vagen/env/__init__.py")
        print("   2. Prepare training dataset")
        print("   3. Configure training parameters")
        print("   4. Start training LLM agent!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Common issues:")
        print("   - Ensure installed: pip install numpy Pillow gymnasium")
        print("   - Ensure running from project root")
        print("   - Check if files bounding_box.py and env.py exist")
        
        return 1


if __name__ == "__main__":
    exit(main())
