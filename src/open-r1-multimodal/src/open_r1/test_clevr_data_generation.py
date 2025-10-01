#!/usr/bin/env python3
"""
Test script for CLEVR spot-the-difference data generation.
This verifies that the data generator works properly before running full training.
"""

import os
import sys
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clevr_spotdiff_generator import create_clevr_spotdiff_data_generator

def test_clevr_data_generation():
    """Test the CLEVR spot-the-difference data generation"""
    
    print("Testing CLEVR Spot-the-Difference Data Generation")
    print("=" * 50)
    
    # Configuration
    images_dir = "/home/colligo/clevr-dataset-gen/output/comparison_images"
    scenes_dir = "/home/colligo/clevr-dataset-gen/output/comparison_scenes"
    num_players = 4
    num_rounds = 3
    
    print(f"Images directory: {images_dir}")
    print(f"Scenes directory: {scenes_dir}")
    print(f"Number of players: {num_players}")
    print(f"Number of rounds: {num_rounds}")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        return False
        
    if not os.path.exists(scenes_dir):
        print(f"ERROR: Scenes directory not found: {scenes_dir}")
        return False
    
    # List available files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    scene_files = [f for f in os.listdir(scenes_dir) if f.endswith('.json')]
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(scene_files)} scene files")
    
    if len(image_files) == 0 or len(scene_files) == 0:
        print("ERROR: No data files found")
        return False
    
    try:
        # Create data generator
        print("\nCreating data generator...")
        data_generator_func, reward_funcs = create_clevr_spotdiff_data_generator(
            images_dir=images_dir,
            scenes_dir=scenes_dir,
            num_players=num_players,
            num_rounds=num_rounds
        )
        
        print("✓ Data generator created successfully")
        print(f"✓ Got {len(reward_funcs)} reward functions")
        
        # Test generating a few samples
        print("\nTesting sample generation...")
        for i in range(3):
            try:
                sample = data_generator_func(epoch=0, sample_idx=i)
                
                print(f"\n--- Sample {i+1} ---")
                print(f"Keys: {list(sample.keys())}")
                
                if 'game_data' in sample:
                    game_data = sample['game_data']
                    print(f"Game ID: {game_data.get('game_id', 'N/A')}")
                    print(f"Spy player: {game_data.get('spy_player', 'N/A')}")
                    print(f"Num players: {game_data.get('num_players', 'N/A')}")
                    print(f"Correct answers: {game_data.get('correct_answers', 'N/A')}")
                    
                    # Check if player images exist
                    player_images = game_data.get('player_images', [])
                    if player_images:
                        print(f"Player images: {len(player_images)} paths")
                        for j, img_path in enumerate(player_images[:2]):  # Check first 2
                            exists = os.path.exists(img_path) if img_path != "/dev/null" else True
                            print(f"  Player {j+1}: {img_path} {'✓' if exists else '✗'}")
                else:
                    print("WARNING: No game_data found in sample")
                
                print(f"✓ Sample {i+1} generated successfully")
                
            except Exception as e:
                print(f"✗ Sample {i+1} failed: {e}")
                traceback.print_exc()
                return False
        
        # Test reward functions
        print("\nTesting reward functions...")
        try:
            # Create dummy completions
            dummy_completions = [
                [{"content": '<think>I think Player 2 is suspicious.</think>\n<answer>"spy": B, "object": A, "movement": C</answer>'}],
                [{"content": '<think>Player 1 seems odd.</think>\n<answer>"spy": A, "object": B, "movement": A</answer>'}],
                [{"content": '<think>Not sure.</think>\n<answer>"spy": C, "object": C, "movement": B</answer>'}]
            ]
            
            # Get game data for rewards
            sample1 = data_generator_func(epoch=0, sample_idx=0)
            sample2 = data_generator_func(epoch=0, sample_idx=1)
            sample3 = data_generator_func(epoch=0, sample_idx=2)
            
            kwargs = {
                'game_data': [sample1['game_data'], sample2['game_data'], sample3['game_data']]
            }
            
            for i, reward_func in enumerate(reward_funcs):
                try:
                    rewards = reward_func(prompts=None, completions=dummy_completions, **kwargs)
                    print(f"✓ Reward function {i+1}: {rewards}")
                except Exception as e:
                    print(f"✗ Reward function {i+1} failed: {e}")
                    return False
        
        except Exception as e:
            print(f"✗ Reward testing failed: {e}")
            traceback.print_exc()
            return False
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("The CLEVR spot-the-difference data generator is working correctly.")
        print(f"Ready for multi-round training with {num_rounds} rounds!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clevr_data_generation()
    
    if success:
        print("\n✅ Data generation test completed successfully!")
        print("You can now run the training with:")
        print("./run_scripts/run_grpo_clevr_spotdiff_multiround.sh")
    else:
        print("\n❌ Data generation test failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1) 