#!/usr/bin/env python3
"""
Test script for CLEVR Spot-the-Difference two-phase training.

This script tests the integration between the CLEVR spot-the-difference game generator
and the modified GRPO trainer to ensure two-phase training works correctly.
"""

import os
import sys
from pathlib import Path

# Add the src path to PYTHONPATH for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from clevr_spotdiff_generator import CLEVRSpotDiffGenerator, create_clevr_spotdiff_data_generator

def test_clevr_spotdiff_generator():
    """Test the basic CLEVR spot-the-difference generator functionality"""
    print("=" * 60)
    print("Testing CLEVR Spot-the-Difference Generator")
    print("=" * 60)
    
    try:
        # Initialize generator with multi-round support
        generator = CLEVRSpotDiffGenerator(
            images_dir="/home/colligo/clevr-dataset-gen/output/comparison_images",
            scenes_dir="/home/colligo/clevr-dataset-gen/output/comparison_scenes",
            num_players=4,
            num_rounds=3
        )
        
        print(f"✓ Generator initialized successfully")
        print(f"  - Loaded {len(generator.comparison_data)} comparison pairs")
        
        # Generate a sample game
        game_data = generator.generate_game_sample(epoch=1, sample_idx=1)
        print(f"✓ Game sample generated successfully")
        print(f"  - Game ID: {game_data['game_id']}")
        print(f"  - Spy player: {game_data['spy_player']}")
        print(f"  - Correct answers: {game_data['correct_answers']}")
        
        # Test clue phase formatting for multiple rounds
        print(f"  - Testing multi-round clue generation (num_rounds: {generator.num_rounds})")
        for round_num in range(1, generator.num_rounds + 1):
            for player_id in range(1, game_data["num_players"] + 1):
                clue_sample = generator.format_clue_phase_sample(game_data, player_id, round_num)
                is_spy = clue_sample['metadata']['is_spy']
                
                # Verify round information is in prompt
                prompt = clue_sample["conversations"][0]["value"]
                assert f"Round {round_num}/{generator.num_rounds}" in prompt
                
                if round_num == 1 and player_id == 1:  # Print details for first sample
                    print(f"    Round {round_num}, Player {player_id}: {'SPY' if is_spy else 'NORMAL'} - prompt contains round info")
        print(f"  ✓ Multi-round clue phase formatting tested successfully")
        
        # Test decision phase formatting with multi-round transcript
        test_clues = """Round 1 - Player 1: I see colorful geometric shapes.
Round 1 - Player 2: The arrangement looks balanced.
Round 2 - Player 1: There are various sized objects.
Round 2 - Player 2: The positioning seems deliberate.
Round 3 - Player 1: I notice distinct material properties.
Round 3 - Player 2: The spatial relationships are clear."""
        decision_sample = generator.format_decision_phase_sample(game_data, test_clues)
        
        # Verify multi-round information is in decision prompt
        decision_prompt = decision_sample["conversations"][0]["value"]
        assert f"{generator.num_rounds} clue rounds" in decision_prompt
        assert f"all {generator.num_rounds} rounds" in decision_prompt
        
        print(f"✓ Decision phase sample generated successfully for {generator.num_rounds} rounds")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in generator test: {e}")
        return False

def test_data_generator_integration():
    """Test the data generator integration with GRPO training format"""
    print("\n" + "=" * 60)
    print("Testing Data Generator Integration")
    print("=" * 60)
    
    try:
        # Create data generator function with multi-round support
        base_generator_func, reward_funcs_tuple = create_clevr_spotdiff_data_generator(
            images_dir="/home/colligo/clevr-dataset-gen/output/comparison_images",
            scenes_dir="/home/colligo/clevr-dataset-gen/output/comparison_scenes",
            num_players=4,
            num_rounds=3
        )
        
        print(f"✓ Data generator created successfully")
        print(f"  - Number of reward functions: {len(reward_funcs_tuple)}")
        
        # Generate a training sample
        sample = base_generator_func(epoch=1, sample_idx=1)
        print(f"✓ Training sample generated successfully")
        
        # Verify sample format
        required_keys = ['image_path', 'problem', 'solution', 'accu_reward_method', 
                        'prompt', 'game_data', 'correct_answers', 'metadata']
        missing_keys = [key for key in required_keys if key not in sample]
        
        if missing_keys:
            print(f"✗ Missing required keys: {missing_keys}")
            return False
        else:
            print(f"✓ Sample format validation passed")
            
        print(f"  - accu_reward_method: {sample['accu_reward_method']}")
        print(f"  - Image path: {sample['image_path'][0].split('/')[-1]}")
        print(f"  - Spy player: {sample['game_data']['spy_player']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data generator test: {e}")
        return False

def test_reward_functions():
    """Test the three CLEVR spot-the-difference reward functions"""
    print("\n" + "=" * 60)
    print("Testing Reward Functions")
    print("=" * 60)
    
    try:
        # Create data generator and reward functions
        base_generator_func, reward_funcs_tuple = create_clevr_spotdiff_data_generator()
        spy_reward_func, object_reward_func, movement_reward_func = reward_funcs_tuple
        
        print(f"✓ Reward functions extracted successfully")
        
        # Generate test data
        sample = base_generator_func(epoch=1, sample_idx=1)
        game_data = sample['game_data']
        
        # Create test completions (correct and incorrect)
        correct_answers = sample['correct_answers']
        
        test_completions = [
            # Correct answer
            [{
                "content": f'<think>Let me analyze...</think>\n<answer>"spy": {correct_answers["spy"]}, "object": {correct_answers["object"]}, "movement": {correct_answers["movement"]}</answer>'
            }],
            # Incorrect answer
            [{
                "content": '<think>Let me analyze...</think>\n<answer>"spy": A, "object": A, "movement": A</answer>'
            }]
        ]
        
        test_kwargs = {
            'game_data': [game_data, game_data]
        }
        
        # Test spy identification reward
        spy_rewards = spy_reward_func(None, test_completions, **test_kwargs)
        print(f"✓ Spy identification rewards: {spy_rewards}")
        
        # Test object identification reward  
        object_rewards = object_reward_func(None, test_completions, **test_kwargs)
        print(f"✓ Object identification rewards: {object_rewards}")
        
        # Test movement identification reward
        movement_rewards = movement_reward_func(None, test_completions, **test_kwargs)
        print(f"✓ Movement identification rewards: {movement_rewards}")
        
        # Verify reward logic
        if spy_rewards[0] >= spy_rewards[1]:  # First should be better (correct)
            print(f"✓ Reward function logic validation passed")
        else:
            print(f"✗ Reward function logic validation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error in reward function test: {e}")
        return False

def test_two_phase_simulation():
    """Simulate the two-phase training process"""
    print("\n" + "=" * 60)
    print("Testing Two-Phase Training Simulation")
    print("=" * 60)
    
    try:
        # Initialize generator with multi-round support
        generator = CLEVRSpotDiffGenerator(num_rounds=3)
        
        # Generate game data
        game_data = generator.generate_game_sample(epoch=1, sample_idx=1)
        print(f"✓ Game data generated")
        print(f"  - Number of rounds: {generator.num_rounds}")
        
        # Phase 1: Simulate multi-round clue generation
        print(f"\n--- Phase 1: Multi-Round Clue Generation ---")
        all_clues = []
        
        # Simulate clue phrases that vary by round
        clue_templates = [
            ["colorful geometric shapes", "spatial arrangements", "distinct visual elements"],
            ["balanced composition", "structured layout", "organized patterns"],
            ["interesting relationships", "notable positioning", "unique characteristics"],
            ["varied object sizes", "different materials", "diverse orientations"]
        ]
        
        for round_num in range(1, generator.num_rounds + 1):
            print(f"\nRound {round_num}:")
            for player_id in range(1, game_data["num_players"] + 1):
                # Create clue phase sample - each player sees all previous clues including current round
                prior_clues = "\n".join(all_clues) if all_clues else "None yet."
                clue_sample = generator.format_clue_phase_sample(game_data, player_id, round_num, prior_clues)
                
                # Simulate clue generation with variation by round
                is_spy = clue_sample['metadata']['is_spy']
                
                # Pick template based on player and add round variation
                base_template = clue_templates[player_id - 1][round_num - 1]
                if is_spy:
                    simulated_clue = f"I observe {base_template} across the scene."
                else:
                    simulated_clue = f"The scene contains {base_template} throughout."
                
                # Immediately add clue so next player in same round can see it
                clue_text = f"Round {round_num} - Player {player_id}: {simulated_clue}"
                all_clues.append(clue_text)
                print(f"  Player {player_id} ({'SPY' if is_spy else 'NORMAL'}): {simulated_clue}")
                
                # Show what the next player will see (for demonstration)
                if player_id < game_data["num_players"]:
                    print(f"    → Next player will see {len(all_clues)} total clues so far")
        
        print(f"\n✓ Generated {len(all_clues)} clues across {generator.num_rounds} rounds")
        
        # Phase 2: Decision making based on clues
        print(f"\n--- Phase 2: Decision Making ---")
        full_transcript = "\n".join(all_clues)
        decision_sample = generator.format_decision_phase_sample(game_data, full_transcript)
        
        print(f"✓ Decision phase sample created")
        print(f"  - Transcript length: {len(full_transcript)} characters")
        print(f"  - Total clues: {len(all_clues)} ({generator.num_rounds} rounds × {game_data['num_players']} players)")
        print(f"  - Correct spy: Player {game_data['spy_player']} (Answer: {game_data['correct_answers']['spy']})")
        
        # Test reward calculation with decision sample
        decision_response = decision_sample["conversations"][1]["value"]
        reward_tuple = generator.calculate_decision_reward(decision_response, game_data)
        print(f"  - Perfect answer rewards: {reward_tuple}")
        
        if all(r == 1.0 for r in reward_tuple):
            print(f"✓ Two-phase simulation successful")
            return True
        else:
            print(f"✗ Reward calculation issue")
            return False
            
    except Exception as e:
        print(f"✗ Error in two-phase simulation: {e}")
        return False

def main():
    """Run all tests"""
    print("CLEVR Spot-the-Difference Training Integration Test")
    print("=" * 60)
    
    tests = [
        ("Basic Generator", test_clevr_spotdiff_generator),
        ("Data Generator Integration", test_data_generator_integration),
        ("Reward Functions", test_reward_functions),
        ("Two-Phase Simulation", test_two_phase_simulation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLEVR spot-the-difference integration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 