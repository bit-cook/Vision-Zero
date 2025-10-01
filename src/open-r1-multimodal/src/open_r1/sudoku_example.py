#!/usr/bin/env python3
"""
Example of using Sudoku generator with dynamic dataset for VLM training
"""

import os
import sys
sys.path.append('src/open_r1/trainer')

from sudoku_dynamic_generator import create_sudoku_data_generator
from trainer.dynamic_dataset import EpochAwareIterableDataset

def sudoku_example():
    """Example usage of Sudoku dynamic generator"""
    
    # Create sudoku data generator and reward function
    data_generator, reward_function = create_sudoku_data_generator(
        clues_range=(25, 35),  # Number of given clues in puzzle
        output_dir="/tmp/sudoku_images"
    )
    
    # Wrapper function to convert sudoku format to expected format
    def sudoku_wrapper(epoch=None, sample_idx=None):
        # Generate sudoku sample
        sample = data_generator(epoch, sample_idx)
        
        # Convert to expected format
        return {
            'image_path': [sample['image']],
            'problem': sample['conversations'][0]['value'].replace('<image>\n', ''),
            'solution': sample['conversations'][1]['value'],
            'accu_reward_method': 'sudoku',
            'sudoku_metadata': sample['metadata'],
            'puzzle_board': sample['puzzle_board'],
            'solution_board': sample['solution_board']
        }
    
    # Create dynamic dataset 
    dataset = EpochAwareIterableDataset(
        data_generator_func=sudoku_wrapper,
        epoch_size=10,  # Generate 10 puzzles per epoch
        seed=42
    )
    
    # Set current epoch
    dataset.set_epoch(1)
    
    print("=== Sudoku Dynamic Dataset Example ===")
    print(f"Epoch size: {dataset.epoch_size}")
    print(f"Current epoch: {dataset.current_epoch}")
    
    # Generate a few samples
    for i, sample in enumerate(dataset):
        if i >= 3:  # Only show first 3 samples
            break
            
        print(f"\n--- Sample {i+1} ---")
        print(f"Image path: {sample['image_path'][0]}")
        print(f"Clues: {sample['sudoku_metadata']['clues']}")
        print(f"Prompt length: {len(sample['problem'])} chars")
        print(f"Response length: {len(sample['solution'])} chars")
        
        # Show first few lines of the conversation
        human_msg = sample['problem']
        gpt_msg = sample['solution']
        
        print(f"\nHuman message (first 200 chars):")
        print(human_msg[:200] + "...")
        
        print(f"\nGPT response (first 300 chars):")
        print(gpt_msg[:300] + "...")
        
        # Test reward calculation - need to reconstruct the original sample format
        original_sample = {
            'puzzle_board': sample['puzzle_board'],
            'solution_board': sample['solution_board'],
            'metadata': sample['sudoku_metadata']
        }
        rewards = reward_function(gpt_msg, original_sample)
        print(f"\nReward scores:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.3f}")

    print("\n=== Integration with grpo_jsonl.py ===")
    print("To use this generator with training, run:")
    print("python grpo_jsonl.py \\")
    print("  --use_dynamic_dataset \\")
    print("  --epoch_size 1000 \\")
    print("  --data_generator_type sudoku \\")
    print("  --sudoku_clues_min 25 \\")
    print("  --sudoku_clues_max 35 \\")
    print("  --sudoku_output_dir /path/to/sudoku/images \\")
    print("  --model_name_or_path /path/to/model \\")
    print("  --output_dir /path/to/output \\")
    print("  --num_train_epochs 3")


if __name__ == "__main__":
    sudoku_example() 