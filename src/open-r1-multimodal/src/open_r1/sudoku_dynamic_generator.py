import random
import copy
import os
import re
import tempfile
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import sys

Board = List[List[int]]

def clone_board(board: Board) -> Board:
    """比 copy.deepcopy 快、且不会出现递归爆栈"""
    return [row[:] for row in board]



class SudokuGenerator:
    def __init__(self, clues_range: Tuple[int, int] = (25, 35), image_size: Tuple[int, int] = (600, 600), board_size: int = 9):
        """
        Initialize Sudoku generator
        
        Args:
            clues_range: Range of clues (given numbers) in the puzzle
            image_size: Size of generated images (width, height)
            board_size: Size of the sudoku board (e.g., 3, 4, 6, 9)
        """
        self.clues_range = clues_range
        self.image_size = image_size
        self.board_size = board_size
        self.max_attempts = 100  # Maximum attempts for generation
        
        # Calculate block size (square root of board_size for standard sudoku)
        # For non-perfect squares, use 1x1 blocks
        import math
        sqrt_size = int(math.sqrt(board_size))
        if sqrt_size * sqrt_size == board_size:
            self.block_size = sqrt_size
        else:
            self.block_size = 1
        # Generate dynamic prompt template based on board size
        self._generate_prompt_template()

    def _generate_prompt_template(self):
        """Generate prompt template based on board size"""
        size = self.board_size
        block_desc = ""
        
        if self.block_size > 1:
            num_blocks = (size // self.block_size) ** 2
            block_desc = f"4. Each of the {num_blocks} {self.block_size} × {self.block_size} sub‑grids must contain all digits 1 to {size} exactly once.\n"
        else:
            block_desc = "4. (No sub-grid constraints for this board size.)\n"
        
        # Generate example grid format
        example_row = "[" + ", ".join(["R"] * size) + "]"
        example_grid = "[" + ", ".join([example_row] * size) + "]"
        
        self.prompt_template = f"""You are given an image containing an unfinished {size} × {size} Sudoku grid. 
Complete the puzzle by following these Sudoku rules:

1. Fill every empty cell with a digit from 1 to {size}.
2. Each row must contain all digits 1 to {size} exactly once.
3. Each column must contain all digits 1 to {size} exactly once.
{block_desc}5. Digits already provided in the grid are fixed and may not be changed.

When you reply, **strictly** obey this output format:

<think>
(Write your entire step‑by‑step reasoning here.  
Do NOT reveal the final grid inside this section.)
</think>
<answer>
{example_grid}
</answer>

Formatting requirements for the `<answer>` block:
• It must contain a single line with a {size}x{size} nested list structure
• Each R represents a digit from 1 to {size}
• The format is exactly: {example_grid}
• Each row is a list of {size} integers separated by commas and spaces
• Do not output any other text before, between, or after the `<think>` and `<answer>` blocks.
Any deviation from these rules will be considered incorrect."""

    # ---------- 基础工具 ---------- #
    def find_empty(self, board: Board) -> Optional[Tuple[int, int]]:
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] == 0:
                    return r, c
        return None

    def is_valid(self, board: Board, row: int, col: int, num: int) -> bool:
        # Check row and column constraints
        if num in board[row] or any(board[r][col] == num for r in range(self.board_size)):
            return False
        
        # Check block constraints (only if block_size > 1)
        if self.block_size > 1:
            block_row = self.block_size * (row // self.block_size)
            block_col = self.block_size * (col // self.block_size)
            for r in range(block_row, min(block_row + self.block_size, self.board_size)):
                for c in range(block_col, min(block_col + self.block_size, self.board_size)):
                    if board[r][c] == num:
                        return False
        
        return True

    # ---------- 迭代求解器 (避免递归) ---------- #
    def solve(self, board: Board) -> bool:
        """Iterative sudoku solver to avoid recursion depth issues"""
        stack = []
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if board[r][c] == 0]
        
        if not empty_cells:
            return True
            
        idx = 0
        while idx < len(empty_cells):
            row, col = empty_cells[idx]
            
            # Find next valid number
            current_num = board[row][col]
            found = False
            
            for num in range(current_num + 1, self.board_size + 1):
                if self.is_valid(board, row, col, num):
                    board[row][col] = num
                    stack.append((idx, current_num))
                    found = True
                    break
            
            if found:
                idx += 1
            else:
                # Backtrack
                board[row][col] = 0
                if not stack:
                    return False
                    
                backtrack_idx, prev_num = stack.pop()
                idx = backtrack_idx
                
        return True

    def count_solutions(self, board: Board, limit: int = 2) -> int:
        """Iterative solution counter to avoid recursion depth issues"""
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if board[r][c] == 0]
        
        if not empty_cells:
            return 1
            
        solutions = 0
        stack = [(0, 0)]  # (cell_index, num_to_try)
        
        while stack and solutions < limit:
            idx, num = stack.pop()
            
            if idx >= len(empty_cells):
                solutions += 1
                continue
                
            row, col = empty_cells[idx]
            
            # Reset cell if we're backtracking
            if num == 0:
                board[row][col] = 0
                
            # Try numbers from num+1 to board_size
            found = False
            for try_num in range(num + 1, self.board_size + 1):
                if self.is_valid(board, row, col, try_num):
                    board[row][col] = try_num
                    # Add backtrack point for this cell
                    stack.append((idx, try_num))
                    # Move to next cell
                    stack.append((idx + 1, 0))
                    found = True
                    break
                    
            if not found:
                board[row][col] = 0
                
        # Clean up
        for r, c in empty_cells:
            board[r][c] = 0
            
        return solutions

    # ---------- 生成完整解 (迭代版本) ---------- #
    def fill_board(self, board: Board) -> bool:
        """Iterative board filling to avoid recursion depth issues"""
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if board[r][c] == 0]
        
        if not empty_cells:
            return True
            
        stack = []
        idx = 0
        
        while idx < len(empty_cells):
            row, col = empty_cells[idx]
            current_num = board[row][col]
            
            # Generate shuffled numbers for randomness
            nums = list(range(current_num + 1, self.board_size + 1))
            random.shuffle(nums)
            
            found = False
            for num in nums:
                if self.is_valid(board, row, col, num):
                    board[row][col] = num
                    stack.append((idx, current_num))
                    found = True
                    break
            
            if found:
                idx += 1
            else:
                # Backtrack
                board[row][col] = 0
                if not stack:
                    return False
                    
                backtrack_idx, prev_num = stack.pop()
                idx = backtrack_idx
                
        return True

    # ---------- 挖空 ---------- #
    def make_puzzle(self, full_board: Board, clues: int = 30) -> Board:
        """Create puzzle by removing numbers while ensuring unique solution"""
        puzzle = clone_board(full_board)
        cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        random.shuffle(cells)

        attempts = 0
        max_attempts = 200
        
        while len(cells) > 0 and sum(1 for r in range(self.board_size) for c in range(self.board_size) if puzzle[r][c] != 0) > clues:
            if attempts > max_attempts:
                break
                
            row, col = cells.pop()
            backup = puzzle[row][col]
            puzzle[row][col] = 0
            
            # Check if still has unique solution
            test_board = clone_board(puzzle)
            if self.count_solutions(test_board, limit=2) != 1:
                puzzle[row][col] = backup
                
            attempts += 1
            
        return puzzle

    def generate_puzzle(self, clues: Optional[int] = None) -> Tuple[Board, Board]:
        """Generate a sudoku puzzle and its solution with retry logic"""
        if clues is None:
            clues = random.randint(*self.clues_range)
        
        for attempt in range(self.max_attempts):
            try:
                # Generate complete solution
                solution_board = [[0] * self.board_size for _ in range(self.board_size)]
                
                # Fill diagonal blocks first for better performance (if block_size > 1)
                if self.block_size > 1:
                    self._fill_diagonal_boxes(solution_board)
                
                if self.fill_board(solution_board):
                    # Create puzzle by removing numbers
                    puzzle_board = self.make_puzzle(clone_board(solution_board), clues)
                    return puzzle_board, solution_board
                    
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                continue
                
        # Fallback: create a simple valid puzzle
        return self._create_fallback_puzzle(clues)
        
    def _fill_diagonal_boxes(self, board: Board):
        """Fill the diagonal blocks to start with a partially filled board"""
        for box_idx in range(0, self.board_size, self.block_size):
            if box_idx + self.block_size <= self.board_size:
                nums = list(range(1, self.board_size + 1))
                random.shuffle(nums)
                idx = 0
                for r in range(box_idx, min(box_idx + self.block_size, self.board_size)):
                    for c in range(box_idx, min(box_idx + self.block_size, self.board_size)):
                        if idx < len(nums):
                            board[r][c] = nums[idx]
                            idx += 1
                    
    def _create_fallback_puzzle(self, clues: int) -> Tuple[Board, Board]:
        """Create a simple fallback puzzle when generation fails"""
        # Create a simple sequential solution
        solution = []
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                # Simple pattern: shift each row by its index
                value = ((r + c) % self.board_size) + 1
                row.append(value)
            solution.append(row)
        
        puzzle = clone_board(solution)
        
        # Randomly remove numbers to reach desired clue count
        cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        random.shuffle(cells)
        
        current_clues = self.board_size * self.board_size
        for r, c in cells:
            if current_clues <= clues:
                break
            puzzle[r][c] = 0
            current_clues -= 1
            
        return puzzle, solution

    # ---------- 绘图 ---------- #
    def draw_puzzle(self, board: Board, filename: str) -> str:
        """Draw sudoku puzzle and save as image file"""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 绘制网格
        for i in range(self.board_size + 1):
            # Thick lines for block boundaries (if block_size > 1)
            if self.block_size > 1 and i % self.block_size == 0:
                lw = 2
            else:
                lw = 0.5
            ax.plot([0, self.board_size], [i, i], 'k-', linewidth=lw)
            ax.plot([i, i], [0, self.board_size], 'k-', linewidth=lw)
        
        # 填数字
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = board[r][c]
                if val != 0:
                    # Adjust font size based on board size
                    font_size = max(8, 20 - (self.board_size - 3) * 2)
                    ax.text(c + 0.5, self.board_size - 0.5 - r, str(val), va='center', ha='center', 
                           fontsize=font_size, fontweight='bold')
        
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename

    def generate_sample(self, epoch: int, sample_idx: int, output_dir: str = "/home/colligo/sudoku_images") -> Dict[str, Any]:
        """Generate a single sudoku training sample - guaranteed to never return None"""
        try:
            # Set seed for reproducibility within epoch

            random.seed(epoch * 10000 + sample_idx)
            
            # Generate puzzle and solution
            puzzle, solution = self.generate_puzzle()
            
            # Create temporary image file
            os.makedirs(output_dir, exist_ok=True)
            image_filename = os.path.join(output_dir, f"sudoku_epoch_{epoch}_sample_{sample_idx}.png")
            self.draw_puzzle(puzzle, image_filename)
            
            # Format solution for comparison
            solution_text = self.format_solution(solution)
            
            return {
                "image": [image_filename],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{self.prompt_template}"
                    },
                    {
                        "from": "gpt", 
                        "value": solution_text
                    }
                ],
                "puzzle_board": [puzzle],
                "solution_board": [solution],
                "metadata": [{
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "clues": sum(1 for r in range(self.board_size) for c in range(self.board_size) if puzzle[r][c] != 0),
                    "board_size": self.board_size
                }]
            }
        except Exception as e:
            print(f"Error generating sudoku sample {sample_idx}: {e}")
            # Return a failsafe sample using fallback puzzle
            puzzle, solution = self._create_fallback_puzzle(30)
            
            # Create temporary image file
            os.makedirs(output_dir, exist_ok=True)
            image_filename = os.path.join(output_dir, f"sudoku_epoch_{epoch}_sample_{sample_idx}_fallback.png")
            self.draw_puzzle(puzzle, image_filename)
            
            # Format solution for comparison
            solution_text = self.format_solution(solution)
            
            # 保持和正常情况一致的格式
            return {
                "image": [image_filename],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{self.prompt_template}"
                    },
                    {
                        "from": "gpt", 
                        "value": solution_text
                    }
                ],
                "puzzle_board": [puzzle],
                "solution_board": [solution],
                "metadata": [{
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "clues": sum(1 for r in range(self.board_size) for c in range(self.board_size) if puzzle[r][c] != 0),
                    "board_size": self.board_size,
                    "fallback": True
                }]
            }

    def format_solution(self, solution: Board) -> str:
        """Format solution board into the required answer format"""
        # Format as a nested list structure
        solution_str = str(solution)
        
        size = self.board_size
        block_desc = ""
        if self.block_size > 1:
            block_desc = f"3. Each {self.block_size}×{self.block_size} box must contain digits 1-{size} exactly once"
        else:
            block_desc = "3. (No block constraints for this size)"
            
        return f"""<think>
Let me analyze this {size}×{size} Sudoku puzzle step by step.

I need to fill in the empty cells following the Sudoku rules:
1. Each row must contain digits 1-{size} exactly once
2. Each column must contain digits 1-{size} exactly once  
{block_desc}

I'll work through this systematically, looking for cells where only one number can fit based on the constraints from rows, columns, and blocks.

After careful analysis and applying logical deduction, I can determine the complete solution.
</think>
<answer>
{solution_str}
</answer>"""

    def calculate_reward(self, predicted_text: str, ground_truth: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Calculate reward based on three criteria:
        1. Format correctness
        2. Matching given numbers
        3. Sudoku rules satisfaction
        
        Note: If given numbers are not perfectly preserved, rules score will be 0.
        
        Returns a tuple of (format_score, given_numbers_score, sudoku_rules_score)
        """
        solution_board = ground_truth["solution_board"]
        puzzle_board = ground_truth["puzzle_board"]
        
        # 1. Check format correctness
        format_score = self._check_format(predicted_text)
        
        if format_score < 1.0:
            # If format is wrong, other scores are 0
            return (format_score, 0.0, 0.0)
            
        # Extract predicted grid from response
        predicted_grid = self._extract_grid(predicted_text)
        if predicted_grid is None:
            return (format_score, 0.0, 0.0)
            
        # 2. Check if given numbers match
        given_numbers_score = self._check_given_numbers(predicted_grid, puzzle_board)
        
        # 3. Check if sudoku rules are satisfied
        # But only if given numbers are perfectly preserved (score = 1.0)
        if given_numbers_score < 1.0:
            sudoku_rules_score = 0.0
        else:
            sudoku_rules_score = self._check_sudoku_rules(predicted_grid)
        
        return (format_score, given_numbers_score, sudoku_rules_score)

    def _check_format(self, text: str) -> float:
        """Check if the response format is correct"""
        # Check for <think> and <answer> blocks
        if "<think>" not in text or "</think>" not in text:
            return 0.0
        if "<answer>" not in text or "</answer>" not in text:
            return 0.0
            
        # Extract answer block
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not answer_match:
            return 0.0
            
        answer_content = answer_match.group(1).strip()
        
        # Check if it looks like a board_size x board_size nested list
        # Pattern: [[...], [...], ..., [...]] with board_size sublists
        size = self.board_size
        if size == 1:
            # Special case for 1x1
            pattern = r'^\[\s*\[\s*\d+\s*\]\s*\]$'
        else:
            # General case
            pattern = rf'^\[\s*\[\s*\d+(?:\s*,\s*\d+){{{size-1}}}\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+){{{size-1}}}\s*\]){{{size-1}}}\s*\]$'
        
        if re.match(pattern, answer_content):
            return 1.0
        else:
            return 0.0

    def _extract_grid(self, text: str) -> Optional[Board]:
        """Extract 9x9 grid from formatted response"""
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not answer_match:
            return None
            
        answer_content = answer_match.group(1).strip()
        
        try:
            # Try to evaluate the string as a Python list
            grid = eval(answer_content)
            
            # Validate the structure
            if not isinstance(grid, list) or len(grid) != self.board_size:
                return None
                
            for row in grid:
                if not isinstance(row, list) or len(row) != self.board_size:
                    return None
                for cell in row:
                    if not isinstance(cell, int) or cell < 1 or cell > self.board_size:
                        return None
            
            return grid
            
        except:
            # If eval fails, try alternative parsing methods
            import json
            try:
                # Try parsing as JSON
                grid = json.loads(answer_content)
                
                # Validate the structure
                if not isinstance(grid, list) or len(grid) != self.board_size:
                    return None
                    
                for row in grid:
                    if not isinstance(row, list) or len(row) != self.board_size:
                        return None
                    for cell in row:
                        if not isinstance(cell, int) or cell < 1 or cell > self.board_size:
                            return None
                
                return grid
                
            except:
                return None

    def _check_given_numbers(self, predicted_grid: Board, puzzle_board: Board) -> float:
        """Check if predicted grid preserves the given numbers from puzzle"""
        try:
            # Validate puzzle_board format
            if not isinstance(puzzle_board, list) or len(puzzle_board) != self.board_size:
                print(f"Warning: puzzle_board format error - not {self.board_size}x{self.board_size} list, got type: {type(puzzle_board)}, length: {len(puzzle_board) if isinstance(puzzle_board, list) else 'N/A'}")
                return 0.0
            
            for i, row in enumerate(puzzle_board):
                if not isinstance(row, list) or len(row) != self.board_size:
                    print(f"Warning: puzzle_board row {i} format error - not {self.board_size}-element list, got type: {type(row)}, length: {len(row) if isinstance(row, list) else 'N/A'}")
                    return 0.0
            
            # Validate predicted_grid format
            if not isinstance(predicted_grid, list) or len(predicted_grid) != self.board_size:
                print(f"Warning: predicted_grid format error - not {self.board_size}x{self.board_size} list")
                return 0.0
            
            for i, row in enumerate(predicted_grid):
                if not isinstance(row, list) or len(row) != self.board_size:
                    print(f"Warning: predicted_grid row {i} format error - not {self.board_size}-element list")
                    return 0.0
            
            correct = 0
            total_given = 0
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if puzzle_board[r][c] != 0:  # Given number
                        total_given += 1
                        if predicted_grid[r][c] == puzzle_board[r][c]:
                            correct += 1
                            
            return correct / total_given if total_given > 0 else 1.0
            
        except Exception as e:
            print(f"Error in _check_given_numbers: {e}")
            print(f"puzzle_board type: {type(puzzle_board)}")
            print(f"predicted_grid type: {type(predicted_grid)}")
            return 0.0

    def _check_sudoku_rules(self, grid: Board) -> float:
        """Check if grid satisfies all Sudoku rules"""
        errors = 0
        size = self.board_size
        expected_numbers = set(range(1, size + 1))
        
        # Calculate total checks: rows + columns + boxes (if applicable)
        total_checks = size + size  # rows + columns
        if self.block_size > 1:
            num_blocks = (size // self.block_size) ** 2
            total_checks += num_blocks
        
        # Check rows
        for r in range(size):
            if len(set(grid[r])) != size or set(grid[r]) != expected_numbers:
                errors += 1
                
        # Check columns  
        for c in range(size):
            column = [grid[r][c] for r in range(size)]
            if len(set(column)) != size or set(column) != expected_numbers:
                errors += 1
                
        # Check blocks (only if block_size > 1)
        if self.block_size > 1:
            for box_r in range(0, size, self.block_size):
                for box_c in range(0, size, self.block_size):
                    box = []
                    for r in range(box_r, min(box_r + self.block_size, size)):
                        for c in range(box_c, min(box_c + self.block_size, size)):
                            box.append(grid[r][c])
                    if len(set(box)) != len(box) or not set(box).issubset(expected_numbers):
                        errors += 1
                    
        return (total_checks - errors) / total_checks


# Convenience function for integration with dynamic dataset
def create_sudoku_data_generator(clues_range: Tuple[int, int] = (25, 35), 
                               output_dir: str = "/tmp/sudoku_images",
                               board_size: int = 9):
    """Create a sudoku data generator function for dynamic dataset"""
    generator = SudokuGenerator(clues_range=clues_range, board_size=board_size)
    
    def data_generator(epoch: int, sample_idx: int) -> Dict[str, Any]:
        try:
            
            result = generator.generate_sample(epoch, sample_idx, output_dir)
            if result is None:
                print(f"Failed to generate sample {sample_idx} in epoch {epoch}: None result")
                # This should never happen with the updated generate_sample, but just in case
                raise ValueError("None result from generate_sample")
            return result
        except Exception as e:
            print(f"Failed to generate sample {sample_idx} in epoch {epoch}: {e}")
            # Last resort fallback - create a simple valid sample
            fallback_puzzle, fallback_solution = generator._create_fallback_puzzle(30)
            fallback_file = os.path.join(output_dir, f"sudoku_epoch_{epoch}_sample_{sample_idx}_emergency.png")
            os.makedirs(output_dir, exist_ok=True)
            generator.draw_puzzle(fallback_puzzle, fallback_file)
            
            return {
                "image": [fallback_file],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{generator.prompt_template}"
                    },
                    {
                        "from": "gpt", 
                        "value": generator.format_solution(fallback_solution)
                    }
                ],
                "puzzle_board": [fallback_puzzle],
                "solution_board": [fallback_solution],
                "metadata": [{
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "clues": sum(1 for r in range(generator.board_size) for c in range(generator.board_size) if fallback_puzzle[r][c] != 0),
                    "board_size": generator.board_size,
                    "emergency_fallback": True
                }]
            }
    
    # Create three separate reward functions for the three components
    def sudoku_format_reward_func(prompts, completions, **kwargs):
        """Sudoku format correctness reward function"""
        return _calculate_sudoku_component_rewards(generator, completions, kwargs, component_index=0)
    
    def sudoku_given_numbers_reward_func(prompts, completions, **kwargs):
        """Sudoku given numbers preservation reward function"""
        return _calculate_sudoku_component_rewards(generator, completions, kwargs, component_index=1)
    
    def sudoku_rules_reward_func(prompts, completions, **kwargs):
        """Sudoku rules satisfaction reward function"""
        return _calculate_sudoku_component_rewards(generator, completions, kwargs, component_index=2)
    
    return data_generator, (sudoku_format_reward_func, sudoku_given_numbers_reward_func, sudoku_rules_reward_func)

def _calculate_sudoku_component_rewards(generator, completions, kwargs, component_index):
    """Helper function to calculate specific component rewards for a batch"""
    try:
        # Get the sudoku metadata from kwargs
        if 'puzzle_board' not in kwargs or 'solution_board' not in kwargs:
            print(f"Warning: Missing puzzle_board or solution_board in kwargs for component {component_index}")
            return [0.0] * len(completions)
        
        # Extract batch data
        puzzle_boards = kwargs['puzzle_board']
        solution_boards = kwargs['solution_board']
        
        # Calculate rewards for each sample in the batch
        rewards = []
        for i, (completion, puzzle_board, solution_board) in enumerate(zip(completions, puzzle_boards, solution_boards)):
            try:
                # Extract content from completion
                if isinstance(completion, list) and len(completion) > 0:
                    content = completion[0].get("content", "")
                else:
                    content = str(completion)
                
                # Create ground truth data structure
                ground_truth = {
                    'puzzle_board': puzzle_board,
                    'solution_board': solution_board
                }
                
                # Calculate detailed rewards - returns tuple of (format, given_numbers, rules)
                reward_tuple = generator.calculate_reward(content, ground_truth)
                
                # Return the specific component score
                rewards.append(reward_tuple[component_index])
                
            except Exception as e:
                print(f"Error calculating component {component_index} reward for sample {i}: {e}")
                rewards.append(0.0)
        
        return rewards
        
    except Exception as e:
        print(f"Error in sudoku component {component_index} reward: {e}")
        return [0.0] * len(completions)


if __name__ == "__main__":
    # Test the generator
    generator = SudokuGenerator()
    
    # Generate a sample
    sample = generator.generate_sample(epoch=10000, sample_idx=20)
    print("Generated sample:")
    print(f"Image: {sample['image']}")
    print(f"Clues: {sample['metadata'][0]['clues']}")
    print(f"Conversation: {sample['conversations'][0]['value'][:100]}...")
    
    # Test reward calculation
    sample_response = """<think>
Let me solve this step by step by analyzing the constraints.
</think>
<answer>
[[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8, 9, 1], [5, 6, 7, 8, 9, 1, 2, 3, 4], [8, 9, 1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9, 1, 2], [6, 7, 8, 9, 1, 2, 3, 4, 5], [9, 1, 2, 3, 4, 5, 6, 7, 8]]
</answer>"""
    
    ground_truth = {
        'puzzle_board': sample['puzzle_board'][0],
        'solution_board': sample['solution_board'][0]
    }
    reward_tuple = generator.calculate_reward(sample_response, ground_truth)
    print(f"Reward scores: format={reward_tuple[0]}, given_numbers={reward_tuple[1]}, rules={reward_tuple[2]}") 