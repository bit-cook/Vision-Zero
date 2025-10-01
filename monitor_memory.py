#!/usr/bin/env python3
"""
Memory monitoring script for CLEVR spot-the-difference training
"""

import torch
import time
import sys
import os
import psutil
from typing import Dict, List, Optional

class MemoryMonitor:
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.memory_history = []
        self.start_time = time.time()
        
    def log_memory(self, stage: str = ""):
        """Log current memory usage"""
        current_time = time.time() - self.start_time
        
        # GPU memory
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                gpu_memory[device] = {
                    "allocated": allocated,
                    "reserved": reserved,
                    "max_allocated": max_allocated
                }
        
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**3  # GB
        
        memory_info = {
            "timestamp": current_time,
            "stage": stage,
            "gpu_memory": gpu_memory,
            "cpu_memory": cpu_memory
        }
        
        self.memory_history.append(memory_info)
        
        # Print to console
        print(f"[MEMORY] {stage} (t={current_time:.1f}s)")
        print(f"  CPU: {cpu_memory:.2f} GB")
        for device, mem in gpu_memory.items():
            print(f"  {device}: Allocated={mem['allocated']:.2f}GB, Reserved={mem['reserved']:.2f}GB, Max={mem['max_allocated']:.2f}GB")
        print()
        
        # Log to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"[MEMORY] {stage} (t={current_time:.1f}s)\n")
                f.write(f"  CPU: {cpu_memory:.2f} GB\n")
                for device, mem in gpu_memory.items():
                    f.write(f"  {device}: Allocated={mem['allocated']:.2f}GB, Reserved={mem['reserved']:.2f}GB, Max={mem['max_allocated']:.2f}GB\n")
                f.write("\n")
    
    def check_memory_growth(self, threshold_gb: float = 1.0) -> bool:
        """Check if memory usage has grown significantly"""
        if len(self.memory_history) < 2:
            return False
        
        latest = self.memory_history[-1]
        initial = self.memory_history[0]
        
        for device in latest["gpu_memory"]:
            growth = latest["gpu_memory"][device]["allocated"] - initial["gpu_memory"][device]["allocated"]
            if growth > threshold_gb:
                print(f"[WARNING] Memory growth detected on {device}: +{growth:.2f}GB")
                return True
        
        return False
    
    def reset_peak_memory(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def generate_report(self) -> str:
        """Generate a memory usage report"""
        if not self.memory_history:
            return "No memory data collected"
        
        report = ["Memory Usage Report", "=" * 50]
        
        # Peak usage
        max_gpu_memory = {}
        for entry in self.memory_history:
            for device, mem in entry["gpu_memory"].items():
                if device not in max_gpu_memory or mem["allocated"] > max_gpu_memory[device]:
                    max_gpu_memory[device] = mem["allocated"]
        
        report.append("Peak GPU Memory Usage:")
        for device, peak in max_gpu_memory.items():
            report.append(f"  {device}: {peak:.2f} GB")
        
        # Memory growth analysis
        if len(self.memory_history) >= 2:
            report.append("\nMemory Growth Analysis:")
            initial = self.memory_history[0]
            final = self.memory_history[-1]
            
            for device in initial["gpu_memory"]:
                initial_mem = initial["gpu_memory"][device]["allocated"]
                final_mem = final["gpu_memory"][device]["allocated"]
                growth = final_mem - initial_mem
                report.append(f"  {device}: {initial_mem:.2f}GB → {final_mem:.2f}GB (Δ{growth:+.2f}GB)")
        
        # Recommendations
        report.append("\nRecommendations:")
        if max_gpu_memory:
            max_usage = max(max_gpu_memory.values())
            if max_usage > 70:  # Assuming 80GB GPU
                report.append("  ⚠️  High memory usage detected - consider reducing batch size")
            elif max_usage > 60:
                report.append("  ⚠️  Moderate memory usage - monitor for stability")
            else:
                report.append("  ✅ Memory usage appears reasonable")
        
        return "\n".join(report)


def analyze_clevr_memory_usage():
    """Analyze memory usage patterns specific to CLEVR training"""
    print("CLEVR Spot-the-Difference Memory Analysis")
    print("=" * 50)
    
    monitor = MemoryMonitor("clevr_memory_log.txt")
    
    # Simulate CLEVR training stages
    print("Simulating CLEVR training memory usage patterns...")
    
    monitor.log_memory("Initial state")
    
    # Simulate model loading
    print("Loading model (simulated)...")
    time.sleep(1)
    monitor.log_memory("Model loaded")
    
    # Simulate data processing
    print("Processing game data (simulated)...")
    time.sleep(1)
    monitor.log_memory("Data processed")
    
    # Simulate clue generation
    print("Generating clues (simulated)...")
    # Create some tensors to simulate memory usage
    if torch.cuda.is_available():
        dummy_tensors = []
        for i in range(4):  # 4 players
            tensor = torch.randn(1000, 1000).cuda()  # Simulate model activations
            dummy_tensors.append(tensor)
    
    monitor.log_memory("Clues generated")
    
    # Simulate decision generation
    print("Generating decisions (simulated)...")
    if torch.cuda.is_available():
        for i in range(4):  # 4 players
            tensor = torch.randn(800, 800).cuda()  # More activations
            dummy_tensors.append(tensor)
    
    monitor.log_memory("Decisions generated")
    
    # Simulate reward calculation
    print("Calculating rewards (simulated)...")
    monitor.log_memory("Rewards calculated")
    
    # Cleanup
    print("Cleaning up memory...")
    if torch.cuda.is_available():
        del dummy_tensors
    monitor.cleanup_memory()
    monitor.log_memory("After cleanup")
    
    # Generate report
    print("\n" + monitor.generate_report())
    
    # Memory growth check
    if monitor.check_memory_growth(0.5):
        print("\n⚠️  Potential memory leak detected!")
    else:
        print("\n✅ No significant memory growth detected")


def get_memory_recommendations():
    """Provide memory optimization recommendations"""
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
    
    recommendations = [
        "Memory Optimization Recommendations for CLEVR Training:",
        "=" * 60,
        "",
        "1. Batch Size Optimization:",
        "   - Start with per_device_train_batch_size=1",
        "   - Use gradient_accumulation_steps=2-4 for effective larger batches",
        "   - For InternVL3-8B: max batch_size * sequence_length < 8000",
        "",
        "2. Sequence Length:",
        "   - max_prompt_length: 3000-5000 tokens",
        "   - max_completion_length: 500-1000 tokens",
        "   - Reduce max_anyres_num to 2-4 for images",
        "",
        "3. Model Configuration:",
        "   - Enable gradient_checkpointing=true",
        "   - Use bf16=true for memory efficiency",
        "   - Set attn_implementation=flash_attention_2",
        "",
        "4. CLEVR-specific optimizations:",
        "   - Process only 1 game per device to avoid OOM",
        "   - Use num_generations=1 initially",
        "   - Enable group normalization for stable training",
        "",
        "5. Memory Management:",
        "   - Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "   - Monitor memory usage with debug logs",
        "   - Clean up cache between training steps",
        "",
        "6. If still experiencing OOM:",
        "   - Reduce num_players from 4 to 3",
        "   - Reduce num_rounds from 2 to 1",
        "   - Use smaller model (InternVL3-2B instead of 8B)",
    ]
    
    print("\n".join(recommendations))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_clevr_memory_usage()
    elif len(sys.argv) > 1 and sys.argv[1] == "recommendations":
        get_memory_recommendations()
    else:
        print("Usage:")
        print("  python monitor_memory.py analyze         - Run memory usage analysis")
        print("  python monitor_memory.py recommendations - Show optimization recommendations")
        print("")
        get_memory_recommendations() 