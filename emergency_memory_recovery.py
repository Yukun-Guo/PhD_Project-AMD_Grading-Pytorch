#!/usr/bin/env python3
"""
Emergency CUDA memory recovery script.
Run this when experiencing severe memory fragmentation or OOM errors.
"""

import os
import sys
import subprocess
import time

def emergency_memory_recovery():
    """Perform emergency memory recovery."""
    print("🚨 EMERGENCY CUDA MEMORY RECOVERY")
    print("=" * 50)
    
    try:
        import torch
        import gc
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        print("📊 Memory status before recovery:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        
        print("\n🔧 Performing emergency cleanup...")
        
        # Step 1: Multiple aggressive cache clearing
        print("1. Aggressive cache clearing...")
        for round_num in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.1)  # Small delay
            torch.cuda.empty_cache()
            print(f"   Round {round_num + 1}/5 completed")
        
        # Step 2: Reset all memory statistics
        print("2. Resetting memory statistics...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Step 3: Force synchronization
        print("3. Forcing CUDA synchronization...")
        torch.cuda.synchronize()
        
        # Step 4: Try to create and destroy large tensors to defragment
        print("4. Attempting memory defragmentation...")
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                reserved = torch.cuda.memory_reserved(i)
                available = (total_memory - reserved) * 0.5  # Use 50% of available
                
                if available > 1024**3:  # At least 1GB
                    tensor_size = int(available / 4)  # 4 bytes per float32
                    print(f"   Creating {available/1024**3:.2f}GB tensor on GPU {i}...")
                    temp_tensor = torch.zeros(tensor_size, device=f'cuda:{i}')
                    del temp_tensor
                    torch.cuda.empty_cache()
                    print(f"   GPU {i} defragmentation completed")
                else:
                    print(f"   GPU {i} insufficient memory for defragmentation")
                    
            except Exception as e:
                print(f"   GPU {i} defragmentation failed: {e}")
        
        # Step 5: Final cleanup
        print("5. Final cleanup...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print("\n📊 Memory status after recovery:")
        total_freed = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            free = total - reserved
            total_freed += free
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
        
        print(f"\n✅ Emergency recovery completed!")
        print(f"💾 Total free memory: {total_freed:.2f}GB")
        
        return total_freed > 1.0  # Return True if we have at least 1GB free total
        
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ Emergency recovery failed: {e}")
        return False

def kill_python_processes():
    """Kill other Python processes that might be using GPU memory."""
    print("\n🔪 Attempting to kill other Python processes...")
    
    try:
        # Get current process PID
        current_pid = os.getpid()
        
        # Find Python processes
        result = subprocess.run(['pgrep', '-f', 'python'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            killed_count = 0
            
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    if pid != current_pid:  # Don't kill ourselves
                        # Check if it's using GPU (nvidia-smi)
                        gpu_check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], 
                                                 capture_output=True, text=True)
                        if str(pid) in gpu_check.stdout:
                            subprocess.run(['kill', str(pid)], check=False)
                            killed_count += 1
                            print(f"   Killed Python process {pid}")
                except:
                    continue
            
            if killed_count > 0:
                print(f"✅ Killed {killed_count} GPU-using Python processes")
                time.sleep(2)  # Wait for processes to die
            else:
                print("ℹ️  No other GPU-using Python processes found")
        else:
            print("ℹ️  No other Python processes found")
            
    except Exception as e:
        print(f"⚠️  Could not kill processes: {e}")

def restart_cuda_context():
    """Restart CUDA context."""
    print("\n🔄 Restarting CUDA context...")
    
    try:
        import torch
        
        # Get current devices
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            # Reset all devices
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
            print(f"✅ Reset {device_count} CUDA devices")
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ CUDA context restart failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Emergency CUDA memory recovery')
    parser.add_argument('--kill-processes', action='store_true', help='Kill other Python processes using GPU')
    parser.add_argument('--restart-cuda', action='store_true', help='Restart CUDA context')
    parser.add_argument('--full-recovery', action='store_true', help='Perform full recovery (all steps)')
    
    args = parser.parse_args()
    
    success = True
    
    if args.kill_processes or args.full_recovery:
        kill_python_processes()
    
    if args.restart_cuda or args.full_recovery:
        success &= restart_cuda_context()
    
    # Always do emergency memory recovery
    success &= emergency_memory_recovery()
    
    if success:
        print("\n🎉 Emergency recovery successful!")
        print("You can now try running your CUDA program again.")
    else:
        print("\n💀 Emergency recovery failed!")
        print("Consider restarting the system or checking for hardware issues.")
    
    sys.exit(0 if success else 1)