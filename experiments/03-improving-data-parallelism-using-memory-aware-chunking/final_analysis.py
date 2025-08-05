#!/usr/bin/env python3
import json
import os
import glob
from collections import defaultdict
import statistics

def parse_profile_filename(filename):
    """Parse profile filename to extract configuration details."""
    parts = filename.replace('.json', '').split('-')
    return {
        'inlines': int(parts[0]),
        'xlines': int(parts[1]),
        'samples': int(parts[2]),
        'chunking_mode': parts[3],
        'workers': int(parts[4]),
        'config_key': f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[4]}"
    }

def analyze_profiles_with_ooms(profile_dir):
    """Analyze profiles accounting for missing files as OOMs."""
    json_files = glob.glob(os.path.join(profile_dir, '*.json'))
    expected_per_mode = 768  # 256 configurations × 3 repetitions
    
    # Count actual files per mode
    mode_counts = defaultdict(int)
    successful_profiles = []
    
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        config = parse_profile_filename(filename)
        mode = config['chunking_mode']
        mode_counts[mode] += 1
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        exec_time = data.get('data', {}).get('execution_time')
        memory_data = data.get('data', {}).get('memory_usage', {})
        
        # Get total peak memory
        total_peak = 0
        for worker_id, mem_info in memory_data.items():
            if isinstance(mem_info, dict) and 'peak_memory_usage' in mem_info:
                total_peak += mem_info['peak_memory_usage']
        
        successful_profiles.append({
            **config,
            'execution_time': exec_time,
            'total_peak_memory_bytes': total_peak
        })
    
    # Calculate OOMs as missing files
    results = {}
    for mode in ['auto', 'evenly_split', 'memaware']:
        successful = mode_counts.get(mode, 0)
        oom_count = expected_per_mode - successful
        
        mode_profiles = [p for p in successful_profiles if p['chunking_mode'] == mode]
        
        if mode_profiles:
            exec_times = [p['execution_time'] for p in mode_profiles]
            memories = [p['total_peak_memory_bytes'] for p in mode_profiles]
            
            results[mode] = {
                'total': expected_per_mode,
                'successful': successful,
                'ooms': oom_count,
                'oom_rate': oom_count / expected_per_mode,
                'median_time': statistics.median(exec_times),
                'median_memory_gb': statistics.median(memories) / (1024**3)
            }
        else:
            results[mode] = {
                'total': expected_per_mode,
                'successful': successful,
                'ooms': oom_count,
                'oom_rate': oom_count / expected_per_mode,
                'median_time': None,
                'median_memory_gb': None
            }
    
    # Find common successful configurations
    config_modes = defaultdict(set)
    config_profiles = defaultdict(list)
    
    for profile in successful_profiles:
        key = profile['config_key']
        config_modes[key].add(profile['chunking_mode'])
        config_profiles[key].append(profile)
    
    common_profiles = []
    for key, modes in config_modes.items():
        if len(modes) == 3:
            common_profiles.extend(config_profiles[key])
    
    # Stats for common configs
    common_stats = {}
    for mode in ['auto', 'evenly_split', 'memaware']:
        mode_common = [p for p in common_profiles if p['chunking_mode'] == mode]
        if mode_common:
            times = [p['execution_time'] for p in mode_common]
            mems = [p['total_peak_memory_bytes'] for p in mode_common]
            common_stats[mode] = {
                'runs': len(mode_common),
                'median_time': statistics.median(times),
                'median_memory_gb': statistics.median(mems) / (1024**3)
            }
    
    return results, common_stats

def print_final_results():
    """Print the final corrected statistics."""
    base_dir = os.path.abspath("out/results/20250413173525")
    profile_dir = os.path.join(base_dir, "profiles")
    
    results, common_stats = analyze_profiles_with_ooms(profile_dir)
    
    print("\n" + "="*80)
    print("FINAL CORRECTED STATISTICS FROM JSON PROFILES")
    print("="*80)
    
    print("\n1. TABLE 1 - Out-of-memory incidents by chunking strategy")
    print("-"*80)
    print(f"{'Chunking Mode':<15} {'OOM Failures':<15} {'Successful':<15} {'Total':<15} {'OOM Rate':<15}")
    print("-"*80)
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        r = results[mode]
        print(f"{mode:<15} {r['ooms']:<15} {r['successful']:<15} {r['total']:<15} {r['oom_rate']:<15.1%}")
    
    print("\n2. TABLE 2 - Performance comparison (runs where all strategies succeeded)")
    print("-"*80)
    print(f"{'Chunking Mode':<15} {'Time (s)':<15} {'Memory (GB)':<15} {'Runs':<15}")
    print("-"*80)
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        if mode in common_stats:
            s = common_stats[mode]
            print(f"{mode:<15} {s['median_time']:<15.1f} {s['median_memory_gb']:<15.2f} {s['runs']:<15}")
    
    print("\n3. CORRECTIONS NEEDED IN PAPER:")
    print("-"*80)
    print("Table 1 - Update OOM counts:")
    print("  - evenly_split: Change from 128 to 243 OOMs")
    print("  - evenly_split: Change OOM rate from 16.7% to 31.6%")
    print("\nTable 2 - Update memory usage values:")
    print("  - All modes: Use the GB values shown above")
    print("  - Note: Memory values in paper appear to be per-worker, not total")

if __name__ == "__main__":
    print_final_results()