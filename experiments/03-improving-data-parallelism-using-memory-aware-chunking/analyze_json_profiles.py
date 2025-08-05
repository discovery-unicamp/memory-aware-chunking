#!/usr/bin/env python3
import json
import os
import glob
from collections import defaultdict
import statistics

def parse_profile_filename(filename):
    """Parse profile filename to extract configuration details."""
    # Example: 100-100-100-auto-1-1744660908-603829.json
    parts = filename.replace('.json', '').split('-')
    return {
        'inlines': int(parts[0]),
        'xlines': int(parts[1]),
        'samples': int(parts[2]),
        'chunking_mode': parts[3],
        'workers': int(parts[4]),
        'timestamp': int(parts[5]),
        'session_id': parts[6],
        'config_key': f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[4]}"  # shape-workers
    }

def analyze_all_profiles(profile_dir):
    """Analyze all JSON profile files directly."""
    json_files = glob.glob(os.path.join(profile_dir, '*.json'))
    print(f"Found {len(json_files)} JSON profile files")
    
    # Data structures for analysis
    all_profiles = []
    failed_profiles = []
    mode_data = defaultdict(lambda: {
        'total_attempts': 0,
        'successful': [],
        'failed': 0,
        'exec_times': [],
        'peak_memories': []
    })
    
    # Process each JSON file
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        config = parse_profile_filename(filename)
        mode = config['chunking_mode']
        mode_data[mode]['total_attempts'] += 1
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if this run was successful
            exec_time = data.get('data', {}).get('execution_time')
            memory_data = data.get('data', {}).get('memory_usage', {})
            
            # Determine if this was an OOM/failed run
            if not exec_time or not memory_data:
                mode_data[mode]['failed'] += 1
                failed_profiles.append(config)
                continue
            
            # Get peak memory across all workers
            peak_memory = 0
            total_peak_memory = 0
            for worker_id, mem_info in memory_data.items():
                if isinstance(mem_info, dict) and 'peak_memory_usage' in mem_info:
                    worker_peak = mem_info['peak_memory_usage']
                    peak_memory = max(peak_memory, worker_peak)
                    total_peak_memory += worker_peak
            
            # If no valid memory data, consider it failed
            if peak_memory == 0:
                mode_data[mode]['failed'] += 1
                failed_profiles.append(config)
                continue
            
            # Successful run
            profile_data = {
                **config,
                'execution_time': exec_time,
                'peak_memory_bytes': peak_memory,
                'total_peak_memory_bytes': total_peak_memory
            }
            
            all_profiles.append(profile_data)
            mode_data[mode]['successful'].append(profile_data)
            mode_data[mode]['exec_times'].append(exec_time)
            mode_data[mode]['peak_memories'].append(total_peak_memory)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            mode_data[mode]['failed'] += 1
            failed_profiles.append(config)
    
    return all_profiles, failed_profiles, mode_data

def find_common_successful_configs(all_profiles):
    """Find configurations where all three strategies succeeded."""
    # Group successful runs by configuration
    config_modes = defaultdict(set)
    config_runs = defaultdict(list)
    
    for profile in all_profiles:
        config_key = profile['config_key']
        mode = profile['chunking_mode']
        config_modes[config_key].add(mode)
        config_runs[config_key].append(profile)
    
    # Find configs where all three modes succeeded
    common_configs = []
    common_runs = []
    
    for config_key, modes in config_modes.items():
        if len(modes) == 3:  # All three modes present
            common_configs.append(config_key)
            # Get all runs for this config
            for run in config_runs[config_key]:
                common_runs.append(run)
    
    return common_configs, common_runs

def print_analysis_results(mode_data, common_configs, common_runs):
    """Print comprehensive analysis results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PROFILE ANALYSIS FROM JSON FILES")
    print("="*80)
    
    # Expected total runs per mode
    expected_runs = 768  # 256 configurations × 3 repetitions
    
    print("\n1. OUT-OF-MEMORY (OOM) ANALYSIS")
    print("-"*80)
    print(f"{'Mode':<15} {'Total Attempts':<15} {'Successful':<15} {'Failed/OOM':<15} {'OOM Rate':<15}")
    print("-"*80)
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        data = mode_data[mode]
        total = data['total_attempts']
        successful = len(data['successful'])
        failed = data['failed']
        oom_rate = failed / total if total > 0 else 0
        
        print(f"{mode:<15} {total:<15} {successful:<15} {failed:<15} {oom_rate:<15.3f}")
    
    print("\n2. PERFORMANCE STATISTICS (All Successful Runs)")
    print("-"*80)
    print(f"{'Mode':<15} {'Runs':<10} {'Median Time (s)':<20} {'Median Memory (GB)':<20}")
    print("-"*80)
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        data = mode_data[mode]
        if data['exec_times']:
            median_time = statistics.median(data['exec_times'])
            median_mem_gb = statistics.median(data['peak_memories']) / (1024**3)
            runs = len(data['successful'])
            print(f"{mode:<15} {runs:<10} {median_time:<20.2f} {median_mem_gb:<20.2f}")
    
    print("\n3. COMMON SUCCESSFUL CONFIGURATIONS")
    print("-"*80)
    print(f"Configurations where all 3 strategies succeeded: {len(common_configs)}")
    print(f"Total runs in common configs: {len(common_runs)}")
    
    # Calculate stats for common successful runs only
    common_mode_stats = defaultdict(lambda: {'times': [], 'memories': []})
    for run in common_runs:
        mode = run['chunking_mode']
        common_mode_stats[mode]['times'].append(run['execution_time'])
        common_mode_stats[mode]['memories'].append(run['total_peak_memory_bytes'])
    
    print("\n4. PERFORMANCE ON COMMON SUCCESSFUL CONFIGURATIONS")
    print("-"*80)
    print(f"{'Mode':<15} {'Runs':<10} {'Median Time (s)':<20} {'Median Memory (GB)':<20}")
    print("-"*80)
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        stats = common_mode_stats[mode]
        if stats['times']:
            runs = len(stats['times'])
            median_time = statistics.median(stats['times'])
            median_mem_gb = statistics.median(stats['memories']) / (1024**3)
            print(f"{mode:<15} {runs:<10} {median_time:<20.2f} {median_mem_gb:<20.2f}")
    
    # Return the calculated values for paper updates
    results = {
        'oom_stats': {},
        'performance_stats': {},
        'common_performance_stats': {}
    }
    
    for mode in ['auto', 'evenly_split', 'memaware']:
        # OOM stats
        data = mode_data[mode]
        results['oom_stats'][mode] = {
            'failed': data['failed'],
            'successful': len(data['successful']),
            'total': data['total_attempts'],
            'oom_rate': data['failed'] / data['total_attempts'] if data['total_attempts'] > 0 else 0
        }
        
        # Performance stats for all successful runs
        if data['exec_times']:
            results['performance_stats'][mode] = {
                'median_time': statistics.median(data['exec_times']),
                'median_memory_gb': statistics.median(data['peak_memories']) / (1024**3),
                'runs': len(data['successful'])
            }
        
        # Performance stats for common configs only
        stats = common_mode_stats[mode]
        if stats['times']:
            results['common_performance_stats'][mode] = {
                'median_time': statistics.median(stats['times']),
                'median_memory_gb': statistics.median(stats['memories']) / (1024**3),
                'runs': len(stats['times'])
            }
    
    return results

def main():
    base_dir = os.path.abspath("out/results/20250413173525")
    profile_dir = os.path.join(base_dir, "profiles")
    
    # Analyze all profiles
    all_profiles, failed_profiles, mode_data = analyze_all_profiles(profile_dir)
    
    # Find common successful configurations
    common_configs, common_runs = find_common_successful_configs(all_profiles)
    
    # Print and return results
    results = print_analysis_results(mode_data, common_configs, common_runs)
    
    return results

if __name__ == "__main__":
    results = main()