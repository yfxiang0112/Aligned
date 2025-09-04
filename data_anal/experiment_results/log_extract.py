import re
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

def weighted_mean(f1_data,f1_kb,w):
    p_integrate = 2
    return (w * (f1_data ** -p_integrate)\
            + (1.-w) * (f1_kb ** -p_integrate)) ** (-1/p_integrate)

def extract_f1_scores_from_logs(log_directory="./logs", file_pattern="*.log"):
    """
    Extract F1 scores from log files and organize them by score type.
    
    Args:
        log_directory: Directory containing log files
        file_pattern: Pattern to match log files
    
    Returns:
        Dictionary with F1 score types as keys and lists of values from all files
    """
    # Find all log files
    log_files = glob.glob(os.path.join(log_directory, file_pattern))
    log_files.sort()  # Sort for consistent ordering
    
    if not log_files:
        print(f"No log files found in {log_directory} with pattern {file_pattern}")
        return {}
    
    print(f"Found {len(log_files)} log files: {[os.path.basename(f) for f in log_files]}")
    
    # Dictionary to store all F1 scores
    f1_scores = defaultdict(list)
    
    # Regex pattern to match F1 scores
    pattern = r'([ \w]*f1[ \w]*): *(\d\.\d*)([\(\)\w ]*)'
    
    # Process each log file
    for file_idx, log_file in enumerate(log_files, 1):
        print(f"\nProcessing file {file_idx}/{len(log_files)}: {os.path.basename(log_file)}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all F1 score matches
            matches = re.findall(pattern, content, re.IGNORECASE)

            counter = 0
            
            if not matches:
                print(f"  No F1 scores found in {os.path.basename(log_file)}")
                continue
            
            # Process each match
            for score_type, score_value, additional in matches:
                # Convert to float and store
                try:
                    score_float = float(score_value)
                    # Handle duplicate keys by appending index if needed
                    
                    # Check if this exact key already exists for this file
                    #while any(final_key in key and key.endswith(f"_{file_idx}") 
                    #         for key in f1_scores.keys()):
                    final_key = f"{(score_type + additional).lower()}_{counter // 4}"
                    counter += 1
                    
                    ## Add file index to make keys unique across files
                    #final_key_with_file = f"{final_key}_file{file_idx}"
                    f1_scores[final_key].append(score_float)
                    
                    #print(f"  Found: {final_key} = {score_float:.4f}")
                    
                except ValueError:
                    print(f"  Could not convert score '{score_value}' to float")
        
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
    
    return f1_scores

def analyze_f1_scores(f1_scores):
    """
    Analyze F1 scores and calculate min/max for each score type.
    
    Args:
        f1_scores: Dictionary with F1 scores
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Group scores by type (ignoring file suffix)
    score_groups = defaultdict(list)
    
    for key, values in f1_scores.items():
        # Extract base score type (remove _fileX suffix)
        base_key = re.sub(r'_file\d+$', '', key)
        score_groups[base_key].extend(values)
    
    # Calculate statistics for each score type
    for score_type, all_values in score_groups.items():
        if all_values:
            analysis[score_type] = {
                'values': all_values,
                'min': min(all_values),
                'max': max(all_values),
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'count': len(all_values)
            }
    
    return analysis

def print_analysis_results(analysis):
    """
    Print the analysis results in a formatted way.
    """
    print("\n" + "="*60)
    print("F1 SCORE ANALYSIS RESULTS")
    print("="*60)
    
    for score_type, stats in analysis.items():
        print(f"\n{score_type.upper()}:")
        print(f"  Values: {[f'{v:.4f}' for v in stats['values']]}")
        print(f"  Count:  {stats['count']}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")

def save_results_to_file(analysis, output_file="f1_analysis_results.txt"):
    """
    Save analysis results to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("F1 Score Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        for score_type, stats in analysis.items():
            f.write(f"{score_type.upper()}:\n")
            f.write(f"  Values: {[f'{v:.4f}' for v in stats['values']]}\n")
            f.write(f"  Count:  {stats['count']}\n")
            f.write(f"  Min:    {stats['min']:.4f}\n")
            f.write(f"  Max:    {stats['max']:.4f}\n")
            f.write(f"  Mean:   {stats['mean']:.4f}\n")
            f.write(f"  Std:    {stats['std']:.4f}\n\n")
    
    print(f"\nResults saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Configuration
    LOG_DIRECTORY = "data_anal/experiment_results/hsa_aug30/"  # Current directory, change as needed
    FILE_PATTERN = "log_MLP_*"  # Pattern for log files

    w_hsa = .3233
    w_eco = .4231
    
    # Extract F1 scores from all log files
    print("Starting F1 score extraction...")
    f1_scores = extract_f1_scores_from_logs(LOG_DIRECTORY, FILE_PATTERN)

    
    if not f1_scores:
        print("No F1 scores found in any log files.")
        exit()

    keys = ['init', 'data_only',\
            'ABL1_refl', 'ABL1_refine', 'ABL2_refl', 'ABL2_refine']

    score_neur = {}
    score_intg = {}
    for i,k in enumerate(keys):

        data_con = .5*(max(f1_scores[f'f1 on test_{i*2}']) + min(f1_scores[f'f1 on test_{i*2}']))
        kb_con = .5*(max(f1_scores[f'f1 on kb_{i*2}']) + min(f1_scores[f'f1 on kb_{i*2}']))
        data_tol = .5*(max(f1_scores[f'f1 on test_{i*2}']) - min(f1_scores[f'f1 on test_{i*2}']))
        kb_tol = .5*(max(f1_scores[f'f1 on kb_{i*2}']) - min(f1_scores[f'f1 on kb_{i*2}']))

        balanced = [weighted_mean(d,k, w_hsa) for d,k in zip(f1_scores[f'f1 on test_{i*2}'], f1_scores[f'f1 on kb_{i*2}'])]
        bal_con = .5*(max(balanced)+min(balanced))
        bal_tol = .5*(max(balanced)-min(balanced))
        score_neur[k] = {'data consistency':f'{data_con:.4f} +- {data_tol:.4f}',\
                'kb consistency': f'{kb_con:.4f} +- {kb_tol:.4f}',\
                'balanced consistency': f'{bal_con:.4f} +- {bal_tol:.4f}'}

        data_con = .5*(max(f1_scores[f'f1 on test_{i*2+1}']) + min(f1_scores[f'f1 on test_{i*2+1}']))
        kb_con = .5*(max(f1_scores[f'f1 on kb_{i*2+1}']) + min(f1_scores[f'f1 on kb_{i*2+1}']))
        data_tol = .5*(max(f1_scores[f'f1 on test_{i*2+1}']) - min(f1_scores[f'f1 on test_{i*2+1}']))
        kb_tol = .5*(max(f1_scores[f'f1 on kb_{i*2+1}']) - min(f1_scores[f'f1 on kb_{i*2+1}']))

        balanced = [weighted_mean(d,k, w_hsa) for d,k in zip(f1_scores[f'f1 on test_{i*2+1}'], f1_scores[f'f1 on kb_{i*2+1}'])]
        bal_con = .5*(max(balanced)+min(balanced))
        bal_tol = .5*(max(balanced)-min(balanced))
        score_intg[k] = {'data consistency':f'{data_con:.4f} +- {data_tol:.4f}',\
                'kb consistency': f'{kb_con:.4f} +- {kb_tol:.4f}',\
                'balanced consistency': f'{bal_con:.4f} +- {bal_tol:.4f}'}
    
    print(pd.DataFrame(score_neur).transpose())
    print(pd.DataFrame(score_intg).transpose())
    ##print(f1_scores.keys())
    ## Analyze the scores
    #print("\nAnalyzing F1 scores...")
    #analysis = analyze_f1_scores(f1_scores)
    #
    ## Print results
    #print_analysis_results(analysis)
    #
    ## Save results to file
    #save_results_to_file(analysis)
    #
    ## Additional summary
    ##print("\n" + "="*60)
    ##print("SUMMARY ACROSS ALL F1 SCORE TYPES:")
    ##print("="*60)
    #
    #all_values = []
    #for stats in analysis.values():
    #    all_values.extend(stats['values'])
    #
    #if all_values:
    #    print(f"Overall Min F1: {min(all_values):.4f}")
    #    print(f"Overall Max F1: {max(all_values):.4f}")
    #    print(f"Overall Mean F1: {np.mean(all_values):.4f}")
    #    print(f"Total F1 scores found: {len(all_values)}")
