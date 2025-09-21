import re
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

def weighted_mean(f1_data,f1_kb,w):
    p_integrate = 5
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
        #print(f"\nProcessing file {file_idx}/{len(log_files)}: {os.path.basename(log_file)}")
        
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
            metric_num = len(matches) // (6*2)
            for score_type, score_value, additional in matches:
                # Convert to float and store
                try:
                    score_float = float(score_value)
                    # Handle duplicate keys by appending index if needed
                    
                    # Check if this exact key already exists for this file
                    #while any(final_key in key and key.endswith(f"_{file_idx}") 
                    #         for key in f1_scores.keys()):
                    final_key = f"{(score_type + additional).lower()}_{counter // metric_num}"
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
    final_df = []
    for data_name in ['norman', 'dixit', 'adamson', 'ecoli']:
        df_lst = []
        for model_name in ['GNN', 'MLP']:

            LOG_DIRECTORY = f"data_anal/experiment_results/{data_name}/"  # Current directory, change as needed
            FILE_PATTERN = f"log_{model_name}_*"  # Pattern for log files
            print(f'{LOG_DIRECTORY}{FILE_PATTERN}:')

            w_hsa = .5
            w_eco = .4231
            
            # Extract F1 scores from all log files
            print("Starting F1 score extraction...")
            f1_scores = extract_f1_scores_from_logs(LOG_DIRECTORY, FILE_PATTERN)

            
            if not f1_scores:
                print("No F1 scores found in any log files.")
                exit()

            #print(f1_scores)

            keys = ['init', 'data_only',\
                    'ABL1_refl', 'ABL1_refine', 'ABL2_refl', 'ABL2_refine']

            score_neur = {}
            score_intg = {}
            for i,k in enumerate(keys):

                data_mean = .5*(max(f1_scores[f'f1 on test_{i*2}']) + min(f1_scores[f'f1 on test_{i*2}']))
                kb_mean = .5*(max(f1_scores[f'f1 on kb_{i*2}']) + min(f1_scores[f'f1 on kb_{i*2}']))
                data_stde = .5*(max(f1_scores[f'f1 on test_{i*2}']) - min(f1_scores[f'f1 on test_{i*2}']))
                kb_stde = .5*(max(f1_scores[f'f1 on kb_{i*2}']) - min(f1_scores[f'f1 on kb_{i*2}']))

                balanced = [weighted_mean(d,k, w_hsa) for d,k in zip(f1_scores[f'f1 on test_{i*2}'], f1_scores[f'f1 on kb_{i*2}'])]
                bal_mean = .5*(max(balanced)+min(balanced))
                bal_stde = .5*(max(balanced)-min(balanced))
                score_neur[k] = {'data_f1_mean': data_mean, 'data_f1_stde': data_stde,\
                        'kb_f1_mean': kb_mean, 'kb_f1_stde': kb_stde,\
                        'bal_f1_mean': bal_mean, 'bal_f1_stde': bal_stde}
                #print(f'{k} neural scores:\ndata cons: {f1_scores[f"f1 on test_{i*2}"]},\nkb cons:   {f1_scores[f"f1 on kb_{i*2}"]}')

                data_mean = .5*(max(f1_scores[f'f1 on test_{i*2+1}']) + min(f1_scores[f'f1 on test_{i*2+1}']))
                kb_mean = .5*(max(f1_scores[f'f1 on kb_{i*2+1}']) + min(f1_scores[f'f1 on kb_{i*2+1}']))
                data_stde = .5*(max(f1_scores[f'f1 on test_{i*2+1}']) - min(f1_scores[f'f1 on test_{i*2+1}']))
                kb_stde = .5*(max(f1_scores[f'f1 on kb_{i*2+1}']) - min(f1_scores[f'f1 on kb_{i*2+1}']))

                balanced = [weighted_mean(d,k, w_hsa) for d,k in zip(f1_scores[f'f1 on test_{i*2+1}'], f1_scores[f'f1 on kb_{i*2+1}'])]
                bal_mean = .5*(max(balanced)+min(balanced))
                bal_stde = .5*(max(balanced)-min(balanced))
                score_intg[k] = {'data_f1_mean': data_mean, 'data_f1_stde': data_stde,\
                        'kb_f1_mean': kb_mean, 'kb_f1_stde': kb_stde,\
                        'bal_f1_mean': bal_mean, 'bal_f1_stde': bal_stde}
                #print(f'{k} integrated scores:\ndata cons: {f1_scores[f"f1 on test_{i*2+1}"]},\nkb cons:   {f1_scores[f"f1 on kb_{i*2+1}"]}\n')
            
            df_neur = pd.DataFrame(score_neur).transpose().reset_index()
            df_intg = pd.DataFrame(score_intg).transpose().reset_index()
            df_neur.rename({'index':'stage'}, axis=1, inplace=True)
            df_intg.rename({'index':'stage'}, axis=1, inplace=True)
            df_neur.insert(1, 'score', ['neural']*len(df_neur))
            df_neur.insert(1, 'model', [model_name]*len(df_neur))
            df_intg.insert(1, 'score', ['integrated']*len(df_intg))
            df_intg.insert(1, 'model', [model_name]*len(df_intg))

            df_neur.insert(0, 'data_name', [data_name]*len(df_neur))
            df_intg.insert(0, 'data_name', [data_name]*len(df_intg))

            df_lst.append(pd.concat([df_neur, df_intg]))
        df = pd.concat(df_lst)
        final_df.append(df)

    final_df = pd.concat(final_df, axis=0).reset_index(drop=True)
    print(final_df)
    final_df.to_csv(f'data_anal/experiment_results/results.csv')
