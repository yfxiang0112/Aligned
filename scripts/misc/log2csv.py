import re
import csv
from collections import defaultdict

def extract_f1_scores(log_file_path, output_csv_path, abl_loops):
    # Patterns to match different F1 score types
    patterns = { 'macro': re.compile(r'macro f1: ([\d\.]+)'), 'micro': re.compile(r'micro f1: ([\d\.]+)'),
        'weighted': re.compile(r'weighted f1: ([\d\.]+)'),
        'class-1': re.compile(r'class -1 f1: ([\d\.]+)'),
        'class0': re.compile(r'class  0 f1: ([\d\.]+)'),
        'class1': re.compile(r'class  1 f1: ([\d\.]+)')
    }
    
    # Dictionary to store F1 scores grouped by their position in each group
    f1_groups = defaultdict(lambda: defaultdict(list))
    current_group = defaultdict(list)
    group_index = 0
    sample_count = 0
    
    with open(log_file_path, 'r') as file:
        for line in file:
            for f1_type, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    f1_score = float(match.group(1))
                    current_group[f1_type].append(f1_score)
                    sample_count += 1
                    
                    # When we have 4 samples for all types, store them and start a new group
                    if all(len(scores) == 2+abl_loops for scores in current_group.values()):
                        for f1_type, scores in current_group.items():
                            f1_groups[group_index][f1_type] = scores
                        current_group = defaultdict(list)
                        group_index += 1
    
    # Generate column names
    columns = ['Group Index']
    for f1_type in ['macro', 'micro', 'weighted', 'class-1', 'class0', 'class1']:
        columns.extend([f'{sample}_{f1_type}_f1' for sample in ['init','pretrain']+[f'abl{i+1}' for i in range(abl_loops)]])
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(columns)
        # Write data rows
        for group_idx, scores_dict in f1_groups.items():
            row = [group_idx]
            for f1_type in ['macro', 'micro', 'weighted', 'class-1', 'class0', 'class1']:
                row.extend(scores_dict.get(f1_type, [None]*4))
            writer.writerow(row)
    

# Example usage
log_file_path = 'scripts/misc/logs/log_5-9_crossval.txt'  # Replace with your log file path
output_csv_path = 'scripts/misc/logs/log_5-9_crossval.csv'   # Output CSV file path
extract_f1_scores(log_file_path, output_csv_path, abl_loops=5)
