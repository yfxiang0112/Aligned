import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd

def extract_metrics_from_file(file_path):
    """
    Extract metrics from a single log file
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            metrics = {}
            
            # Extract p value
            p_match = re.findall(r'KB recovery: incompleteness p = ([\d\.]+)', content)
            if p_match:
                metrics['p'] = [float(x) for x in p_match]
            
            # Extract F1 scores
            f1_initial_match = re.findall(r'F1 on initial KB.*combined:\s+([\d\.]+)', content)
            if f1_initial_match:
                metrics['f1_initial_KB_combined'] = [float(x) for x in f1_initial_match]
            
            f1_closure_match = re.findall(r'F1 on closure KB.*combined:\s+([\d\.]+)', content)
            if f1_closure_match:
                metrics['f1_closure_KB_combined'] = [float(x) for x in f1_closure_match]
            
            # Extract graph metrics
            assortativity_match = re.findall(r'degree_assortativity:\s+([\d\.\-]+)', content)
            if assortativity_match:
                metrics['degree_assortativity'] = [float(x) for x in assortativity_match]
            
            modularity_match = re.findall(r'modularity:\s+([\d\.\-]+)', content)
            if modularity_match:
                metrics['modularity'] = [float(x) for x in modularity_match]
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
    return metrics

def process_directory(directory_path):
    """
    Process all files in a directory and extract metrics
    """
    all_metrics = defaultdict(list)
    p_values = []
    
    # Get all files in the directory
    for filename in os.listdir(directory_path):
        #if filename.endswith('.txt') or filename.endswith('.log'):  # Add other extensions if needed
        file_path = os.path.join(directory_path, filename)
        metrics = extract_metrics_from_file(file_path)
        
        if metrics and 'p' in metrics:
            p_value = metrics['p']
            p_values.append(p_value)
            
            # Store metrics by p value
            for key, value in metrics.items():
                if key != 'p':  # Don't store p value in the metrics list
                    all_metrics[key].append(value)
    
    return all_metrics, p_values

def calculate_statistics(metrics_dict):
    """
    Calculate mean and other statistics for each metric
    """
    statistics = {}
    
    for metric_name, values in metrics_dict.items():
        if values:  # Only calculate if there are values

            statistics[metric_name + '_mean'] =\
                 list(.5*(np.min(np.array(values), axis=0) + np.max(np.array(values), axis=0)))

            statistics[metric_name + '_tol'] =\
                 list(.5*(np.max(np.array(values), axis=0) - np.min(np.array(values), axis=0)))
    
    return statistics

def main():
    # Specify your directory path here
    directory_path = "data_anal/refine"
    
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    # Process all files in the directory
    all_metrics, p_values = process_directory(directory_path)
    print(all_metrics)
    
    if not all_metrics:
        print("No valid metrics found in the files.")
        return
    
    # Calculate statistics
    stats = calculate_statistics(all_metrics)
    print(stats)
    stats['f1_closure_KB_combined_mean'].insert(0,1.)
    stats['f1_closure_KB_combined_tol'].insert(0,0.)
    stats['f1_initial_KB_combined_mean'].insert(0,1.)
    stats['f1_initial_KB_combined_tol'].insert(0,0.)
    df = pd.DataFrame(stats)
    df.to_csv('data_anal/refine/log.csv')
    
    ## Print results
    #print("\n" + "="*60)
    #print("METRICS SUMMARY")
    #print("="*60)
    #
    #for metric_name, stat in stats.items():
    #    print(f"\n{metric_name}:")
    #    print(f"  Mean: {stat['mean']:.6f}")
    #    print(f"  Std: {stat['std']:.6f}")
    #    print(f"  Min: {stat['min']:.6f}")
    #    print(f"  Max: {stat['max']:.6f}")
    #    print(f"  Count: {stat['count']}")
    #
    #print("\n" + "="*60)
    #print("MEAN VALUES (what you requested):")
    #print("="*60)
    #print(f"f1_initial_KB_combined: {stats['f1_initial_KB_combined']['mean']:.6f}")
    #print(f"f1_closure_KB_combined: {stats['f1_closure_KB_combined']['mean']:.6f}")
    #print(f"degree_assortativity: {stats['degree_assortativity']['mean']:.6f}")
    #print(f"modularity: {stats['modularity']['mean']:.6f}")
    #
    ## Also save results to a file
    #output_file = "metrics_summary.txt"
    #with open(output_file, 'w') as f:
    #    f.write("METRICS SUMMARY\n")
    #    f.write("="*60 + "\n")
    #    for metric_name, stat in stats.items():
    #        f.write(f"\n{metric_name}:\n")
    #        f.write(f"  Mean: {stat['mean']:.6f}\n")
    #        f.write(f"  Std: {stat['std']:.6f}\n")
    #        f.write(f"  Min: {stat['min']:.6f}\n")
    #        f.write(f"  Max: {stat['max']:.6f}\n")
    #        f.write(f"  Count: {stat['count']}\n")
    #
    #print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
