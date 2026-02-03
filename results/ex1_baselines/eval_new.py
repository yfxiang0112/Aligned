import pandas as pd
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import load_npz
import glob
import os
import argparse
from datetime import datetime

from aligned.reasoner import RegulatoryKB


def weighted_mean(f1_data, f1_kb, w):
    p_integrate = 2
    return (w * (f1_data ** -p_integrate)
            + (1.-w) * (f1_kb ** -p_integrate)) ** (-1/p_integrate)


def setup_output_file(system, dataset):
    """Setup output file for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data_anal", "experiment_results", dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Find the next available log number
    log_num = 1
    while True:
        log_filename = f"log_{system}_{log_num}"
        log_path = os.path.join(output_dir, log_filename)
        if not os.path.exists(log_path):
            break
        log_num += 1

    print(f"Starting evaluation for system: {system}, dataset: {dataset}")
    print(f"Output file: {log_path}")
    return log_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate EGOAL predictions on different systems and datasets')

    parser.add_argument('--system', type=str, choices=['scfoundation', 'scgpt', 'gears', 'additive'], required=True,
                        help='Prediction system to evaluate (scfoundation, scgpt, gears, or additive)')

    parser.add_argument('--dataset', type=str, choices=['adamson', 'dixit', 'norman'], required=True,
                        help='Dataset name (adamson, dixit, or norman)')

    parser.add_argument('--predictions_dir', type=str, default='data_anal/pert_benchmark',
                        help='Directory containing prediction files')

    parser.add_argument('--predictions_pattern', type=str, default='all_predictions*.json',
                        help='Pattern to match prediction files')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computations (cuda or cpu)')

    parser.add_argument('--closure_t', type=int, default=5,
                        help='Number of closure iterations for knowledge base')

    parser.add_argument('--weight', type=float, default=0.3233,
                        help='Weight for balanced F1 score calculation')

    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for discretizing continuous predictions')

    return parser.parse_args()


def main():
    args = parse_arguments()
    output_path = setup_output_file(args.system, args.dataset)

    print(f"Arguments: {vars(args)}")

    # Store all output for writing to file
    output_lines = []
    output_lines.append(
        f"Starting evaluation for system: {args.system}, dataset: {args.dataset}")
    output_lines.append(f"Arguments: {vars(args)}")

    # Configure paths based on dataset (all are human datasets now)
    test_metadata = pd.read_csv(
        f'dataset/human/{args.dataset}_test_set.csv', index_col=0)
    test_idx_path = f'dataset/human/{args.dataset}_test_idx.npy'
    Y_path = f'dataset/human/{args.dataset}_Y.npz'
    Y_con_path = f'dataset/human/{args.dataset}_Y_con.npz'
    X_path = f'dataset/human/{args.dataset}_X.npz'
    kb_pos_path = f'rules/human/{args.dataset}_KB_P.npz'
    kb_neg_path = f'rules/human/{args.dataset}_KB_N.npz'
    predictions_dir = os.path.join(
        args.predictions_dir, f'{args.system}_{args.dataset}')

    print(f"Using predictions directory: {predictions_dir}")
    print(f"Using KB paths: {kb_pos_path}, {kb_neg_path}")
    output_lines.append(f"Using predictions directory: {predictions_dir}")
    output_lines.append(f"Using KB paths: {kb_pos_path}, {kb_neg_path}")

    file_pattern = args.predictions_pattern
    log_files = glob.glob(os.path.join(predictions_dir, file_pattern))
    log_files.sort()  # Sort for consistent ordering

    if not log_files:
        error_msg = f"No prediction files found matching pattern {file_pattern} in {predictions_dir}"
        print(error_msg)
        output_lines.append(error_msg)
        return

    print(f"Found {len(log_files)} prediction files")
    output_lines.append(f"Found {len(log_files)} prediction files")

    data_f1_lst, kb_f1_lst, bal_f1_lst = [], [], []

    for file_idx, log_file in enumerate(log_files, 1):
        print(f"\nProcessing file {file_idx}: {log_file}")
        output_lines.append(f"Processing file {file_idx}: {log_file}")
        pred_result = json.load(open(log_file, 'r'))

        # Load test data
        if os.path.exists(test_idx_path):
            test_idx = np.load(test_idx_path)
            Y_true = load_npz(Y_path).toarray()[test_idx]
            X = load_npz(X_path).toarray()[test_idx]
        else:
            # If no test_idx file, use all data
            Y_true = load_npz(Y_path).toarray()
            X = load_npz(X_path).toarray()
            warning_msg = f"Test index file {test_idx_path} not found, using all data"
            print(warning_msg)
            output_lines.append(warning_msg)

        if test_metadata is not None:
            # Human system with metadata
            indices = list(test_metadata.apply(lambda x: (
                x['data_start_idx'], x['data_end_idx+1']), axis=1))
            pert_keys = test_metadata['pert'].apply(
                lambda x: f'{eval(x)[0]}_{eval(x)[1]}' if 'ctrl' not in x else list(set(eval(x))-{'ctrl'})[0])
            predictions = list(pert_keys.apply(lambda x: pred_result[x]))

            # Find the maximum row index needed
            max_row = max(end for _, end in indices) if indices else 0

            # Find the maximum column length (assuming all predictions[i] have same length)
            max_col = max(len(a) for a in predictions) if predictions else 0

            # Initialize the matrix with zeros
            Y = np.zeros((max_row, max_col))

            # Fill the matrix according to the specifications
            for i, (start, end) in enumerate(indices):
                if i < len(predictions):
                    # Convert predictions[i] to numpy array and ensure proper shape
                    a_array = np.array(predictions[i])
                    # Repeat the array for the specified row range
                    for row in range(start, end):
                        Y[row, :len(a_array)] = a_array
        else:
            # Fallback for cases where metadata is not available
            if isinstance(pred_result, dict):
                # If it's a dictionary, try to extract the main prediction array
                Y = np.array(list(pred_result.values())
                             [0] if pred_result else [])
            else:
                Y = np.array(pred_result)

            # Ensure Y has the right shape to match Y_true
            if Y.shape != Y_true.shape:
                warning_msg = f"Prediction shape {Y.shape} doesn't match true shape {Y_true.shape}"
                print(warning_msg)
                output_lines.append(warning_msg)
                # Try to reshape or truncate as needed
                if Y.size == Y_true.size:
                    Y = Y.reshape(Y_true.shape)
                else:
                    min_size = min(Y.size, Y_true.size)
                    Y = Y.flatten()[:min_size].reshape(-1,
                                                       Y_true.shape[1] if len(Y_true.shape) > 1 else 1)
                    Y_true = Y_true.flatten()[:min_size].reshape(Y.shape)

        # Eval MSE if continuous labels are available
        if os.path.exists(Y_con_path):
            if os.path.exists(test_idx_path):
                Y_true_con = load_npz(Y_con_path).toarray()[test_idx]
            else:
                Y_true_con = load_npz(Y_con_path).toarray()

            criterion = torch.nn.MSELoss(reduction='mean')
            mse_score = criterion(torch.tensor(Y).float(),
                                  torch.tensor(Y_true_con).float())
            print(f'MSE: {mse_score}')
            output_lines.append(f'MSE: {mse_score}')

        # Discretize predictions
        Y = np.where(np.abs(Y) > args.threshold, np.sign(Y), 0)

        print(f'Prediction shape: {Y.shape}')
        output_lines.append(f'Prediction shape: {Y.shape}')

        # Eval data consistency
        f1_data = f1_score(Y_true.flatten(), Y.flatten(), average="macro")
        print(f'Data F1: {f1_data}')
        output_lines.append(f'Data F1: {f1_data}')

        confusion = confusion_matrix(
            Y_true.flatten(), Y.flatten()).astype(np.float64)
        confusion /= np.sum(confusion)
        print(f'Confusion matrix:\n{confusion}')
        output_lines.append(f'Confusion matrix:\n{confusion}')

        # Eval on KB deduction
        device = args.device
        try:
            KB = RegulatoryKB(pos_trn_pth=kb_pos_path,
                              neg_trn_pth=kb_neg_path, device=device)
            KB.closure_(T=args.closure_t, closure_type='weighted')
            Y_deduction = KB.deduce(torch.tensor(
                X).float().to(device)).to('cpu').numpy()

            f1_kb = f1_score(Y_deduction.flatten(),
                             Y.flatten(), average="macro")
            f1_bal = weighted_mean(f1_data, f1_kb, args.weight)
            print(f'KB F1: {f1_kb}\nBalanced F1: {f1_bal}')
            output_lines.append(f'KB F1: {f1_kb}\nBalanced F1: {f1_bal}')
        except Exception as e:
            error_msg = f"Error in KB evaluation: {e}"
            print(error_msg)
            output_lines.append(error_msg)
            f1_kb = 0.0
            f1_bal = f1_data

        data_f1_lst.append(f1_data)
        kb_f1_lst.append(f1_kb)
        bal_f1_lst.append(f1_bal)

    # Final summary statistics
    print("\n------\nintegrated result:")
    output_lines.append("\n------\nintegrated result:")

    data_f1_mean = 0.5*(max(data_f1_lst)+min(data_f1_lst))
    data_f1_std = 0.5*(max(data_f1_lst)-min(data_f1_lst))
    result_line = f'f1 on test: {data_f1_mean:.4f} ± {data_f1_std:.4f}'
    print(result_line)
    output_lines.append(result_line)

    kb_f1_mean = 0.5*(max(kb_f1_lst)+min(kb_f1_lst))
    kb_f1_std = 0.5*(max(kb_f1_lst)-min(kb_f1_lst))
    result_line = f'f1 on KB: {kb_f1_mean:.4f} ± {kb_f1_std:.4f}'
    print(result_line)
    output_lines.append(result_line)

    bal_f1_mean = 0.5*(max(bal_f1_lst)+min(bal_f1_lst))
    bal_f1_std = 0.5*(max(bal_f1_lst)-min(bal_f1_lst))
    result_line = f'integrated F1: {bal_f1_mean:.4f} ± {bal_f1_std:.4f}'
    print(result_line)
    output_lines.append(result_line)

    print(f"Evaluation completed. Results saved to: {output_path}")

    # Write all results to the output file
    with open(output_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')


if __name__ == "__main__":
    main()
