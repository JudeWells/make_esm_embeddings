import os
import pandas as pd

"""
Filter criteria for DMS datasets:
sequence length less than 500
number of mutants greater than 500
number of mutants less than 5000
randomly sample 10
"""

def calculate_avg_mem_per_res():
    mem_df = pd.read_csv('runtime_and_memory.csv')
    mem_df['avg_mem_per_res'] = mem_df['Disk space used'] / (mem_df['Number of sequences'] * mem_df['Sequence length'])
    return mem_df.avg_mem_per_res.mean()


if __name__=="__main__":
    summary_df = pd.read_csv('../mutatants_gt500_lt_5000_seq_lt500.csv')
    mem_per_res_gb = calculate_avg_mem_per_res()
    summary_df['estimated_disk_space'] = summary_df.DMS_total_number_mutants * summary_df.seq_len * mem_per_res_gb
    prot_gym_dir = '../ProteinGym/Benchmarks'
    new_rows = []
    id2metrics = {}
    for predictor_type in ['DMS_supervised', 'DMS_zero_shot']:
        for metric in ['Spearman', 'MCC', 'MSE']:
            result_dir = os.path.join(prot_gym_dir, predictor_type, 'substitutions', metric)
            if os.path.exists(result_dir):
                for fname in os.listdir(result_dir):
                    if fname.endswith('.csv') and 'summary' not in fname.lower():
                        df = pd.read_csv(os.path.join(result_dir, fname))
                        for i, row in df.iterrows():
                            scores = row[[c for c in df.columns if c != 'DMS_id']]
                            if metric == 'MSE':
                                best_score = scores.min()
                            else:
                                best_score = scores.max()
                            new_row = {
                                'DMS_id': row.DMS_id,
                                f'best_{metric}_{predictor_type}': best_score,
                                f'mean_{metric}_{predictor_type}': scores.mean(),

                        }
                            if 'fold' in fname.lower():
                                fold_type = fname.split('_fold_')[1].split('.')[0]
                                new_row['fold_type'] = fold_type
                            new_rows.append(new_row)



