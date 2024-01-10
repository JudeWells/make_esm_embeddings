import os
import pandas as pd

result_dir = '../metadatas'
new_rows = []
for data_name in os.listdir(result_dir):
    if not data_name.endswith('.txt'):
        continue
    metadata_path = os.path.join(result_dir, data_name)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            row_dict = {}
            for line in f.readlines():
                col, val = line.split(': ')
                if col in ['Disk space used', 'Execution time']:
                    row_dict[col] = val.split(' ')[0]
                else:
                    row_dict[col] = val.split('/')[-1].strip()
            new_rows.append(row_dict)
df = pd.DataFrame(new_rows)
for col in ['Number of sequences', 'Sequence length', 'Batch size']:
    df[col] = df[col].astype(int)
for col in ['Disk space used', 'Execution time']:
    df[col] = df[col].astype(float)
bp=1