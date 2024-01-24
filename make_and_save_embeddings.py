import shutil
import sys
import argparse
import torch
import esm
import time
import numpy as np
import pandas as pd
import os
import tarfile

"""
Note that in the MCC zero shot benchmarks has the following ranking:
TranceptEVE L
TranceptEVE M
TranceptEVE S
GEMME
VESPA
EVE (ensemble)
Tranception L
MSA Transformer (ensemble)
EVE (single)
ESM-IF1
MSA Transformer (single)
Tranception M
DeepSequence (ensemble)
ESM2 (650M)
Tranception S
ESM-1v (ensemble)
ESM2 (3B)
DeepSequence (single)
ESM2 (15B)
VESPAl
ESM-1b
"""

def compress_and_measure(save_path):
    # Compress embeddings directory to tar gzip
    gzip_file_path = save_path + '.tar.gz'
    with tarfile.open(gzip_file_path, "w:gz") as tar:
        tar.add(save_path, arcname=os.path.basename(save_path))

    # Measure the size of the gzip file
    file_size = os.path.getsize(gzip_file_path)
    file_size_gb = file_size / (1024 ** 3)  # Convert to gigabytes

    # Remove the original embeddings directory
    os.system(f"rm -rf {save_path}")

    print(f"Size of compressed file: {file_size_gb} GB")

    return file_size_gb

def make_embeddings_from_df(csv_path, save_path, model, batch_converter, batch_size=4, test_mode=False):
    start_time = time.time()

    # Read the CSV file
    df = pd.read_csv(csv_path)
    seq_len = len(df.iloc[0].mutated_sequence)
    if seq_len > 1000:
        batch_size = 1
    if seq_len > 800:
        batch_size = 2
    if seq_len > 500:
        batch_size = 3
    if test_mode:
        df = df.iloc[:batch_size + 1]

    file_index = 0
    embeddings_batch = []
    logits_batch = []

    os.makedirs(os.path.join(save_path, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'logits'), exist_ok=True)

    # Process the data in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_filepath = os.path.join(save_path, f'embeddings_{str(file_index).zfill(6)}.npy')
        if os.path.exists(batch_filepath):
            print(f"Skipping batch {file_index} - already completed")
            file_index += 1
            continue
        batch = list(batch_df[['mutant', 'mutated_sequence']].itertuples(index=False, name=None))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            for rep, logit, name in zip(results['representations'][33], results['logits'], batch_df.mutant.values):
                np.save(os.path.join(*[save_path, 'embeddings', f'{name}.npy']), rep.cpu().numpy())
                np.save(os.path.join(*[save_path, 'logits', f'{name}.npy']), logit.cpu().numpy())

    # Measure execution time
    end_time = time.time()
    execution_time = end_time - start_time
    # compress embeddings director to tar gzip
    file_size = compress_and_measure(save_path)

    print(f"Execution time: {execution_time} seconds")
    print(f"Disk space used: {file_size} GB")

    # Write metadata as a one-line csv file
    with open(os.path.join(f'../{csv_path.split("/")[-1].split(".")[0].strip()}_metadata.txt'), 'w') as file:
        colnames = ['csv_file', 'num_sequences', 'sequence_length', 'execution_time_sec', 'disk_space_used_gb', 'batch_size']
        file.write(f"{','.join(colnames)}\n")
        file.write(f"{csv_path},{len(df)},{seq_len},{execution_time},{file_size},{batch_size}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode')
    parser.add_argument('--csv_index', type=int, help='Index of the CSV file to process', default=None, nargs='?')
    args = parser.parse_args()

    test_mode = args.test_mode
    csv_index = args.csv_index

    csv_dir = '../DMS_ProteinGym_substitutions'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files = [(os.path.getsize(os.path.join(csv_dir, f)),f) for f in csv_files]
    csv_files.sort()
    csv_files = [f for _, f in csv_files]
    # Load the ESM model
    ### Alternative models ###
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # 33 layer ESM-2 model with 650M params, trained on UniRef50
    # model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S() # 33 layer transformer model with 650M params, trained on Uniref90. This is model 1 of a 5 model ensemble.
    # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() # 33 layer transformer model with 650M params, trained on Uniref50 Sparse.
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    if csv_index is not None:
        csv_files = [csv_files[csv_index]]
    for f in csv_files:
        print(f)
        if not f.endswith('.csv'):
            continue
        save_path = f'../DMS_embeddings/{f.split(".")[0]}'
        if os.path.exists(save_path + '.tar.gz'):
            print(f"Skipping {f} - already completed")
            continue
        os.makedirs(save_path, exist_ok=True)
        input_csv_path = os.path.join(csv_dir, f)
        shutil.copy(input_csv_path, save_path)
        make_embeddings_from_df(input_csv_path, save_path,
                                model, batch_converter, batch_size=4, test_mode=test_mode)
