import sys
import torch
import esm
import time
import numpy as np
import pandas as pd
import os

def make_embeddings_from_df(csv_path, save_path, model, batch_converter, batch_size=4, test_mode=False):
    start_time = time.time()
    # Read the CSV file
    df = pd.read_csv(csv_path)
    embeddings = []
    logits = []
    seq_len = len(df.iloc[0].mutated_sequence)
    if seq_len > 1000:
        batch_size = 1
    if test_mode:
        df = df.iloc[:batch_size +1]
    # Process the data in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch = list(batch_df[['mutant', 'mutated_sequence']].itertuples(index=False, name=None))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            representations = results["representations"][33]
            embeddings.extend(representations.cpu().numpy())
            logits.extend(results["logits"].cpu().numpy())

    # Concatenate all embeddings
    all_embeddings = np.array(embeddings)
    all_logits = np.array(logits)
    # Save embeddings to disk
    np.save(os.path.join(save_path, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(save_path, 'logits.npy'), all_logits)

    # Measure execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Measure disk space used
    file_size = os.path.getsize(os.path.join(save_path, 'embeddings.npy'))
    # convert to gigabytes
    file_size = file_size / 1e9
    print(f"Disk space used: {file_size} GigaBytes")
    with open(os.path.join(save_path, 'metadata.txt'), 'w') as file:
        file.write(f"CSV file: {csv_path}\n")
        file.write(f"Number of sequences: {len(df)}\n")
        file.write(f"Sequence length: {seq_len}\n")
        file.write(f"Execution time: {execution_time} seconds\n")
        file.write(f"Disk space used: {file_size} GigaBytes\n")
        file.write(f"Batch size: {batch_size}\n")

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        test_mode = True
    else:
        test_mode = False
    csv_dir = '../DMS_datasets'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files = [(os.path.getsize(os.path.join(csv_dir, f)),f) for f in csv_files]
    csv_files.sort()
    csv_files = [f for _, f in csv_files]
    # Load the ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    for f in csv_files:
        print(f)
        if not f.endswith('.csv'):
            continue
        save_path = f'../DMS_datasets/embeddings/{f.split(".")[0]}'
        os.makedirs(save_path, exist_ok=True)
        input_csv_path = os.path.join(csv_dir, f)
        make_embeddings_from_df(input_csv_path, save_path,
                                model, batch_converter, batch_size=2, test_mode=test_mode)
