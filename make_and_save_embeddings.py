import sys
import torch
import esm
import time
import numpy as np
import pandas as pd
import os
import tarfile

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
    if test_mode:
        df = df.iloc[:batch_size + 1]

    file_index = 0
    embeddings_batch = []
    logits_batch = []

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
            representations = results["representations"][33]
            embeddings_batch.extend(representations.cpu().numpy())
            logits_batch.extend(results["logits"].cpu().numpy())

            # Save every 100 batches
            if len(embeddings_batch) >= 100 or i + batch_size >= len(df):
                np.save(batch_filepath, np.array(embeddings_batch))
                embeddings_batch = []
                np.save(os.path.join(save_path, f'logits_{str(file_index).zfill(6)}.npy'), np.array(logits_batch))
                logits_batch = []
                file_index += 1

    # Measure execution time
    end_time = time.time()
    execution_time = end_time - start_time
    # compress embeddings director to tar gzip
    file_size = compress_and_measure(save_path)

    print(f"Execution time: {execution_time} seconds")
    print(f"Disk space used: {file_size} GB")

    # Write metadata
    with open(os.path.join(f'../{csv_path.split("/")[-1].split(".")[0].strip()}_metadata.txt'), 'w') as file:
        file.write(f"CSV file: {csv_path}\n")
        file.write(f"Number of sequences: {len(df)}\n")
        file.write(f"Sequence length: {seq_len}\n")
        file.write(f"Execution time: {execution_time} seconds\n")
        file.write(f"Disk space used: {file_size} GB\n")
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
        save_path = f'../DMS_datasets/embeddings_full/{f.split(".")[0]}'
        if os.path.exists(f"{save_path}/embeddings.npy"):
            print(f"Skipping {f} - already completed")
            continue
        os.makedirs(save_path, exist_ok=True)
        input_csv_path = os.path.join(csv_dir, f)
        make_embeddings_from_df(input_csv_path, save_path,
                                model, batch_converter, batch_size=2, test_mode=test_mode)
