"""
Created by Jude Wells
"""

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

def decompress_directory(gzip_file_path):
    if os.path.exists(gzip_file_path):
        with tarfile.open(gzip_file_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(gzip_file_path))
    else:
        print(f"File {gzip_file_path} not found.")

def compress_and_measure(save_path):
    # Compress embeddings directory to tar gzip
    gzip_file_path = save_path + '.tar.gz'
    with tarfile.open(gzip_file_path, "w:gz") as tar:
        tar.add(save_path, arcname=os.path.basename(save_path))
    # Remove the original embeddings directory
    os.system(f"rm -rf {save_path}")


def reconstruct_wt_sequence(mutant, mutated_sequence):
    # Assumes mutant is in the format XnnY, where X is original, nn is position, Y is mutated
    position = int(''.join(filter(str.isdigit, mutant))) - 1  # Convert position to zero-based index
    original_aa = mutant[0]  # Original amino acid
    wt_sequence = mutated_sequence[:position] + original_aa + mutated_sequence[position+1:]
    return wt_sequence

def create_and_save_wt_embedding(model, batch_converter, wt_sequence, save_path):
    batch_labels, batch_strs, batch_tokens = batch_converter([(None, wt_sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        wt_embedding = results['representations'][33].cpu().numpy()[0]  # Extract embedding for the WT sequence
        logits = results['logits'].cpu().numpy()[0]  # Extract logits for the WT sequence
        np.save(os.path.join(save_path, 'embeddings', 'wt.npy'), wt_embedding)
        np.save(os.path.join(save_path, 'logits', 'wt.npy'), logits)

if __name__ == "__main__":
    csv_dir = '../DMS_ProteinGym_substitutions'
    root_dir_local = '/Users/judewells/Documents/dataScienceProgramming/protein_gym/DMS_embeddings'
    root_dir = '/SAN/orengolab/nsp13/protein_gym/DMS_embeddings'
    if os.path.exists(root_dir_local):
        root_dir = root_dir_local
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    # Load the ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    for i, csv_file in enumerate(csv_files):
        tar_gz_path = os.path.join(root_dir, csv_file.replace('.csv', '.tar.gz'))
        save_path = tar_gz_path.replace('.tar.gz', '')

        # Decompress the directory
        decompress_directory(tar_gz_path)
        if not os.path.exists(save_path):
            print(f"Directory {save_path} not found.")
            continue
        # Process each CSV file in the directory to reconstruct WT sequence and create embeddings
        for file in os.listdir(save_path):
            if file.endswith('.csv'):
                csv_path = os.path.join(save_path, file)
                df = pd.read_csv(csv_path)
                row = df.iloc[0]
                mutations = row.mutant.split(":")
                sequence = row['mutated_sequence']
                for mutation in mutations:
                    sequence = reconstruct_wt_sequence(mutation, sequence)
                wt_sequence = sequence
                create_and_save_wt_embedding(model, batch_converter, wt_sequence, save_path)
                # Recompress the directory
                compress_and_measure(save_path)
