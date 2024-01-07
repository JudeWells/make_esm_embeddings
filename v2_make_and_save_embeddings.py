import torch
import esm
import time
import numpy as np
import pandas as pd
import os

def make_embeddings_from_df(csv_path, save_path, batch_size=4):
    start_time = time.time()

    # Load the ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Read the CSV file
    df = pd.read_csv(csv_path)
    embeddings = []
    logits = []
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

if __name__ == "__main__":
    csv_path = "/Users/judewells/Downloads/DMS_ProteinGym_substitutions/YAP1_HUMAN_Araya_2012.csv"
    make_embeddings_from_df(
        csv_path,
        "./",
        batch_size=2
    )
