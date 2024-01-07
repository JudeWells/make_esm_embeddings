import torch
import esm
import time
import numpy as np
"""
Example layout of protein gym csv file:
mutant,mutated_sequence,DMS_score,DMS_score_bin
A14D,MISNAKIARINELDAKAKAGVITEEEKAEQQKLRQEYLK,-0.0444123855520972,1
A14E,MISNAKIARINELEAKAKAGVITEEEKAEQQKLRQEYLK,0.0698257174212999,1
A14F,MISNAKIARINELFAKAKAGVITEEEKAEQQKLRQEYLK,-0.0332472153874889,1
"""
def make_embeddings_from_df(csv_path, save_path, batch_size=2):
    """"
    csv_path: path to (protein gym) csv file with columns: up_id, sequence, proteome_id
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    embeddings = []
    with open(csv_path, 'r') as file:
        columns = file.readline().strip().split(',')
        batch = []
        for line in file:
            mutant, mutated_sequence, DMS_score, DMS_score_bin = line.strip().split(',')
            batch.append([mutant, mutated_sequence])
            if len(batch) == batch_size:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                representations = results["representations"][33]
                logits = results["logits"][33]
                # add this batch's representations and logits to the list of all representations and logits

        # once file is finished save all of the results to disk


if __name__=="__main__":
    make_embeddings_from_df("/Users/judewells/Downloads/DMS_ProteinGym_substitutions/YAP1_HUMAN_Araya_2012.csv",
                            "", batch_size=2)







