#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=82:0:0
# -l gpu=false
#$ -S /bin/bash
#$ -N embed_test
#$ -wd /SAN/orengolab/nsp13/protein_gym/make_esm_embeddings
# -t 1 # subjobs for ${SGE_TASK_ID}
#$ -o /SAN/orengolab/nsp13/protein_gym/qsub_logs/
#$ -j y
hostname
date
source /share/apps/source_files/python/python-3.9.5.source
source /SAN/orengolab/nsp13/esm_env/bin/activate
cd /SAN/orengolab/nsp13/protein_gym/make_esm_embeddings
export TORCH_HOME=/SAN/orengolab/nsp13/esm_env/torch_home
python3 make_esm_embeddings.py
date