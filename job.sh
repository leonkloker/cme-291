#!/bin/bash
#SBATCH --partition=k80
#SBATCH --gres=gpu

python3 graph_transformer_spektral.py
