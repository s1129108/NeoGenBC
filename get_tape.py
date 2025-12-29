import os
import torch
from tape import ProteinBertModel, TAPETokenizer
from tqdm import tqdm
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-in", "--input_folder", type=str)
parser.add_argument("-out", "--out_folder", type=str)

args = parser.parse_args()

input_folder = glob.glob(os.path.join(args.input_folder, "*"))
out_folder = args.out_folder

# 初始化模型和分詞器
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')

for path in tqdm(input_folder, desc="Processing", unit="file"):
    try:
        with open(path) as f:
            fasta = f.readlines()
        title = fasta[0][1:].strip()
        sequence = fasta[1].strip()
        out_path = os.path.join(out_folder, title)
        
        token_ids = torch.tensor([tokenizer.encode(sequence)])
        output = model(token_ids)
        sequence_output = output[0][:, 1:-1, :].cpu().detach().numpy()
        
        np.save(out_path + ".npy", sequence_output)
    except Exception as e:
        error_message = f"An error occurred while processing {path}: {e}"
        print(error_message)

        log_mode = 'a' if os.path.exists("TAPE_log.txt") else 'w'
        with open("TAPE_log.txt", log_mode) as log:
            log.write(title + "\n")
        continue
