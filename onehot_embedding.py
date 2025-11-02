import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar      
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))

df = pd.read_pickle("./data/processed_mutations.dataset")
# print(df['label'].value_counts())
df = df[(df['label'] == 0) | (df['label'] == 1) | (df['label'] == 3) | (df['label'] == 2)]
df = df[df['mut0'].str.len() <= 5000]
df = df[df['mut0'].str.len() >= 512]
df = df[df['par0'].str.len() <= 5000]
df['pos'] = df['Feature range(s)'].str[0].str.split('-', expand=True)[0].astype(int)
df = df[df['Feature range(s)'].str[0].str.split('-', expand=True)[0] == df['Feature range(s)'].str[0].str.split('-', expand=True)[1]]
# df = df[df['mut1'].str.len()-df['pos'] > 256]
# df = df[df['pos'] > 256]
df.reset_index(drop=True, inplace=True)

def one_hot_encode_protein_sequence(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)))

    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            one_hot_matrix[i, aa_to_index[aa]] = 1
        else:
            raise ValueError(f"Unknown aa：{aa}")
    
    return one_hot_matrix

dataset = []
for i in tqdm(range(len(df))):
    pos = int(df.loc[i, 'Feature range(s)'][0].split('-')[0])
    mut0_emb = one_hot_encode_protein_sequence(df['mut0'][i]).astype(np.float32)
    padded_mut0 = F.pad(torch.from_numpy(mut0_emb), (0, 0, 256, 256), "constant", 0)
    mut0_emb = padded_mut0[(pos - 1):(pos + 512)]
    mut1_emb = one_hot_encode_protein_sequence(df['mut1'][i]).astype(np.float32)
    padded_mut1 = F.pad(torch.from_numpy(mut1_emb), (0, 0, 256, 256), "constant", 0)
    mut1_emb = padded_mut1[(pos - 1):(pos + 512)]
    par0_emb = one_hot_encode_protein_sequence(df['par0'][i]).astype(np.float32)
    par0_emb = np.vstack((par0_emb, np.zeros((1025,20)))).astype(np.float32)
    par0_emb = par0_emb[:1024]
    entry = (torch.tensor(mut0_emb), torch.tensor(mut1_emb), torch.tensor(par0_emb), torch.tensor(df['label'][i]))
    dataset.append(entry)
    
    # print(par0_emb.dtype,par0_emb.shape,par0_emb)
torch.save(dataset, 'data/embeddings/onehot_embedding.pt')