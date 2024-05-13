#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
import torch.nn.functional as F
import h5py
import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar      
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))

def get_T5_model():
    model = T5EncoderModel.from_pretrained("prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer
model, tokenizer = get_T5_model()

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

def prottrans_embedding(sequence):
    seq = ' '.join(list(sequence))
    seq = [seq]
    # print(seq)
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
    # print(ids)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()
    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)

    return features

# embedding_df = pd.DataFrame()
# embedding_df['mut0']= df['mut0'].apply(prottrans_embedding)
# embedding_df.to_csv('data/embeddings/prottrans_embedding.csv')
dataset = []
filename = 'data/embeddings/prottrans_embedding_base.pt'
# for j in range(0,3):
# for i in tqdm(range(j * 1000, min((j+1) * 1000, len(df)))):
for i in tqdm(range(0, len(df))):
    if i > len(df):
        break
    pos = int(df.loc[i, 'Feature range(s)'][0].split('-')[0])
    mut0_emb = prottrans_embedding(df['mut0'][i])[0].astype(np.float32)
    padded_mut0 = F.pad(torch.from_numpy(mut0_emb), (0, 0, 256, 256), "constant", 0)
    mut0_emb = padded_mut0[(pos - 1):(pos + 512)]
    # print(mut0_emb.shape)
    mut1_emb = prottrans_embedding(df['mut1'][i])[0].astype(np.float32)
    padded_mut1 = F.pad(torch.from_numpy(mut1_emb), (0, 0, 256, 256), "constant", 0)
    mut1_emb = padded_mut1[(pos - 1):(pos + 512)]
    par0_emb = prottrans_embedding(df['par0'][i])[0].astype(np.float32)
    par0_emb = np.vstack((par0_emb, np.zeros((1025,1024)))).astype(np.float32)
    par0_emb = par0_emb[:1024]
    entry = (torch.tensor(mut0_emb), torch.tensor(mut1_emb), torch.tensor(par0_emb), torch.tensor(df['label'][i]))
    dataset.append(entry)

if os.path.exists(filename):
    existing_data = torch.load(filename)  # 加载已有数据
    dataset = existing_data + dataset            # 追加新数据
    torch.save(dataset, filename) 
    dataset = []
else:    
    torch.save(dataset, filename) 
    
