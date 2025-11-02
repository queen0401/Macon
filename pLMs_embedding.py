import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer, AutoTokenizer, EsmModel
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))

def get_model_and_tokenizer(model_name):
    if model_name == "protT5":
        model = T5EncoderModel.from_pretrained("model/prot_t5_xl_half_uniref50-enc")
        tokenizer = T5Tokenizer.from_pretrained('model/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    elif model_name == "protbert":
        model = BertModel.from_pretrained("model/prot_bert")
        tokenizer = BertTokenizer.from_pretrained("model/prot_bert", do_lower_case=False)
    elif model_name == "esm":
        model = EsmModel.from_pretrained("model/esm2_t33_650M_UR50D")
        tokenizer = AutoTokenizer.from_pretrained("model/esm2_t33_650M_UR50D", do_lower_case=False )
    else:
        raise ValueError("Unsupported model name. Choose from 'protT5', 'protbert', or 'esm'.")

    model = model.to(device)
    model = model.eval()
    return model, tokenizer

config = {
    "model_name": "protT5",  # Change this to 'protbert' or 'esm' as needed
}

model, tokenizer = get_model_and_tokenizer(config["model_name"])

df = pd.read_pickle("./data/processed_mutations.dataset")
# print(df['label'].value_counts())
df = df[(df['label'] == 0) | (df['label'] == 1) | (df['label'] == 3) | (df['label'] == 2)]
print(len(df))
df = df[df['mut0'].str.len() <= 5000]
df = df[df['mut0'].str.len() >= 512]
df = df[df['par0'].str.len() <= 5000]
df['pos'] = df['Feature range(s)'].str[0].str.split('-', expand=True)[0].astype(int)
df = df[df['Feature range(s)'].str[0].str.split('-', expand=True)[0] == df['Feature range(s)'].str[0].str.split('-', expand=True)[1]]
# df = df[df['mut1'].str.len()-df['pos'] > 256]
# df = df[df['pos'] > 256]
print(len(df))
df.reset_index(drop=True, inplace=True)

def plms_embedding(sequence):
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

dataset = []
filename = f'data/embeddings/{config["model_name"]}_embedding_base.pt'

for i in tqdm(range(0, len(df))):
    if i > len(df):
        break
    pos = int(df.loc[i, 'Feature range(s)'][0].split('-')[0])
    mut0_emb = plms_embedding(df['mut0'][i])[0].astype(np.float32)
    padded_mut0 = F.pad(torch.from_numpy(mut0_emb), (0, 0, 256, 256), "constant", 0)
    mut0_emb = padded_mut0[(pos - 1):(pos + 512)]
    # print(mut0_emb.shape)
    mut1_emb = plms_embedding(df['mut1'][i])[0].astype(np.float32)
    padded_mut1 = F.pad(torch.from_numpy(mut1_emb), (0, 0, 256, 256), "constant", 0)
    mut1_emb = padded_mut1[(pos - 1):(pos + 512)]
    par0_emb = plms_embedding(df['par0'][i])[0].astype(np.float32)
    par0_emb = np.vstack((par0_emb, np.zeros((1025,1024)))).astype(np.float32)
    par0_emb = par0_emb[:1024]
    entry = (torch.tensor(mut0_emb), torch.tensor(mut1_emb), torch.tensor(par0_emb), torch.tensor(df['label'][i]))
    dataset.append(entry)

torch.save(dataset, filename) 
    
