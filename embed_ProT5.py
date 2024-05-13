#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))

seq_path = "./data/processed_mutations.dataset"

# whether to retrieve embeddings for each residue in a protein
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True
per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = False
per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = False
sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

#@title Network architecture for secondary structure prediction. { display-mode: "form" }
# Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet( torch.nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )

        self.dssp8_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )


    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x         = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat   = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat   = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(  x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat
    

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer

#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=1000, max_batch=100):
    if sec_struct:
        sec_struct_model = load_sec_struct_model()

    results = {
        "residue_embs": dict(),
        "protein_embs": dict(),
        "sec_structs": dict()
    }

    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()

    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict, 1), total=len(seq_dict), desc="Embedding Progress"):
        seq = ' '.join(list(seq))
        seq_len = len(seq)
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct:
                d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if sec_struct:
                    results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[1].detach().cpu().numpy().squeeze()
                if per_residue:
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')
    return results


#@title Write embeddings to disk. { display-mode: "form" }
# def save_embeddings(emb_dict,out_path):
#     with h5py.File(str(out_path), "w") as hf:
#         for sequence_id, embedding in emb_dict.items():
#             # noinspection PyUnboundLocalVariable
#             sequence_id = str(sequence_id)
#             # embedding = np.array(embedding)
#             hf.create_dataset(sequence_id, data=embedding)
#     return None

#@title Write predictions to disk. { display-mode: "form" }
def write_prediction_fasta(predictions, out_path):
  class_mapping = {0:"H",1:"E",2:"L"}
  with open(out_path, 'w+') as out_f:
      out_f.write( '\n'.join(
          [ ">{}\n{}".format(
              seq_id, ''.join( [class_mapping[j] for j in yhat] ))
          for seq_id, yhat in predictions.items()
          ]
            ) )
  return None

def pad_embeddings(results):
    for sequence_id, embedding in results["residue_embs"].items():
        # results["residue_embs"][sequence_id] = np.vstack((np.zeros((1025,1024)), results["residue_embs"][sequence_id], np.zeros((1025,1024))))
        pos = pos = int(df.loc[sequence_id, 'Feature range(s)'][0].split('-')[0])
        # print(sequence_id, ':',results["residue_embs"][sequence_id][pos])
        results["residue_embs"][sequence_id] = results["residue_embs"][sequence_id][(pos - 257):(pos + 256)]
    return results

def pad_par_embeddings(results):
    for sequence_id, embedding in results["residue_embs"].items():
        # results["residue_embs"][sequence_id] = np.vstack((np.zeros((1025,1024)), results["residue_embs"][sequence_id], np.zeros((1025,1024))))
        # pos = pos = int(df.loc[sequence_id, 'Feature range(s)'][0].split('-')[0])
        # # print(sequence_id, ':',results["residue_embs"][sequence_id][pos])
        # results["residue_embs"][sequence_id] = results["residue_embs"][sequence_id][(pos - 257):(pos + 256)]
        results["residue_embs"][sequence_id] = np.vstack((results["residue_embs"][sequence_id], np.zeros((1025,1024))))
        results["residue_embs"][sequence_id] = results["residue_embs"][sequence_id][:1024]
    return results

# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
df = pd.read_pickle("./data/processed_mutations.dataset")
# print(df['label'].value_counts())
df = df[(df['label'] == 0) | (df['label'] == 1) | (df['label'] == 3)]
df = df[df['mut0'].str.len() <= 5000]
df['pos'] = df['Feature range(s)'].str[0].str.split('-', expand=True)[0].astype(int)
df = df[df['mut1'].str.len()-df['pos'] > 256]
df = df[df['pos'] > 256]
# df = df.head(22)
df.reset_index(drop=True, inplace=True)
mut0_dict = df['mut0'].to_dict()
mut1_dict = df['mut1'].to_dict()
par0_dict = df['par0'].to_dict()

# Compute embeddings and/or secondary structure predictions
# results_mut0 = get_embeddings( model, tokenizer, mut0_dict, per_residue, per_protein, sec_struct)
# results_mut1 = get_embeddings( model, tokenizer, mut1_dict, per_residue, per_protein, sec_struct)
# results_par0 = get_embeddings( model, tokenizer, par0_dict, per_residue, per_protein, sec_struct)

# results_mut0 = pad_embeddings(results_mut0)
# results_mut1 = pad_embeddings(results_mut1)
# results_par0 = pad_embeddings(results_par0)

# Store per-residue embeddings
# save_embeddings(results_mut0["residue_embs"], "./protT5/output/mut0_embeddings.h5")
# save_embeddings(results_mut1["residue_embs"], "./protT5/output/mut1_embeddings.h5")
# save_embeddings(results_par0["residue_embs"], "./protT5/output/par0_embeddings.h5")

def save_embeddings(emb_dict, out_path):
    with h5py.File(str(out_path), "a") as hf:  # Use "a" mode to append to existing file
        for sequence_id, embedding in emb_dict.items():
            sequence_id = str(sequence_id)
            if sequence_id in hf:
                print(sequence_id)
                del hf[sequence_id]  # Remove existing dataset with the same name if it exists
            # print(sequence_id, embedding.shape)
            hf.create_dataset(sequence_id, data=embedding)
    return None


def get_embeddings_and_save(model, tokenizer, seqs, per_residue, per_protein, sec_struct, par = False, 
                            max_residues=4000, max_seq_len=1000, max_batch=100, save_interval=5, save_path="embeddings.h5"):
    # if sec_struct:
    #     sec_struct_model = load_sec_struct_model()

    results = {
        "residue_embs": dict(),
        "protein_embs": dict(),
        "sec_structs": dict()
    }

    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()

    num_processed = 0

    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict, 1), total=len(seq_dict), desc="Embedding Progress"):
        seq = ' '.join(list(seq))
        seq_len = len(seq)
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()
            # print(seqs)
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct:
                d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if sec_struct:
                    results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[1].detach().cpu().numpy().squeeze()
                if per_residue:
                    # print(emb.detach().cpu().numpy().squeeze().shape)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

            num_processed += len(pdb_ids)

            if num_processed >= save_interval:
                if par:
                    results = pad_par_embeddings(results)
                else:
                    results = pad_embeddings(results)
                save_embeddings(results["residue_embs"], save_path)
                results = {
        "residue_embs": dict(),
        "protein_embs": dict(),
        "sec_structs": dict()
    }
                num_processed = 0

    # Save the remaining embeddings
    if par:
        results = pad_par_embeddings(results)
    else:
        results = pad_embeddings(results)
    save_embeddings(results["residue_embs"], save_path)

    # passed_time = time.time() - start
    # avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    # print('\n############# EMBEDDING STATS #############')
    # print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    # print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    # print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
    #     passed_time / 60, avg_time))
    # print('\n############# END #############')
    return results

results_mut0 = get_embeddings_and_save( model, tokenizer, mut0_dict, per_residue, per_protein, sec_struct, save_path="./protT5/output/mut0_embeddings_013.h5")
results_mut1 = get_embeddings_and_save( model, tokenizer, mut1_dict, per_residue, per_protein, sec_struct, save_path="./protT5/output/mut1_embeddings_013.h5")
results_par0 = get_embeddings_and_save( model, tokenizer, par0_dict, per_residue, per_protein, sec_struct, par=True, save_path="./protT5/output/par0_embeddings_013.h5")

