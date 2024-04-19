import torch

import numpy as np
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import random
class Gpt(nn.Module):
    def __init__(self):
        super(Gpt,self).__init__()
        self.decoder=Decoder()
        self.projection = nn.Linear(d_model,vocab_size)
        
    def forward(self,inputs):
        outputs,attns = self.decoder(inputs)
        dec_logits = self.projection(outputs)
        return dec_logits.view(-1,dec_logits.size(-1)), attns
        
 

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
           
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()
    
    def forward(self,q,k,v,attn_mask):
        scores = torch.matmul(q,k.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill_(attn_mask,float('-inf'))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,v)
        return context,attn
        
            
class FeedForward(nn.Module):
    def __init__(self,drop_out=0.1):
        super(FeedForward,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ffs,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ffs,out_channels=d_model,kernel_size=1)
        self.actv1 = NewGELU()
        self.residual_drop = nn.Dropout(p=drop_out)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,input):
        residual = input
        outputs = self.actv1(self.conv1(input.transpose(1,2)))
        outputs = self.conv2(outputs).transpose(1,2)
        return self.norm(self.residual_drop(outputs))
        
          
class MultiheadAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super(MultiheadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k*n_heads)
        self.W_K = nn.Linear(d_model,d_k*n_heads)
        self.W_V = nn.Linear(d_model,d_v*n_heads)
        
        self.attn_drop = nn.Dropout(p=dropout)
        self.residual_drop = nn.Dropout(p=dropout)
        
        self.norm = nn.LayerNorm(d_model)
    def forward(self,Q,K,V,attn_mask):
        
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)        
        k_s = self.W_K(K).view(batch_size,-1,n_heads,d_k).transpose(1,2) 
        v_s = self.W_V(V).view(batch_size,-1,n_heads,d_v).transpose(1,2) 
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
        context, attn  = ScaledDotProductAttention()(q_s,k_s,v_s,attn_mask)
        context = self.attn_drop(context)
        context=context.transpose(1,2).contiguous().view(batch_size,-1,d_model)
        
        return self.norm(self.residual_drop(residual+context)), attn
        
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn = MultiheadAttention()
        self.ffn = FeedForward()
    def forward(self,dec_inputs,dec_self_attn_mask):
        
        outputs,attn = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        outputs = self.ffn(outputs)
        return outputs,attn 
        
        
        
class Decoder(nn.Module):
    def __init__(self,dropout=0.1):
        super(Decoder,self).__init__()
        self.src_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, src_len, d_model))
        
        self.emb_drop = nn.Dropout(p=dropout)
        self.layer = nn.ModuleList(DecoderLayer() for _ in range(n_layer))
    def forward(self, inputs):
        t = x.size(1)
        
        src_embedding = self.src_emb(inputs)
        pos_embedding = self.pos_emb[:, :t, :]
        
        dec_outputs = self.emb_drop(src_embedding+pos_embedding)
        
        pad_mask = get_attn_pad_mask(inputs,inputs).to(device)
        
        subsequent_mask = get_attn_subsequent_mask(inputs).to(device)
        
        self_attn_mask = torch.gt((pad_mask+subsequent_mask),1)
        
        dec_self_attns = []
        for layer in self.layer:
            dec_outputs, attn = layer(dec_outputs,self_attn_mask)
            
            dec_self_attns.append(attn)
        return dec_outputs,attn
        
        
def get_attn_pad_mask(Q,K):
    batch_size, lenQ = Q.size()
    batch_size, lenK = K.size()
    
    attn_pad_mask = K.data.eq(0).unsqueeze(1)
    return attn_pad_mask.expand(batch_size,lenQ,lenK)
    
def get_attn_subsequent_mask(Q):
    batch_size, lenQ = Q.size()
    attn_mask_shape = [batch_size,lenQ,lenQ]
    subsequent_mask = np.triu(np.ones(attn_mask_shape),k=1)
    subsequent_mask= torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
    
        




class CharDataset(Dataset):
    
    

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i+1 for i,ch in enumerate(chars) }
        self.itos = { i+1:ch for i,ch in enumerate(chars) }
        
        # customize vocab for futher usage
        self.stoi["[PAD]"] = 0
        self. itos[0]="[PAD]"
        
        self.block_size = block_size
        self.vocab_size = vocab_size+1
        self.data = data
        
        with open('char_to_idx.pkl', 'wb') as f:
            pickle.dump(self.stoi, f)

        with open('idx_to_char.pkl', 'wb') as f:
            pickle.dump(self.itos, f)
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        
        From https://github.com/karpathy/minGPT/blob/feature/lightning/play_char.ipynb
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
    
d_model = 512

n_layer = 8
n_heads = 8
d_k=d_v=64
src_len=128
batch_size=512
d_ffs = 4*d_model
learning_rate=3e-4

text = open('JinYong/金庸三联版/倚天屠龙记.txt', 'r',encoding='utf-8').read() # don't worry we won't run out of file handles
print(text[:100])
train_dataset = CharDataset(text, src_len)    
vocab_size = train_dataset.vocab_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = Gpt()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


no_decay = ['bias','LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n,p in model.named_parameters() if not any (nd in n for nd in no_decay)],
        'weight_decay':0.01},
    {'params': [p for n,p in model.named_parameters() if any (nd in n for nd in no_decay)],
        'weight_decay':0.0},
]


for data in train_loader:
    x,y=data
    print("Batch shape:", x)
    print("Batch shape:", y)
    break
    