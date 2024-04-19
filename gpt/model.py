import torch
import numpy as np
import torch.nn as nn
import math

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, src_len, **kwargs):
        self.vocab_size = vocab_size
        self.src_len = src_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class Gpt(nn.Module):
    def __init__(self,config):
        super(Gpt,self).__init__()
        self.config=config
        self.decoder=Decoder(config)
        self.norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(self.config.d_model,self.config.vocab_size,bias=False)
    def forward(self,inputs):
        b,t = inputs.size()
        assert t<=self.config.src_len ,"Cannot forward, model max src len is exhausted"
        outputs,attns = self.decoder(inputs)
        outputs = self.norm(outputs)
        dec_logits = self.projection(outputs)
        return dec_logits, attns
    def get_block_size(self):
        return self.config.src_len    
    
    
    def configure_optimizers(self,config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        from https://github.com/karpathy/minGPT.git
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        return optimizer

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
           
class ScaledDotProductAttention(nn.Module):
    def __init__(self,drop_out=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.attn_drop = nn.Dropout(drop_out)
    def forward(self,q,k,v,attn_mask):
        scores = torch.matmul(q,k.transpose(-1,-2))/np.sqrt(k.size(-1))
        scores.masked_fill_(attn_mask,float('-inf'))
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.attn_drop(attn)
        context = torch.matmul(attn,v)
        return context,attn
        
            
class FeedForward(nn.Module):
    def __init__(self,config,drop_out=0.1):
        super(FeedForward,self).__init__()
        self.conv1 = nn.Linear(config.d_model,4*config.d_model)
        self.conv2 = nn.Linear(4*config.d_model,config.d_model)
        self.actv1 = nn.GELU()
        self.residual_drop = nn.Dropout(p=drop_out)
    def forward(self,input):
        
        outputs = self.actv1(self.conv1(input))
        outputs = self.conv2(outputs)
        return self.residual_drop(outputs)
        
          
class MultiheadAttention(nn.Module):
    def __init__(self,config,dropout=0.1):
        super(MultiheadAttention,self).__init__()
        self.config = config
        self.W_Q = nn.Linear(self.config.d_model,self.config.d_k*self.config.n_heads)
        self.W_K = nn.Linear(self.config.d_model,self.config.d_k*self.config.n_heads)
        self.W_V = nn.Linear(self.config.d_model,self.config.d_v*self.config.n_heads)
        self.linear = nn.Linear(self.config.d_model,self.config.d_model)
        self.residual_drop = nn.Dropout(p=dropout)
        
    def forward(self,Q,K,V,attn_mask):
        
        _, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size,-1,self.config.n_heads,self.config.d_k).transpose(1,2)        
        k_s = self.W_K(K).view(batch_size,-1,self.config.n_heads,self.config.d_k).transpose(1,2) 
        v_s = self.W_V(V).view(batch_size,-1,self.config.n_heads,self.config.d_v).transpose(1,2) 
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.config.n_heads,1,1)
        
        
        context, attn  = ScaledDotProductAttention()(q_s,k_s,v_s,attn_mask)
        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.config.d_model)
        output = self.linear(context)
        return self.residual_drop(output), attn
        
class DecoderLayer(nn.Module):
    def __init__(self,config):
        super(DecoderLayer,self).__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dec_self_attn = MultiheadAttention(config)
        self.ffn = FeedForward(config)
        
    def forward(self,dec_inputs,dec_self_attn_mask):
        residual = dec_inputs
        dec_inputs = self.norm1(dec_inputs)
        outputs,attn = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs = residual + outputs
        dec_outputs = dec_outputs + self.ffn(self.norm2(dec_outputs))
        return dec_outputs,attn 
        
        
        
class Decoder(nn.Module):
    def __init__(self,config,dropout=0.1):
        super(Decoder,self).__init__()
        self.src_emb = nn.Embedding(config.vocab_size,config.d_model)
        self.pos_emb = nn.Embedding(config.src_len,config.d_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_drop = nn.Dropout(p=dropout)
        self.layer = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layer))
    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)

        pos = pos.unsqueeze(0).expand_as(inputs).to(self.device)
        src_embedding = self.src_emb(inputs)
        pos_embedding = self.pos_emb(pos)
        
        dec_outputs = self.emb_drop(src_embedding+pos_embedding)
        subsequent_mask = get_attn_subsequent_mask(inputs).to(self.device)
        
        dec_self_attn_mask = torch.gt((subsequent_mask), 0)
        
        dec_self_attns = []
        for layer in self.layer:
            dec_outputs, attn = layer(dec_outputs,dec_self_attn_mask)
            
            dec_self_attns.append(attn)
        return dec_outputs,attn
        
    
def get_attn_subsequent_mask(Q):
    batch_size, lenQ = Q.size()
    attn_mask_shape = [batch_size,lenQ,lenQ]
    subsequent_mask = np.triu(np.ones(attn_mask_shape),k=1)
    subsequent_mask= torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask