import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from tqdm import tqdm


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


    
class Trainer:
    def __init__(self,train_dataset,model,tconf,num_epoch=5,is_train=True):
        self.tconf=tconf
        self.vocab_size = train_dataset.vocab_size
        self.data = train_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=model
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(tconf)
    def train(self,is_train=True):
        tokens=0
        train_loader = DataLoader(self.data, batch_size=self.tconf.batch_size, shuffle=True, num_workers=0)
        for epoch in range(self.tconf.max_epochs):
            total_loss=[]
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for idx,(x,y) in pbar:
                x=x.to(self.device)
                y=y.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits,_ = self.model(x)
                    loss = F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
                    loss = loss.mean()
                    
                    total_loss.append(loss.item())
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    if self.tconf.lr_decay:
                        tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if tokens < self.tconf.warmup_tokens:
                            # linear warmup
                            lr_mult = float(tokens) / float(max(1, self.tconf.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(tokens - self.tconf.warmup_tokens) / float(max(1, self.tconf.final_tokens - self.tconf.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = self.tconf.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = self.tconf.learning_rate
                
                
                
                pbar.set_description(f"epoch {epoch+1} iter {idx}: train loss {loss.item():.5f}. lr {lr:e}")
        print("Training complete!")
        #torch.save(model, 'gpt2_model.pth')

        
        