"""PPO on fixed reservoir features. No anatomical synapses are trained."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
class Readout(nn.Module):
    def __init__(self,size=256):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(size),nn.Linear(size,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.actor=nn.Linear(64,9);self.critic=nn.Linear(64,1)
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
    def forward(self,x):
        h=self.net(x)
        return self.actor(h),self.critic(h).squeeze(-1)
class Learner:
    def __init__(self,brain,seed=7,lr=3e-4):
        torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
        self.model=Readout(brain.feature_size);self.brain_config=brain.config()
        self.optim=torch.optim.Adam(self.model.parameters(),lr=lr,eps=1e-5)
        self.buffer=[];self.total_steps=0;self.updates=0;self.last_loss=0.;self.imitation_steps=0
    def act(self,features,deterministic=False):
        with torch.no_grad():
            logits,value=self.model(torch.as_tensor(features,dtype=torch.float32))
            dist=Categorical(logits=logits)
            action=torch.argmax(logits) if deterministic else dist.sample()
        return int(action),float(dist.log_prob(action)),float(value),dist.probs.numpy()
    def add(self,features,action,logprob,value,reward,done):
        self.buffer.append((features.copy(),action,logprob,value,float(reward),bool(done)))
        self.total_steps+=1
    def update(self,next_value=0.,epochs=4):
        if len(self.buffer)<2:return {}
        x,a,old_log,v,r,done=map(np.asarray,zip(*self.buffer))
        adv=np.zeros(len(r),np.float32);gae=0.
        for i in reversed(range(len(r))):
            following=next_value if i==len(r)-1 else v[i+1]
            alive=1.-float(done[i]);delta=r[i]+.99*following*alive-v[i]
            gae=delta+.99*.95*alive*gae;adv[i]=gae
        returns=adv+v;adv=(adv-adv.mean())/(adv.std()+1e-8)
        tx=torch.tensor(x,dtype=torch.float32);ta=torch.tensor(a,dtype=torch.long)
        tl=torch.tensor(old_log,dtype=torch.float32);tv=torch.tensor(returns,dtype=torch.float32);tad=torch.tensor(adv,dtype=torch.float32)
        losses=[]
        for _ in range(epochs):
            for idx in torch.randperm(len(x)).split(64):
                logits,values=self.model(tx[idx]);dist=Categorical(logits=logits)
                ratio=(dist.log_prob(ta[idx])-tl[idx]).exp()
                obj=torch.minimum(ratio*tad[idx],ratio.clamp(.8,1.2)*tad[idx])
                loss=-obj.mean()+.5*(values-tv[idx]).square().mean()-.01*dist.entropy().mean()
                self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),.5);self.optim.step()
                losses.append(float(loss.detach()))
        self.buffer.clear();self.updates+=1;self.last_loss=float(np.mean(losses))
        return dict(loss=self.last_loss,updates=self.updates)
    def imitate(self,features,actions,epochs=8):
        if not len(features):return
        x=torch.tensor(np.asarray(features),dtype=torch.float32);a=torch.tensor(actions,dtype=torch.long)
        for _ in range(epochs):
            for idx in torch.randperm(len(x)).split(128):
                logits,_=self.model(x[idx]);loss=nn.functional.cross_entropy(logits,a[idx])
                self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),1.);self.optim.step()
        self.imitation_steps+=len(features)
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(version=1,state=self.model.state_dict(),optimizer=self.optim.state_dict(),brain=self.brain_config,steps=self.total_steps,updates=self.updates,imitation_steps=self.imitation_steps)
        temp=path.with_name(path.name+".tmp");torch.save(payload,temp);temp.replace(path)
    def load(self,path,brain):
        d=torch.load(path,map_location="cpu",weights_only=True)
        if d.get("version")!=1:raise ValueError("Unsupported checkpoint version")
        expected=brain.config()
        for key in ("fingerprint","seed","feature_size","model_version"):
            if d["brain"].get(key)!=expected.get(key):raise ValueError("Checkpoint graph/adapter mismatch: "+key)
        self.model.load_state_dict(d["state"]);self.optim.load_state_dict(d["optimizer"])
        self.total_steps=d.get("steps",0);self.updates=d.get("updates",0);self.imitation_steps=d.get("imitation_steps",0)
        return self
