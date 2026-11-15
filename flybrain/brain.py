"""Fixed leaky rate reservoir on anatomical edges; PPO trains the artificial readout."""
import hashlib,json
import numpy as np
from scipy.sparse import csr_matrix
from .vision import OBS_SIZE
from .connectome import DEFAULT_GRAPH
class Brain:
    pools=128
    feature_size=256
    def __init__(self,path=None,seed=7):
        self.seed=seed
        rng=np.random.default_rng(seed)
        if path=="synthetic":
            n=256
            self.ids=np.arange(n);self.pre=rng.integers(0,n,n*8);self.post=rng.integers(0,n,n*8)
            self.weights=rng.uniform(1,10,len(self.pre)).astype(np.float32)
            self.signs=rng.choice([-1.,1.],n,p=[.2,.8]).astype(np.float32)
            self.types=np.full(n,"synthetic");self.classes=np.full(n,"synthetic")
            self.positions=np.full((n,3),np.nan)
            self.meta=dict(kind="synthetic_control",dataset="synthetic — NOT fly connectome",neurons=n,edges=len(self.pre),synapse_count=None)
        else:
            with np.load(path or DEFAULT_GRAPH,allow_pickle=False) as d:
                for attr,key in [("ids","ids"),("pre","pre"),("post","post"),("weights","weight"),("signs","sign"),("types","types"),("classes","classes"),("positions","positions")]:setattr(self,attr,d[key].copy())
                self.meta=json.loads(str(d["metadata"]))
        self.n=len(self.ids)
        h=hashlib.sha256()
        for arr in (self.ids,self.pre,self.post,self.weights,self.signs,self.classes):h.update(arr.tobytes())
        self.fingerprint=h.hexdigest()
        vals=np.log1p(self.weights)*self.signs[self.pre]
        matrix=csr_matrix((vals,(self.post,self.pre)),shape=(self.n,self.n),dtype=np.float32)
        sums=np.asarray(abs(matrix).sum(axis=1)).ravel()
        self.matrix=matrix.multiply((.85/np.maximum(sums,1e-8))[:,None]).tocsr()
        self.inputs=rng.integers(0,OBS_SIZE,(self.n,6))
        self.input_weights=rng.normal(0,.65,(self.n,6)).astype(np.float32)
        self.visual=np.isin(self.classes,["visual_projection","ol_intrinsic","ol_sensory","synthetic"])
        if not self.visual.any():raise ValueError("No visual input cells in graph")
        self.pool_index=np.arange(self.n)%self.pools
        self.count=np.maximum(1,np.bincount(self.pool_index,minlength=self.pools))
        self.reset()
    def reset(self):
        self.activity=np.zeros(self.n,np.float32)
        self.steps=0
    def step(self,observation):
        drive=(observation[self.inputs]*self.input_weights).sum(axis=1)*self.visual
        for _ in range(2):self.activity=.55*self.activity+.45*np.tanh(drive+self.matrix@self.activity)
        self.steps+=1
        signed=np.bincount(self.pool_index,weights=self.activity,minlength=self.pools)/self.count
        absolute=np.bincount(self.pool_index,weights=np.abs(self.activity),minlength=self.pools)/self.count
        return np.concatenate((signed,absolute)).astype(np.float32)
    def config(self):
        return dict(seed=self.seed,fingerprint=self.fingerprint,feature_size=self.feature_size,model_version=1,metadata=self.meta)
