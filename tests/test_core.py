import io,json,socket,time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from flybrain.vision import Vision,OBS_SIZE
from flybrain.control import reward,reflex
from flybrain.brain import Brain
from flybrain.learning import Learner
from flybrain.protocol import FrameParser,Server,decode_png
ROOT=Path(__file__).resolve().parents[1]
def test_reference_screenshot():
    s=Vision().process(np.array(Image.open(ROOT/'examples/reference_screenshot.png')))
    assert s.valid and len(s.threats)==6
    assert s.soul==pytest.approx((527.24,244.93),abs=1)
    assert s.arena==pytest.approx((236,107,602,474),abs=2)
    assert s.mask[173,213] and s.observation.shape==(OBS_SIZE,)
def test_blank_fails_closed():
    assert not Vision().process(np.zeros((100,100,3),np.uint8)).valid
    assert reflex(Vision().process(np.zeros((100,100,3),np.uint8)))==0

def test_damage_reward():
    assert reward(1/30,1)<-.99 and reward(1/30,1,True,True)<-2.99
def test_real_graph():
    b=Brain();assert b.n==1024 and len(b.pre)==62888
    assert int(b.weights.sum())==718607
    assert b.meta['dataset']=='male-cns:v1.0'
    for _ in range(100):f=b.step(np.ones(OBS_SIZE,np.float32))
    assert np.max(np.abs(b.activity))<=1 and f.shape==(256,)
    k=100;assert np.sign(b.matrix[b.post[k],b.pre[k]])==b.signs[b.pre[k]]
    b.reset();assert not b.activity.any()
def test_ppo_checkpoint(tmp_path):
    import torch
    b=Brain('synthetic');l=Learner(b);old=l.model.actor.weight.detach().clone()
    for i in range(32):
        f=b.step(np.ones(OBS_SIZE,np.float32)*(i%2));a,lp,v,_=l.act(f);l.add(f,a,lp,v,-1 if i%4==0 else .01,i==31)
    assert np.isfinite(l.update()['loss']) and not torch.equal(old,l.model.actor.weight)
    p=tmp_path/'checkpoint.pt';l.save(p);assert Learner(b).load(p,b).total_steps==32
    with pytest.raises(ValueError,match='mismatch'):Learner(b).load(p,Brain('synthetic',seed=9))
def packet(seq=1):
    f=io.BytesIO();Image.new('RGB',(10,10),(255,0,0)).save(f,format='PNG');data=f.getvalue()
    return json.dumps(dict(v=1,seq=seq,episode=1,png_bytes=len(data))).encode()+b'\n'+data
def test_protocol_fragments():
    p=FrameParser();frames=[];blob=packet()+packet(2)
    for i in range(0,len(blob),3):frames.extend(p.feed(blob[i:i+3]))
    assert [h['seq'] for h,_ in frames]==[1,2]
    assert decode_png(frames[0][1]).shape==(10,10,3)
    with pytest.raises(ValueError):FrameParser().feed(b'x'*9000)
    with pytest.raises(ValueError):FrameParser().feed(b'{"v":1,"seq":1,"episode":1,"png_bytes":999999999}\n')
def test_loopback():
    s=Server(0);c=socket.create_connection(s.listener.getsockname(),timeout=1)
    try:
        s.poll(.1);c.sendall(packet());frames=[];deadline=time.monotonic()+2
        while not frames and time.monotonic()<deadline:frames=s.poll(.02)
        assert len(frames)==1;s.reply(frames[0][0],6)
        assert json.loads(c.recv(2048))==dict(v=1,seq=1,episode=1,action=6)
        c.sendall(packet())
        with pytest.raises(ValueError,match='order'):s.poll(.5)
    finally:c.close();s.close()
