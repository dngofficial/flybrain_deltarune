"""Connect live Rudinn screenshots to the fly-inspired controller."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .brain import Brain
from .learning import Learner
from .vision import Vision
from .control import NAMES,reflex,reward

def setup(a):
    b=Brain(a.graph,seed=a.brain_seed);l=Learner(b,seed=a.seed)
    if a.checkpoint:l.load(a.checkpoint,b)
    return b,l

def serve(a):
    from .protocol import Server,decode_png
    from .dashboard import Dashboard
    b,l=setup(a);v=Vision();server=Server(a.port);ui=None if a.no_ui else Dashboard(b)
    previous=None;episode=None;connected=False;hits=0;last=None;last_probs=None;last_action=0
    stats=dict(mode='WAITING FOR KRISTAL');path=Path(a.log);path.parent.mkdir(parents=True,exist_ok=True)
    print(f"Listening on 127.0.0.1:{a.port}. Press F8 in Kristal.",flush=True)
    def finish_pending():
        nonlocal previous
        if a.learn and l.buffer:
            # Bootstrap the last confirmed transition, then discard the action
            # whose result never arrived. Do not invent a zero-reward sample.
            l.update(l.act(previous[0])[2] if previous else 0.)
            l.save(a.out)
        previous=None
    try:
        with path.open('a',buffering=1) as log:
            while not ui or not ui.closed:
                if ui:ui.events()
                try:frames=server.poll(.01)
                except (ValueError,ConnectionError,OSError) as exc:
                    print('Connection reset:',exc,flush=True);server.disconnect();frames=[]
                if connected and not server.client:
                    finish_pending();episode=None;v.reset();b.reset();last=None;stats=dict(mode='DISCONNECTED / HUMAN CONTROL')
                connected=bool(server.client)
                for h,png in frames:
                    start=time.perf_counter();ep=h['episode']
                    if ep!=episode:
                        finish_pending();episode=ep;v.reset();b.reset();hits=0
                    done=bool(h.get('terminal',False));dt=float(h.get('dt',0));hits+=int(h.get('hits',0))
                    r=reward(dt,h.get('hits',0),done,bool(h.get('died',False)))
                    if png:last=v.process(decode_png(png),max(dt,1/30))
                    valid=bool(png and last and last.valid and h.get('active',False) and not done)
                    if previous and a.learn:
                        f,action,lp,value=previous
                        if h.get('action_applied')==action:l.add(f,action,lp,value,r,done or not valid)
                        else:
                            # Manual/foreign action: end this rollout before continuing.
                            if l.buffer:l.update(l.act(f)[2]);l.save(a.out)
                    previous=None
                    if not valid:b.reset()
                    f=b.step(last.observation) if valid else None
                    if a.learn and len(l.buffer)>=a.rollout:
                        l.update(l.act(f)[2] if valid else 0.);l.save(a.out)
                    probs=None
                    if valid:
                        if a.mode=='policy':
                            action,lp,value,probs=l.act(f,deterministic=not a.learn)
                            if a.learn:previous=(f,action,lp,value)
                        else:action=reflex(last)
                    else:action=0
                    server.reply(h,action)
                    if done:v.reset();b.reset();last=None
                    stats=dict(mode='KRISTAL / '+a.mode.upper()+' / '+('LEARNING' if a.learn else 'EVALUATION'),reward=r,hits=hits,episode=ep,steps=l.total_steps if a.learn else h['seq'],loss=l.last_loss,updates=l.updates,latency_ms=(time.perf_counter()-start)*1000)
                    log.write(json.dumps(dict(**stats,seq=h['seq'],action=action,terminal=done,vision_valid=valid))+'\n')
                    last_action,last_probs=action,probs
                    if h['seq']%100==0:print(f"Frame {h['seq']} | hits {hits} | {NAMES[action]} | {stats['latency_ms']:.1f}ms",flush=True)
                if ui:ui.draw(last,stats,last_probs,last_action)
    except KeyboardInterrupt:pass
    finally:
        if a.learn:finish_pending();l.update(0.);l.save(a.out)
        server.close()
        if ui:ui.close()

def main():
    p=argparse.ArgumentParser(description="Live EnemyRecreations Rudinn controller")
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("serve")
    q.add_argument("--graph",help="Connectome NPZ path or synthetic control graph")
    q.add_argument("--brain-seed",type=int,default=7)
    q.add_argument("--seed",type=int,default=7)
    q.add_argument("--checkpoint")
    q.add_argument("--mode",choices=["reflex","policy"],default="reflex")
    q.add_argument("--rollout",type=int,default=256)
    q.add_argument("--port",type=int,default=65432)
    q.add_argument("--learn",action="store_true")
    q.add_argument("--no-ui",action="store_true")
    q.add_argument("--out",default="checkpoints/rudinn.pt")
    q.add_argument("--log",default="runs/rudinn.jsonl")
    a=p.parse_args()
    if a.rollout<2:p.error("--rollout must be at least 2")
    if a.learn and a.mode!="policy":p.error("--learn requires --mode policy")
    serve(a)
