"""Nonlearning geometric baseline and shared reward definition."""
import numpy as np
ACTIONS=np.array([(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)],np.float32)
NAMES=["STAY","LEFT","RIGHT","UP","DOWN","UP LEFT","UP RIGHT","DOWN LEFT","DOWN RIGHT"]
def reward(dt,hits,terminal=False,died=False):
    return .02*max(0.,dt)-max(0,int(hits))-(2. if died else 0.)
def reflex(seen):
    if not seen.valid:
        return 0
    sx,sy=seen.soul
    x0,y0,x1,y1=seen.arena
    speed=seen.radius*15
    scores=[]
    for dx,dy in ACTIONS:
        score=0.
        for t in (.08,.18,.35):
            px=np.clip(sx+dx*speed*t,x0+seen.radius,x1-seen.radius)
            py=np.clip(sy+dy*speed*t,y0+seen.radius,y1-seen.radius)
            for b in seen.threats:
                clearance=np.hypot(px-b.x-b.vx*t,py-b.y-b.vy*t)-(seen.radius+b.radius+2)
                score+=np.exp(-max(-10,clearance)/(seen.radius*2))/(1+t)
            wall=min(px-x0,x1-px,py-y0,y1-py)
            score+=.08*np.exp(-wall/(seen.radius*2))
        px=sx+dx*speed*.18
        py=sy+dy*speed*.18
        score+=.035*(((px-(x0+x1)/2)/(x1-x0))**2+((py-(y0+y1)/2)/(y1-y0))**2)
        score+=.0002*(abs(dx)+abs(dy))
        scores.append(score)
    return int(np.argmin(scores))
