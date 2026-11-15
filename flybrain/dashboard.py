"""Local inspection window. Real soma positions and measured activity, no decorative brain."""
import numpy as np
import pygame
from .control import NAMES
from PIL import Image
BG=(9,13,20);PANEL=(16,22,32);MUTED=(137,153,173);TEXT=(225,233,242)
LIME=(135,245,143);CYAN=(83,205,245);PINK=(255,94,145);GOLD=(255,202,103)
class Dashboard:
    def __init__(self,brain):
        pygame.init();self.screen=pygame.display.set_mode((1240,810))
        pygame.display.set_caption("Fly / Dodge Lab")
        self.font=pygame.font.SysFont("consolas",17);self.small=pygame.font.SysFont("consolas",13)
        self.big=pygame.font.SysFont("consolas",28,bold=True)
        self.brain=brain;self.closed=False;self.paused=False;self.last_draw=-100
        self.clock=pygame.time.Clock()
        pos=brain.positions.astype(float).copy();self.valid=np.isfinite(pos).all(axis=1)
        if self.valid.any():
            lo=np.nanpercentile(pos[self.valid],1,axis=0);hi=np.nanpercentile(pos[self.valid],99,axis=0)
            self.xy=np.clip(((pos-lo)/np.maximum(1,hi-lo))[:,[0,1]],0,1)
        else:
            a=np.linspace(0,2*np.pi,brain.n,endpoint=False)
            self.xy=np.column_stack((.5+.43*np.cos(a),.5+.43*np.sin(a)));self.valid[:]=True
        self.edges=np.random.default_rng(7).choice(len(brain.pre),min(650,len(brain.pre)),replace=False)
    def events(self):
        out=[]
        for e in pygame.event.get():
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key in (pygame.K_ESCAPE,pygame.K_q)):self.closed=True
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_r:out.append("reset")
                elif e.key==pygame.K_SPACE:self.paused=not self.paused
        return out
    def text(self,s,x,y,color=TEXT,font=None):self.screen.blit((font or self.font).render(str(s),True,color),(x,y))
    def panel(self,rect,label):
        pygame.draw.rect(self.screen,PANEL,rect,border_radius=10);self.text(label,rect[0]+16,rect[1]+12,CYAN,self.small)
    def draw(self,seen=None,stats=None,probs=None,action=0):
        stats=stats or {};now=pygame.time.get_ticks()
        if now-self.last_draw<33:return
        self.last_draw=now;s=self.screen;s.fill(BG)
        self.text("FLY / DODGE LAB",24,16,TEXT,self.big)
        self.text(stats.get("mode","WAITING FOR KRISTAL"),410,27,LIME)
        self.text("LOCAL EXPERIMENT / v0.1",968,30,MUTED,self.small)
        self.panel((20,68,514,430),"01  LIVE KRISTAL FRAME / MODEL INPUT")
        if seen is not None:
            im=Image.fromarray(seen.image).convert("RGB");im.thumbnail((480,358))
            s.blit(pygame.image.frombuffer(im.tobytes(),im.size,"RGB"),(277-im.width//2,110+(358-im.height)//2))
            self.text(seen.status,36,472,LIME if seen.valid else PINK,self.small)
        else:self.text("Waiting for a Rudinn defense frame.",42,250,MUTED)
        self.panel((550,68,670,430),"02  CONNECTOME ACTIVITY / SOMA x-y PROJECTION")
        self.text(f"{self.brain.meta['dataset']} | {self.brain.n:,} neurons",568,102,TEXT,self.small)
        self.text(f"{len(self.brain.pre):,} directed edges | fixed rate dynamics",568,122,MUTED,self.small)
        points=self.xy*np.array([605,285])+np.array([580,162])
        for k in self.edges:
            i,j=self.brain.pre[k],self.brain.post[k]
            if self.valid[i] and self.valid[j]:pygame.draw.line(s,(31,46,61),points[i],points[j])
        mag=np.minimum(1,np.abs(self.brain.activity)*4)
        for i in range(self.brain.n):
            if not self.valid[i]:continue
            base=CYAN if self.brain.visual[i] else (GOLD if "descending" in self.brain.classes[i] else PINK)
            color=tuple(int(c*(.35+.65*mag[i])) for c in base)
            pygame.draw.circle(s,color,points[i],2 if mag[i]<.35 else 3)
        self.text("Real somas | sampled edges | brightness = activity magnitude",568,472,MUTED,self.small)
        self.panel((20,514,514,248),"03  RETINA + REINFORCEMENT")
        if seen is not None:
            g=(seen.retina*255).astype(np.uint8)
            rgb=np.stack((g*.35,g,g*.8),axis=-1).astype(np.uint8)
            surface=pygame.surfarray.make_surface(rgb.transpose(1,0,2))
            s.blit(pygame.transform.scale(surface,(144,144)),(36,552))
        self.text("16x16 retina",36,706,MUTED,self.small)
        for label,key,y,color,fmt in [("Reward","reward",552,GOLD,"+.3f"),("Hits","hits",580,PINK,""),("Episode","episode",608,TEXT,""),("Steps","steps",636,TEXT,","),("PPO loss","loss",664,MUTED,".4f")]:
            self.text(f"{label:9s} {stats.get(key,0):{fmt}}",205,y,color)
        self.text("Damage: -1 / hit; death: -2",205,706,MUTED,self.small)
        self.panel((550,514,670,248),"04  MOTOR READOUT")
        self.text("ACTION",568,554,MUTED,self.small);self.text(NAMES[action],568,576,LIME,self.big)
        self.text(f"Vision + control: {stats.get('latency_ms',0):.1f} ms",568,626,TEXT,self.small)
        self.text(f"Learning updates: {stats.get('updates',0)}",568,648,MUTED,self.small)
        self.text("Trainable: artificial readout",568,680,MUTED,self.small)
        self.text("Anatomical edges: fixed",568,702,MUTED,self.small)
        probs=np.asarray(probs) if probs is not None else np.eye(9)[action]
        for i,p in enumerate(probs):
            x=904+(i%3)*98;y=552+(i//3)*61
            pygame.draw.rect(s,(26,36,48),(x,y,88,48),border_radius=5)
            pygame.draw.rect(s,LIME if i==action else (48,104,120),(x,y+43,max(1,int(88*p)),5),border_radius=2)
            self.text(NAMES[i],x+5,y+5,TEXT,self.small);self.text(f"{p:.0%}",x+5,y+23,MUTED,self.small)
        self.text("Q / ESC close   |   KRISTAL: F8 toggle control",24,781,MUTED,self.small)
        pygame.display.flip()
    def close(self):pygame.quit()
