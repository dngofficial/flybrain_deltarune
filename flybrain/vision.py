"""Pixel-only observations. Health and game object coordinates never enter this module."""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

GRID = 16
OBS_SIZE = 2 * GRID * GRID + 6

@dataclass
class Threat:
    x: float
    y: float
    radius: float
    vx: float = 0.
    vy: float = 0.

@dataclass
class Seen:
    image: np.ndarray
    arena: tuple | None = None
    soul: tuple | None = None
    radius: float = 6.
    threats: list = field(default_factory=list)
    mask: np.ndarray | None = None
    retina: np.ndarray = field(default_factory=lambda: np.zeros((GRID, GRID), np.float32))
    observation: np.ndarray = field(default_factory=lambda: np.zeros(OBS_SIZE, np.float32))
    status: str = "No green arena"
    tracked_soul: bool = False
    @property
    def valid(self):
        return self.arena is not None and self.soul is not None

def components(mask, minimum=4):
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    if not n:
        return [], labels
    areas = np.bincount(labels.ravel())
    return [(i,s,int(areas[i])) for i,s in enumerate(ndi.find_objects(labels),1)
            if s is not None and areas[i]>=minimum], labels

class Vision:
    def __init__(self, soul_memory=.2, arena=None):
        self.soul_memory = soul_memory
        self.manual_arena = arena
        self.reset()
    def reset(self):
        self.last = None
        self.previous_grid = np.zeros((GRID, GRID), np.float32)
        self.missing_soul = 0.
    def process(self, rgb, dt=1/30):
        rgb = np.asarray(rgb,dtype=np.uint8)[...,:3].copy()
        out = Seen(rgb)
        r,g,b = (rgb[...,k].astype(np.int16) for k in range(3))
        green = (g>85)&(g>r*1.5+25)&(g>b*1.35+20)
        items,labels = components(green,30)
        boxes = []
        for ident,(ys,xs),area in items:
            w,h = xs.stop-xs.start,ys.stop-ys.start
            region = labels[ys,xs]==ident
            if min(w,h)<35 or not .2<w/h<5 or area/(w*h)>.4:
                continue
            if region.sum(0).max()<.55*h or region.sum(1).max()<.55*w:
                continue
            thick = max(2,round(area/(2*(w+h))))
            boxes.append((w*h,(xs.start+thick,ys.start+thick,xs.stop-thick,ys.stop-thick)))
        out.arena = self.manual_arena or (max(boxes)[1] if boxes else None)
        if out.arena is None:
            self.reset()
            return out
        x0,y0,x1,y1 = map(int,out.arena)
        height,width = rgb.shape[:2]
        if not (0<=x0<x1<=width and 0<=y0<y1<=height):
            out.arena=None
            out.status="Arena override outside image"
            return out
        inner=np.zeros((height,width),bool)
        inner[y0:y1,x0:x1]=True
        red=(r>105)&(r>g*1.7+25)&(r>b*1.45+20)&inner
        items,labels=components(red,5)
        if items:
            ident,(ys,xs),area=max(items,key=lambda x:x[2])
            yy,xx=np.nonzero(labels==ident)
            out.soul=(float(xx.mean()),float(yy.mean()))
            out.radius=max(3.,.43*max(xs.stop-xs.start,ys.stop-ys.start))
            self.missing_soul=0.
        elif self.last and self.last.soul and self.last.arena==out.arena:
            self.missing_soul+=dt
            if self.missing_soul<=self.soul_memory:
                out.soul=self.last.soul
                out.radius=self.last.radius
                out.tracked_soul=True
        if out.soul is None:
            out.status="No red soul — neutral action"
            self.last=out
            self.previous_grid.fill(0)
            return out
        margin=round(.45*max(x1-x0,y1-y0))
        crop=np.zeros_like(inner)
        crop[max(0,y0-margin):min(height,y1+margin),max(0,x0-margin):min(width,x1+margin)]=True
        minimum=np.minimum(np.minimum(r,g),b)
        maximum=np.maximum(np.maximum(r,g),b)
        white=(minimum>145)&(maximum-minimum<75)&crop
        # Close small gaps in antialiased/dashed outlines, then fill black interiors.
        white=ndi.binary_closing(white,iterations=max(1,round(out.radius/6)))
        white &= ~ndi.binary_dilation(green,iterations=1)
        items,labels=components(white,5)
        out.mask=np.zeros_like(white)
        for ident,(ys,xs),area in items:
            w,h=xs.stop-xs.start,ys.stop-ys.start
            if max(w,h)>.7*max(x1-x0,y1-y0):
                continue
            filled=ndi.binary_fill_holes(labels[ys,xs]==ident)
            yy,xx=np.nonzero(filled)
            tx,ty=float(xs.start+xx.mean()),float(ys.start+yy.mean())
            if np.hypot(tx-out.soul[0],ty-out.soul[1])<out.radius*.65:
                continue
            out.mask[ys,xs] |= filled
            out.threats.append(Threat(tx,ty,max(2.,max(w,h)/2)))
        if self.last and self.last.valid and self.last.arena==out.arena and self.last.threats and out.threats:
            old=np.array([(t.x,t.y) for t in self.last.threats])
            new=np.array([(t.x,t.y) for t in out.threats])
            dist=np.linalg.norm(new[:,None]-old[None,:],axis=2)
            rows,cols=linear_sum_assignment(dist)
            gate=max(8.,min(x1-x0,y1-y0)*4*dt)
            for i,j in zip(rows,cols):
                if dist[i,j]<=gate:
                    out.threats[i].vx=(new[i,0]-old[j,0])/max(dt,1e-4)
                    out.threats[i].vy=(new[i,1]-old[j,1])/max(dt,1e-4)
        sx,sy=out.soul
        scale=max(x1-x0,y1-y0)*.8
        yy,xx=np.nonzero(out.mask)
        gx=np.floor(((xx-sx)/scale+1)*GRID/2).astype(int)
        gy=np.floor(((yy-sy)/scale+1)*GRID/2).astype(int)
        good=(gx>=0)&(gx<GRID)&(gy>=0)&(gy<GRID)
        out.retina[gy[good],gx[good]]=1.
        extra=np.array([(sx-x0)/(x1-x0),(x1-sx)/(x1-x0),(sy-y0)/(y1-y0),(y1-sy)/(y1-y0),
                        out.radius/(x1-x0),out.radius/(y1-y0)],np.float32)
        out.observation=np.concatenate((out.retina.ravel(),(out.retina-self.previous_grid).ravel(),extra))
        self.previous_grid=out.retina.copy()
        out.status="Soul tracked briefly" if out.tracked_soul else "Visual lock"
        self.last=out
        return out

def annotate(seen):
    im=Image.fromarray(seen.image).convert("RGB")
    d=ImageDraw.Draw(im)
    if seen.arena:
        d.rectangle(seen.arena,outline=(60,255,130),width=2)
    for t in seen.threats:
        d.ellipse((t.x-t.radius,t.y-t.radius,t.x+t.radius,t.y+t.radius),outline=(255,195,65),width=2)
        d.line((t.x,t.y,t.x+t.vx*.15,t.y+t.vy*.15),fill=(65,205,255),width=2)
    if seen.soul:
        x,y=seen.soul
        q=seen.radius+4
        d.ellipse((x-q,y-q,x+q,y+q),outline=(255,80,150),width=2)
    return im
