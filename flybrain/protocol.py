"""Bounded loopback TCP: JSON header + PNG bytes, JSON-line action replies."""
import io,json,select,socket,time
import numpy as np
from PIL import Image
MAX_HEADER=8192
MAX_IMAGE=8*1024*1024
class FrameParser:
    def __init__(self):self.buf=bytearray();self.header=None
    def feed(self,data):
        self.buf.extend(data)
        if len(self.buf)>MAX_IMAGE+MAX_HEADER:raise ValueError("Frame too large")
        frames=[]
        while True:
            if self.header is None:
                end=self.buf.find(b"\n")
                if end<0:
                    if len(self.buf)>MAX_HEADER:raise ValueError("Header too large")
                    break
                if end>MAX_HEADER:raise ValueError("Header too large")
                h=json.loads(self.buf[:end]);del self.buf[:end+1]
                size=h.get("png_bytes")
                if h.get("v")!=1 or type(size) is not int or not 0<=size<=MAX_IMAGE:raise ValueError("Invalid frame")
                if type(h.get("seq")) is not int or type(h.get("episode")) is not int:raise ValueError("Missing sequence/episode")
                self.header=h
            size=self.header["png_bytes"]
            if len(self.buf)<size:break
            blob=bytes(self.buf[:size]);del self.buf[:size]
            frames.append((self.header,blob));self.header=None
        return frames
def decode_png(blob):
    with Image.open(io.BytesIO(blob)) as im:
        if im.width*im.height>8_000_000:raise ValueError("Image dimensions too large")
        return np.array(im.convert("RGB"))
class Server:
    def __init__(self,port=65432):
        self.listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.listener.bind(("127.0.0.1",port));self.listener.listen(1);self.listener.setblocking(False)
        self.client=None;self.parser=FrameParser();self.last_seq=-1
    def disconnect(self):
        if self.client:self.client.close()
        self.client=None;self.parser=FrameParser();self.last_seq=-1
    def poll(self,timeout=.01):
        if not self.client:
            r,_,_=select.select([self.listener],[],[],timeout)
            if r:self.client,_=self.listener.accept();self.client.setblocking(False);self.last_seq=-1
            return []
        r,_,_=select.select([self.client],[],[],timeout)
        if not r:return []
        data=self.client.recv(65536)
        if not data:self.disconnect();return []
        frames=self.parser.feed(data)
        for h,_ in frames:
            if h["seq"]<=self.last_seq:raise ValueError("Out-of-order observation")
            self.last_seq=h["seq"]
        return frames
    def reply(self,h,action):
        if not self.client:return
        raw=json.dumps(dict(v=1,seq=h["seq"],episode=h["episode"],action=int(action)),separators=(",",":")).encode()+b"\n"
        view=memoryview(raw);deadline=time.monotonic()+1
        while view:
            if time.monotonic()>deadline:raise TimeoutError("Action send timed out")
            _,ready,_=select.select([],[self.client],[],.01)
            if ready:
                count=self.client.send(view)
                if not count:raise ConnectionError("Disconnected")
                view=view[count:]
    def close(self):self.disconnect();self.listener.close()
