from pathlib import Path
import pytest
from lupa.luajit21 import LuaRuntime
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def lua():
    l=LuaRuntime(unpack_returned_tuples=True)
    l.execute('''
    DT=1/60;DTMULT=.5;clock=0;calls=0;human_moves=0
    fake={}
    function fake:settimeout() end
    function fake:setoption() end
    function fake:connect() return 1 end
    function fake:close() end
    function fake:send(data,from) return #data end
    function fake:receive(pattern,prefix)
      if self.response then local s=self.response;self.response=nil;return s end
      if self.drop then return nil,'closed','' end
      return nil,'timeout',prefix or ''
    end
    package.preload.socket=function() return {tcp=function() return fake end} end
    Utils={hook=function(target,key,hook)
      local orig=target[key] or function() end
      target[key]=function(...) return hook(orig,...) end
    end}
    JSON={encode=function(h) last_header=h;return 'HEADER' end,decode=function() return CMD end}
    Kristal={getLibConfig=function() return nil end}
    Soul={doMovement=function() human_moves=human_moves+1 end,onDamage=function() end}
    soul={speed=4,can_move=true,transitioning=false,getColor=function() return 1,0,0 end,
      move=function(self,x,y,s) moved={x,y,s};return true end}
    Game={state='BATTLE',battle={state='DEFENDING',soul=soul,enemies={{id='rudinn'}}},party={}}
    function Game:update()
      calls=calls+1;last_dt=DT;Soul.doMovement(soul)
      if hit then Soul.onDamage(soul,{},1);hit=false end
      if die then self.state='GAMEOVER';die=false end
    end
    love={timer={getTime=function() return clock end},draw=function() end}
    SCREEN_CANVAS={newImageData=function() return {encode=function() return {getString=function() return 'PNG' end} end,release=function() end} end}
    ''')
    l.globals().lib=l.execute((ROOT/'kristal/flybrain/lib.lua').read_text());l.execute('lib:init()');return l
def test_lua_compile():
    l=LuaRuntime()
    for p in (ROOT/'kristal').rglob('*.lua'):l.compile(p.read_text(),name=str(p))
def test_exact_step_damage_terminal(lua):
    lua.execute('lib:connect();Game:update();love.draw();Game:update()')
    assert lua.eval('calls')==0
    lua.execute('CMD={v=1,seq=lib.seq,episode=lib.episode,action=6};fake.response="{}";hit=true;Game:update();love.draw()')
    assert lua.eval('calls')==1 and lua.eval('moved[1]')==1 and lua.eval('moved[2]')==-1
    assert lua.eval('last_dt')==pytest.approx(1/30) and lua.eval('DT')==pytest.approx(1/60)
    assert lua.eval('last_header.hits')==1 and lua.eval('last_header.action_applied')==6
    lua.execute('CMD={v=1,seq=lib.seq,episode=lib.episode,action=2};fake.response="{}";hit=true;die=true;Game:update();Game:update()')
    assert lua.eval('last_header.terminal') and lua.eval('last_header.died')
    assert lua.eval('last_header.hits')==1 and lua.eval('last_header.png_bytes')==0
def test_stale_timeout(lua):
    lua.execute('lib:connect();Game:update();love.draw();CMD={v=1,seq=999,episode=1,action=2};fake.response="{}";Game:update()')
    assert lua.eval('calls')==0
    lua.execute('clock=20;Game:update()')
    assert not lua.eval('lib.enabled') and lua.eval('human_moves')==1
def test_disconnect(lua):
    lua.execute('lib:connect();Game:update();love.draw();fake.drop=true;Game:update()')
    assert not lua.eval('lib.enabled') and lua.eval('human_moves')==1

def test_only_rudinn_is_controlled(lua):
    lua.execute("Game.battle.enemies={{id='werewire'}};lib:connect();Game:update()")
    assert lua.eval('human_moves')==1 and not lua.eval('lib.in_episode')
