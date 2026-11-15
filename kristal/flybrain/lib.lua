-- Pixel observations plus confirmed bullet damage. F8 enables/disables control.
local Lib = {}
local ACTIONS = {{0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}}
function Lib:init()
    self.enabled = false
    self.status = "OFF / F8 to enable"
    self.seq,self.episode = 0,0
    self.hits,self.elapsed,self.action = 0,0,0
    self.port = Kristal.getLibConfig("flybrain","port") or 65432
    self.watchdog = Kristal.getLibConfig("flybrain","watchdog_seconds") or 15
    self.step_dt = math.max(1/120,math.min(1/15,Kristal.getLibConfig("flybrain","step_seconds") or 1/30))
    local ok,socket = pcall(require,"socket")
    self.socket_lib = ok and socket or nil
    if not ok then self.status = "LuaSocket unavailable" end
    Utils.hook(Soul,"doMovement",function(orig,soul,...)
        if self.enabled and self.connected and self.in_episode and self:isActive() then
            local move = ACTIONS[self.action+1] or ACTIONS[1]
            soul.moving_x,soul.moving_y = move[1],move[2]
            if move[1] ~= 0 or move[2] ~= 0 then
                if not soul:move(move[1],move[2],soul.speed*DTMULT) then
                    soul.moving_x,soul.moving_y = 0,0
                end
            end
            return
        end
        return orig(soul,...)
    end)
    Utils.hook(Soul,"onDamage",function(orig,soul,bullet,amount,...)
        if self.enabled and self.in_episode and amount and amount > 0 then self.hits=self.hits+1 end
        return orig(soul,bullet,amount,...)
    end)
    Utils.hook(Game,"update",function(orig,game,...)
        self:pump()
        if not self.enabled or not self.connected then return orig(game,...) end
        local active = self:isActive()
        if self.in_episode and not active then self.terminal=true end
        if self.terminal then
            if not self.awaiting then self:queueFrame(nil,true) end
            return
        end
        if not active then return orig(game,...) end
        if not self.in_episode then
            self.episode=self.episode+1
            self.in_episode=true
            self.hits,self.elapsed,self.action,self.applied=0,0,0,0
            self.need_frame=true
            return
        end
        if self.awaiting or self.need_frame then return end
        local old_dt,old_mult=DT,DTMULT
        DT,DTMULT=self.step_dt,self.step_dt*30
        self.applied=self.action
        orig(game,...)
        DT,DTMULT=old_dt,old_mult
        self.elapsed=self.step_dt
        if not self:isActive() then self.terminal=true else self.need_frame=true end
    end)
    -- Capture the native canvas only after love.draw has unbound it.
    Utils.hook(love,"draw",function(orig,...)
        orig(...)
        if self.enabled and self.connected and self.need_frame and not self.awaiting then
            if not SCREEN_CANVAS then self:disconnect("No screen canvas");return end
            local ok_image,data=pcall(function() return SCREEN_CANVAS:newImageData() end)
            if not ok_image then self:disconnect("Screenshot capture failed");return end
            local ok_png,encoded=pcall(function() return data:encode("png") end)
            if ok_png then self:queueFrame(encoded:getString(),false)
            else self:disconnect("Screenshot encode failed") end
            if data.release then data:release() end
        end
    end)
end
function Lib:isActive()
    local b=Game and Game.battle
    if not b or Game.state~="BATTLE" or b.state~="DEFENDING" then return false end
    local found_rudinn=false
    for _,enemy in ipairs(b.enemies or {}) do
        if enemy.id == "rudinn" and not enemy.defeated then found_rudinn=true;break end
    end
    if not found_rudinn then return false end
    local s=b.soul
    if not s or s.transitioning or not s.can_move then return false end
    local r,g,blue=s:getColor()
    return r and r>.5 and r>(g or 0)+.3 and r>(blue or 0)+.3
end
function Lib:disconnect(reason)
    if self.socket then self.socket:close() end
    self.socket=nil
    self.connected,self.enabled,self.in_episode=false,false,false
    self.awaiting,self.need_frame,self.terminal=false,false,false
    self.tx,self.rx,self.action=nil,"",0
    self.status=(reason or "OFF").." / F8"
    print("[flybrain] "..self.status)
end
function Lib:connect()
    if not self.socket_lib then self.status="LuaSocket unavailable; see README";print(self.status);return end
    self:disconnect("Connecting")
    self.enabled=true
    self.socket=self.socket_lib.tcp()
    self.socket:settimeout(0)
    self.socket:setoption("tcp-nodelay",true)
    self.started_connect=love.timer.getTime()
    self.last_activity=self.started_connect
    self.status="CONNECTING"
end
function Lib:pump()
    if not self.enabled or not self.socket then return end
    local now=love.timer.getTime()
    if not self.connected then
        local ok,err=self.socket:connect("127.0.0.1",self.port)
        if ok or (err and err:lower():find("already connected",1,true)) then
            self.connected=true;self.status="CONNECTED / F8 human control";self.last_activity=now
        elseif now-self.started_connect>3 then self:disconnect("Cannot reach Python server")
        elseif err~="timeout" and not (err and err:lower():find("progress",1,true)) then self:disconnect("Connect failed: "..tostring(err)) end
        return
    end
    if self.tx then
        local sent,err,partial=self.socket:send(self.tx,self.txpos)
        self.txpos=(sent or partial or (self.txpos-1))+1
        if self.txpos>#self.tx then self.tx=nil end
        if err and err~="timeout" then self:disconnect("Send failed");return end
    end
    for _=1,8 do
        local line,err,partial=self.socket:receive("*l",self.rx or "")
        if line then
            self.rx=""
            if #line>2048 then self:disconnect("Oversized action");return end
            local ok,cmd=pcall(JSON.decode,line)
            if ok and type(cmd)=="table" and cmd.v==1 and self.awaiting and cmd.seq==self.seq and cmd.episode==self.episode then
                local a=cmd.action
                if type(a)~="number" or a%1~=0 or a<0 or a>8 then self:disconnect("Invalid action");return end
                self.last_activity=now;self.action=a;self.awaiting=false
                if self.terminal then self.in_episode,self.terminal=false,false;self.action=0 end
            end
        else
            self.rx=partial or ""
            if #self.rx>2048 then self:disconnect("Oversized action");return end
            if err~="timeout" then self:disconnect("Python disconnected");return end
            break
        end
    end
    if self.awaiting and now-self.last_activity>self.watchdog then self:disconnect("Python timed out; human control restored") end
end
function Lib:queueFrame(png,terminal)
    if self.awaiting then return end
    self.seq=self.seq+1
    local hp=0
    for _,member in ipairs(Game.party or {}) do hp=hp+math.max(0,member:getHealth()) end
    local h={v=1,seq=self.seq,episode=self.episode,png_bytes=png and #png or 0,dt=self.elapsed,hits=self.hits,hp=hp,
        terminal=terminal,died=(Game.state=="GAMEOVER"),active=not terminal,action_applied=self.applied or 0,source="kristal"}
    self.tx=JSON.encode(h).."\n"..(png or "")
    self.txpos=1;self.awaiting=true;self.need_frame=false
    self.hits,self.elapsed=0,0;self.last_activity=love.timer.getTime()
end
function Lib:onKeyPressed(key,is_repeat)
    if key=="f8" and not is_repeat then
        if self.enabled then self:disconnect("Human control") else self:connect() end
        return true
    end

end
function Lib:postDraw()
    love.graphics.push("all")
    love.graphics.setColor(0,0,0,.8);love.graphics.rectangle("fill",4,4,410,22)
    love.graphics.setColor(.65,1,.65,1);love.graphics.setFont(Assets.getFont("main",16))
    love.graphics.print("FLY: "..(self.status or "OFF"),8,5)
    love.graphics.pop()
end
function Lib:unload() self:disconnect("Unloaded") end
function Lib:cleanup() self:disconnect("Cleaned up") end
return Lib
