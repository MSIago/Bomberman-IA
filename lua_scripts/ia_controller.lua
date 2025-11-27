local estadoFile = "game_state.csv"
local acaoFile = "action.csv"
local header_written = false

-- Endereços prováveis (Teste com RAM Watch se não funcionar!)
local ADDR_X      = 0x004A
local ADDR_Y      = 0x0D54
local ADDR_VIDAS  = 0x1D23
local ADDR_TEMPO  = 0x0E1D

function salvar_estado()
    local file = io.open(estadoFile, "a")
    if file then
        local pX = mainmemory.read_u16_le(ADDR_X)
        local pY = mainmemory.read_u16_le(ADDR_Y)
        -- Score não é vital agora, mandamos 0
        local score = 0 
        local vidas = mainmemory.readbyte(ADDR_VIDAS)
        local morto = 0 -- Simplificado

        -- Escreve cabeçalho apenas se arquivo estiver vazio
        if not header_written then
            file:write("posX,posY,score,vidas,morto\n")
            header_written = true
        end
        
        -- Formato CSV simples
        file:write(string.format("%d,%d,%d,%d,%d\n", pX, pY, score, vidas, morto))
        file:close()
    end
end

function ler_acao()
    local file = io.open(acaoFile, "r")
    if not file then return nil end
    local acao = file:read("*l")
    file:close()
    return acao
end

function executar_acao(acao)
    if acao == nil then return end
    joypad.set({}) 
    if acao == "up" then joypad.set({Up = true}, 1)
    elseif acao == "down" then joypad.set({Down = true}, 1)
    elseif acao == "left" then joypad.set({Left = true}, 1)
    elseif acao == "right" then joypad.set({Right = true}, 1)
    elseif acao == "bomb" then joypad.set({B = true}, 1)
    end
end

while true do
    salvar_estado()
    local acao = ler_acao()
    if acao then executar_acao(acao) end
    emu.frameadvance()
end