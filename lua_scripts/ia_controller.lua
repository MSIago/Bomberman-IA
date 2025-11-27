-- ARQUIVO: ia_controller.lua

local estadoFile = "game_state.csv"
local acaoFile = "action.csv"
local header_written = false

--CONFIGURAÇÃO DE MEMÓRIA
local ADDR_X      = 0x004A  -- Posição X
local ADDR_Y      = 0x0D54  -- Posição Y
local ADDR_VIDAS  = 0x0D7D  -- Vidas

-- Variáveis para detectar se a IA travou (Game Over / Menu)
local last_x = 0
local last_y = 0
local frames_parado = 0

function salvar_estado()
    local file = io.open(estadoFile, "a")
    if file then
        local pX = mainmemory.read_u16_le(ADDR_X)
        local pY = mainmemory.read_u16_le(ADDR_Y)
        local vidas = mainmemory.readbyte(ADDR_VIDAS)
        local score = 0 
        local morto = 0 
        
        if not header_written then
            file:write("posX,posY,score,vidas,morto\n")
            header_written = true
        end
        
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
    
    joypad.set({}) -- Limpa comandos anteriores

    -- DEBUG VISUAL: Mostra a ação no canto da tela
    gui.text(10, 10, "IA: " .. acao, "white")
    
    -- Debug de coordenadas e vidas
    local curX = mainmemory.read_u16_le(ADDR_X)
    local curY = mainmemory.read_u16_le(ADDR_Y)
    local curVidas = mainmemory.readbyte(ADDR_VIDAS)
    gui.text(10, 25, string.format("X:%d Y:%d V:%d", curX, curY, curVidas), "yellow")

    -- Mapeamento dos Botões
    if acao == "up" then 
        joypad.set({Up = true}, 1)
    elseif acao == "down" then 
        joypad.set({Down = true}, 1)
    elseif acao == "left" then 
        joypad.set({Left = true}, 1)
    elseif acao == "right" then 
        joypad.set({Right = true}, 1)
        
    elseif acao == "bomb" then
        -- === BOTÃO DE BOMBA (A) ===
        joypad.set({A = true}, 1)
        
        gui.text(50, 50, "BOMBA (A)!")
        
        -- Timing estendido (3 frames) para garantir que solte a bomba
        emu.frameadvance()
        joypad.set({A = true}, 1)
        emu.frameadvance()
        joypad.set({A = true}, 1)
        
    elseif acao == "wait" then
        -- Fica parado
    end
end

-- === FUNÇÃO CRÍTICA: RESET AUTOMÁTICO ===
function verificar_travamento()
    local curX = mainmemory.read_u16_le(ADDR_X)
    local curY = mainmemory.read_u16_le(ADDR_Y)
    local vidas = mainmemory.readbyte(ADDR_VIDAS)

    -- Verifica se a posição mudou
    if curX == last_x and curY == last_y then
        frames_parado = frames_parado + 1
    else
        frames_parado = 0 -- Se moveu, reseta contador
    end

    last_x = curX
    last_y = curY
    
    -- Mostra contador na tela se estiver parado há um tempo
    if frames_parado > 60 then
        gui.text(10, 100, "Parado: " .. frames_parado, "orange")
    end

    -- RESET:
    -- 1. Se ficar parado por 300 frames (5 segundos) -> Assume que travou no Menu/Continue
    -- 2. OU Se tiver 0 vidas e ficar parado um pouco -> Assume que morreu
    if frames_parado > 300 or (vidas == 0 and frames_parado > 60) then
        print("💀 RESETANDO JOGO (Inatividade ou Morte detetada)...")
        gui.text(50, 100, "RESETANDO...", "red")
        
        -- Carrega o SaveState 1 quando perder todas as vidas
        savestate.loadslot(1)
        
        -- Reseta variáveis
        frames_parado = 0
        
        -- Limpa o arquivo CSV para não misturar episódios
        local file = io.open(estadoFile, "w")
        if file then file:close() end
        header_written = false
    end
end

-- Inicialização
print(" Iniciando!")
print("Endereços: X=" .. string.format("%X", ADDR_X) .. " Y=" .. string.format("%X", ADDR_Y) .. " Vidas=" .. string.format("%X", ADDR_VIDAS))
gui.clearGraphics()

-- Loop Principal
while true do
    salvar_estado()
    
    -- Verifica se precisa resetar antes de mover
    verificar_travamento()
    
    local acao = ler_acao()
    if acao then 
        executar_acao(acao) 
    end
    
    emu.frameadvance()
end