-- ==============================================================================
-- ARQUIVO: ia_controller.lua (Versão Final - Força Bruta para Bomba)
-- ==============================================================================

local estadoFile = "game_state.csv"
local acaoFile = "action.csv"
local header_written = false

-- ==============================================================================
-- ⚙️ CONFIGURAÇÃO DE MEMÓRIA (Seus Endereços)
-- ==============================================================================
local ADDR_X      = 0x004A  -- Posição X
local ADDR_Y      = 0x0D54  -- Posição Y
local ADDR_VIDAS  = 0x0D7D  -- Vidas
local ADDR_SCORE  = 0x0000  -- Score (Zero por enquanto)
-- ==============================================================================

function salvar_estado()
    local file = io.open(estadoFile, "a")
    if file then
        -- Leitura da memória
        local pX = mainmemory.read_u16_le(ADDR_X)
        local pY = mainmemory.read_u16_le(ADDR_Y)
        local vidas = mainmemory.readbyte(ADDR_VIDAS)
        local score = 0 
        
        -- O Python vai calcular se morreu baseada na queda de vidas
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
    
    -- Mostra coordenadas para confirmar que está lendo certo
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
        -- TESTE 1: TENTANDO APENAS O BOTÃO 'A'
        joypad.set({A=true}, 1) 
        
        gui.text(50, 50, "TESTANDO BOTAO A", "red")
        
        -- Mantemos o timing longo para garantir
        emu.frameadvance()
        joypad.set({A=true}, 1)
        emu.frameadvance()
        joypad.set({A=true}, 1)
        
    elseif acao == "wait" then
        -- Não faz nada
    end
end

-- Inicialização
print("🤖 Script Bomberman Iniciado!")
print("Endereços carregados: X=" .. string.format("%X", ADDR_X) .. " Y=" .. string.format("%X", ADDR_Y) .. " Vidas=" .. string.format("%X", ADDR_VIDAS))
gui.clearGraphics()

-- Loop Principal
while true do
    salvar_estado()
    local acao = ler_acao()
    
    if acao then 
        executar_acao(acao) 
    end
    
    emu.frameadvance()
end