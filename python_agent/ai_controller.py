import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import random
import math
import numpy as np

# --- CONFIGURAÇÕES ---
# Ajuste o caminho se necessário (use r"" antes da string)
BASE_DIR = r"C:\Users\Dellarosa\Desktop\AI-Player\lua_scripts"
STATE_FILE = os.path.join(BASE_DIR, "game_state.csv")
ACTION_FILE = os.path.join(BASE_DIR, "action.csv")

# Ações possíveis
ACTIONS = ["up", "down", "left", "right", "bomb", "wait"]

# Hiperparâmetros
INPUT_SIZE = 5   # X, Y, Score, Vidas, Morto
OUTPUT_SIZE = len(ACTIONS)
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

def select_action(state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(OUTPUT_SIZE)]], dtype=torch.long)

# Inicialização
policy_net = DQN(INPUT_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
steps_done = 0
last_line = 0

print("🚀 Cérebro da IA Conectado! Aguardando dados...")

while True:
    # 1. Verifica se o arquivo existe
    if not os.path.exists(STATE_FILE):
        time.sleep(0.1)
        continue
        
    try:
        # 2. Leitura Segura: Ignora linhas quebradas
        try:
            df = pd.read_csv(STATE_FILE, on_bad_lines='skip')
        except pd.errors.EmptyDataError:
            time.sleep(0.1)
            continue

        if df.empty or len(df) <= last_line:
            time.sleep(0.01)
            continue

        # 3. CONVERSÃO FORÇADA (O Segredo para não dar erro)
        # Pega apenas a última linha
        last_row = df.iloc[-1]
        
        # Cria uma lista segura, forçando conversão para float um por um
        # Se der erro na conversão, vira 0.0
        safe_state = []
        for i in range(INPUT_SIZE): # 0 a 4
            try:
                # Tenta pegar pela posição da coluna (0=X, 1=Y, etc)
                val = pd.to_numeric(last_row.iloc[i], errors='coerce')
                if np.isnan(val): val = 0.0
                safe_state.append(float(val))
            except:
                safe_state.append(0.0)

        # 4. Transforma em Tensor (agora garantido que é float)
        state = torch.tensor(safe_state, dtype=torch.float32).unsqueeze(0)

        # 5. Decide Ação
        action_idx = select_action(state, policy_net, steps_done)
        action_str = ACTIONS[action_idx.item()]
        steps_done += 1

        # 6. Escreve Ação
        with open(ACTION_FILE, "w") as f:
            f.write(action_str)
            
        last_line = len(df)
        
        # Log simplificado para debug
        if steps_done % 50 == 0:
            print(f"Passo: {steps_done} | Ação: {action_str} | Estado Lido: {safe_state}")

    except Exception as e:
        # Mostra o erro mas não para o programa
        print(f"Aviso (tentando recuperar): {e}")
        time.sleep(1)