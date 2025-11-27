import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import random
import math
import numpy as np

#CONFIGURAÇÕES
BASE_DIR = r"C:\Users\Dellarosa\Desktop\AI-Player\lua_scripts"
STATE_FILE = os.path.join(BASE_DIR, "game_state.csv")
ACTION_FILE = os.path.join(BASE_DIR, "action.csv")
MODEL_FILE = os.path.join(BASE_DIR, "bomberman_brain.pth")

ACTIONS = ["up", "down", "left", "right", "bomb", "wait"]

# Hiperparâmetros
INPUT_SIZE = 5   # X, Y, Score, Vidas, Morto
OUTPUT_SIZE = len(ACTIONS)
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000 # Aumentei para ela explorar mais

# REDE NEURAL (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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

#INICIALIZAÇÃO
policy_net = DQN(INPUT_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
steps_done = 0
last_line = 0

# Variáveis de Estado Anterior
prev_vidas = None
prev_x = None
prev_y = None

# Variável nova para detectar IA travada na parede
stuck_steps = 0 

if os.path.exists(MODEL_FILE):
    print(f" MODELO CARREGADO: {MODEL_FILE}")
    try:
        checkpoint = torch.load(MODEL_FILE)
        policy_net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        steps_done = checkpoint['steps']
        print(f"Nível: {steps_done} passos.")
    except:
        print("Erro ao carregar, criando novo.")
else:
    print("Iniciando a primeira vez...")

print(" IA Rodando com Punição de Parede Ativada!")

#LOOP
while True:
    if not os.path.exists(STATE_FILE):
        time.sleep(0.1)
        continue
        
    try:
        # Leitura Segura
        try:
            df = pd.read_csv(STATE_FILE, on_bad_lines='skip', dtype=str)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
        except pd.errors.EmptyDataError:
            time.sleep(0.1)
            continue

        if df.empty or len(df) <= last_line:
            time.sleep(0.01)
            continue

        # Prepara Estado
        last_row = df.iloc[-1]
        current_state_list = []
        for i in range(INPUT_SIZE):
            current_state_list.append(float(last_row.iloc[i]))

        state_tensor = torch.tensor(current_state_list, dtype=torch.float32).unsqueeze(0)
        
        cur_x = current_state_list[0]
        cur_y = current_state_list[1]
        cur_vidas = current_state_list[3]

        # --- CÁLCULO DE RECOMPENSA ---
        reward = 0.1 # Sobrevivência base
        
        if prev_vidas is not None:
            # 1. MORTE REAL
            if cur_vidas < prev_vidas:
                print("MORREU!(-100)")
                reward = -100.0
                stuck_steps = 0 # Reseta contador se morreu
            
            # 2. DETECÇÃO DE PAREDE (STUCK)
            elif prev_x is not None and prev_y is not None:
                # Se moveu menos de 1 pixel (está travada)
                if abs(cur_x - prev_x) < 1 and abs(cur_y - prev_y) < 1:
                    stuck_steps += 1
                    
                    # Punição progressiva: Quanto mais tempo parada, pior fica
                    reward -= 1.0 * (stuck_steps / 10) 
                    
                    # Se ficar presa por 30 frames (meio segundo), considera "Morte Técnica"
                    if stuck_steps > 30:
                        print("PRESA NA PAREDE! (-50 e Reset de Estratégia)")
                        reward = -50.0
                        stuck_steps = 0 # Reseta para não punir infinitamente
                else:
                    # Se moveu, reseta o contador de stuck
                    stuck_steps = 0
                    # Recompensa leve por se mover (incentiva exploração)
                    reward += 0.2

        # Atualiza memória
        prev_vidas = cur_vidas
        prev_x = cur_x
        prev_y = cur_y

        # Decisão e Ação
        action_idx = select_action(state_tensor, policy_net, steps_done)
        action_str = ACTIONS[action_idx.item()]
        steps_done += 1

        with open(ACTION_FILE, "w") as f:
            f.write(action_str)
            
        last_line = len(df)
        
        # Save
        if steps_done % 500 == 0:
            print(f"Salvando (Passo {steps_done})")
            torch.save({
                'model_state': policy_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'steps': steps_done
            }, MODEL_FILE)
        
        # Log
        if steps_done % 50 == 0:
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            modo = "Chutando" if random.random() < eps else "Pensando"
            print(f"Passo: {steps_done} | Vidas: {int(cur_vidas)} | Ação: {action_str} {modo} | Reward: {reward:.2f}")

    except Exception as e:
        print(f"Erro: {e}")
        time.sleep(1)