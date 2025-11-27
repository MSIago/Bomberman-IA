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
BASE_DIR = r"C:\Users\Dellarosa\Desktop\AI-Player\lua_scripts"
STATE_FILE = os.path.join(BASE_DIR, "game_state.csv")
ACTION_FILE = os.path.join(BASE_DIR, "action.csv")
MODEL_FILE = os.path.join(BASE_DIR, "bomberman_brain.pth") # <--- ARQUIVO DO CÉREBRO

ACTIONS = ["up", "down", "left", "right", "bomb", "wait"]

# Hiperparâmetros
INPUT_SIZE = 5   # X, Y, Score, Vidas, Morto
OUTPUT_SIZE = len(ACTIONS)
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000 # Aumentei para ela explorar mais tempo

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
    # Epsilon Decay: Diminui a aleatoriedade com o tempo
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    # Se o random for maior que epsilon, usa o CÉREBRO (O que aprendeu)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Senão, chuta uma ação aleatória (Exploração)
        return torch.tensor([[random.randrange(OUTPUT_SIZE)]], dtype=torch.long)

# --- INICIALIZAÇÃO ---
policy_net = DQN(INPUT_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
steps_done = 0
last_line = 0

# === PARTE NOVA: CARREGAR O CÉREBRO ===
if os.path.exists(MODEL_FILE):
    print(f"🧠 Cérebro encontrado: {MODEL_FILE}")
    try:
        checkpoint = torch.load(MODEL_FILE)
        policy_net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        steps_done = checkpoint['steps']
        print(f"✅ Carregado com sucesso! Continuando do passo {steps_done}...")
    except Exception as e:
        print(f"⚠️ Erro ao carregar (criando novo): {e}")
else:
    print("✨ Nenhum cérebro salvo. Criando uma IA nova do zero...")
# =======================================

print("🚀 IA Conectada e pronta para treinar!")

while True:
    if not os.path.exists(STATE_FILE):
        time.sleep(0.1)
        continue
        
    try:
        # Leitura Blindada do CSV
        try:
            df = pd.read_csv(STATE_FILE, on_bad_lines='skip')
        except pd.errors.EmptyDataError:
            time.sleep(0.1)
            continue

        if df.empty or len(df) <= last_line:
            time.sleep(0.01)
            continue

        # Conversão forçada para números
        last_row = df.iloc[-1]
        safe_state = []
        for i in range(INPUT_SIZE):
            try:
                val = pd.to_numeric(last_row.iloc[i], errors='coerce')
                if np.isnan(val): val = 0.0
                safe_state.append(float(val))
            except:
                safe_state.append(0.0)

        state = torch.tensor(safe_state, dtype=torch.float32).unsqueeze(0)

        # Decide Ação
        action_idx = select_action(state, policy_net, steps_done)
        action_str = ACTIONS[action_idx.item()]
        steps_done += 1

        # Escreve Ação
        with open(ACTION_FILE, "w") as f:
            f.write(action_str)
            
        last_line = len(df)
        
        # === PARTE NOVA: SALVAR O CÉREBRO ===
        # Salva a cada 500 passos (ajuste se quiser)
        if steps_done % 500 == 0:
            print(f"💾 Salvando progresso no passo {steps_done}...")
            torch.save({
                'model_state': policy_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'steps': steps_done
            }, MODEL_FILE)
        # ====================================
        
        if steps_done % 100 == 0:
            # Mostra se está explorando (Random) ou Usando o Cérebro (AI)
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            tipo = "🎲 Chute" if random.random() < eps else "🧠 Inteligência"
            print(f"Passo: {steps_done} | Vidas: {int(safe_state[3])} | Modo: {tipo}")

    except Exception as e:
        print(f"Erro: {e}")
        time.sleep(1)