# 💣 Super Bomberman AI (WIP)

Projeto em desenvolvimento de uma IA baseada em Deep Q-Learning para jogar **Super Bomberman (SNES)**, utilizando Python (PyTorch) e BizHawk (Lua).

---

## 🛠️ Pré-requisitos

Para executar este projeto, você precisará do seguinte ambiente:

1.  **Emulador:** [BizHawk](https://tasvideos.org/BizHawk) (Versão recente com suporte a Lua).
2.  **ROM:** *Super Bomberman (USA).sfc* (Deve ser a versão Americana para compatibilidade de memória)(em anexo)
3.  **Python:** Versão 3.9 ou superior(foi usada a 3.11).
4.  **Dependências Python:**
    Execute o comando abaixo para instalar as bibliotecas necessárias:
    ```bash
    pip install torch pandas numpy
    ```

---

## ⚙️ Configuração

### 1. Endereços de Memória (RAM Map)
O script Lua (`ia_controller.lua`) está configurado para a versão **USA** do jogo. Certifique-se de que os endereços no arquivo coincidem:

* **Posição X:** `0x004A`
* **Posição Y:** `0x0D54`
* **Vidas:** `0x0D7D`

### 2. Configuração de Auto-Reset (Save State)
A IA possui um sistema automático para reiniciar o jogo quando morre ou trava em menus. Para isso funcionar:

1.  Abra o jogo no BizHawk.
2.  Inicie a **Fase 1**.
3.  Assim que a fase começar (e você tiver 5 vidas), pressione **Shift + F1** no teclado.
    * Isso criará um *Save State* no **Slot 1**.
    * O script carregará este slot automaticamente quando necessário.

### 3. Caminhos de Arquivo
Verifique se a variável `BASE_DIR` no arquivo `ai_controller.py` aponta corretamente para a pasta onde estão os scripts Lua e CSV:

```python
BASE_DIR = r"D:\Caminho\Para\Seu\Projeto\lua_scripts"
