# Snake AI • Neuroevolução com Algoritmos Genéticos

Implementação de inteligência artificial autônoma para o clássico jogo **Snake**, utilizando **Redes Neurais Artificiais (MLP)** otimizadas puramente por **Algoritmos Genéticos (GA)**, sem retropropagação de erro (*zero backpropagation*), sem aprendizado por reforço tabular/Q-learning e sem qualquer intervenção ou demonstração humana.

---

## Destaques do Projeto

- **Autonomia Total**: Os pesos da rede neural evoluem a partir de pesos puramente aleatórios via seleção natural.
- **Recorde no Tabuleiro 15x15**: **204 maçãs** comidas de 222 possíveis (**92% do tabuleiro preenchido** pela serpente).
- **Consistência Estatística**: 96% das partidas superam 71 maçãs (média de 183.3 ± 37.5 maçãs em benchmark de 100 partidas independentes).
- **Generalização Zero-Shot (20x20)**: O modelo treinado em 15x15 foi testado diretamente em uma grade 20x20 (400 células) sem nenhum retreinamento e atingiu **380 maçãs** (**95.7% de ocupação**), provando a emergência de uma heurística abstrata e invariante à escala.
- **Simulação de Alta Performance**: Motor vetorial e BFS Flood-Fill compilados em C/LLVM via Numba JIT, atingindo mais de **18.000 passos por segundo** em simulação acelerada.

---

## Requisitos de Sistema

- **Versão do Python**: **Python 3.12** (compatível com Python 3.10 a 3.12).
- **Sistemas Operacionais Suportados**:
  - Windows 10 / 11 (64-bit)
  - Linux (Ubuntu 20.04+, Debian, Fedora, Arch)
  - macOS (Apple Silicon M1/M2/M3 e Intel)
- **Hardware Recomendado**:
  - **CPU**: Qualquer processador quad-core moderno (o modo CPU utiliza multiprocessamento paralelo).
  - **GPU (Opcional)**: Placa de vídeo NVIDIA com suporte a CUDA para avaliação populacional em lote ultra-rápida via PyTorch (`torch.bmm`).

---

## Instalação e Configuração

### 1. Clonar ou Acessar o Repositório

Navegue até a pasta raiz do projeto:
```bash
cd snake_ga
```

### 2. Criar o Ambiente Virtual

Recomenda-se o uso de um ambiente virtual isolado:
```bash
python -m venv .venv
```

### 3. Ativar o Ambiente Virtual

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Prompt de Comando - CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 4. Instalar as Dependências

Instale os pacotes listados em `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Nota sobre PyTorch e Aceleração por GPU (CUDA):**
> Caso possua uma GPU NVIDIA e deseje utilizar aceleração por hardware durante o treinamento, instale a versão do PyTorch com suporte a CUDA:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```
> Caso utilize apenas CPU, o comando padrão `pip install -r requirements.txt` já instala o PyTorch com suporte a CPU.

---

## Como Assistir a IA Jogando (`play.py`)

O script `play.py` carrega o melhor genoma treinado (`checkpoints/best_overall.npy`) e renderiza a partida em tempo real utilizando Pygame.

### Execução Padrão

```bash
python play.py
```

### Argumentos e Configurações

| Argumento | Padrão | Descrição |
| :--- | :---: | :--- |
| `--checkpoint <path>` | `checkpoints/best_overall.npy` | Caminho do modelo `.npy` ou checkpoint `.npz` a carregar. |
| `--width <N>` | `15` | Largura da grade em células. |
| `--height <N>` | `15` | Altura da grade em células. |
| `--fps <N>` | `12` | Velocidade inicial da simulação em quadros por segundo. |
| `--cell-size <N>` | `40` | Tamanho de cada célula em pixels na tela. |
| `--no-rays` | `False` | Oculta as linhas dos 8 raios de visão da cobra. |
| `--shield` | `False` | Ativa o filtro heurístico de sobrevivência (action masking). |

### Exemplos Úteis

- **Assistir a IA em alta velocidade (60 FPS)**:
  ```bash
  python play.py --fps 60
  ```
- **Testar a generalização no tabuleiro 20x20**:
  ```bash
  python play.py --width 20 --height 20 --cell-size 32 --fps 30
  ```
- **Assistir sem as linhas de visão**:
  ```bash
  python play.py --no-rays
  ```

### Controles Interativos no Teclado

Durante a execução da janela gráfica do jogo:

| Tecla | Ação |
| :---: | :--- |
| `[ESPAÇO]` | Pausar / Continuar a partida |
| `[SETA CIMA]` | Aumentar a velocidade da simulação (+4 FPS) |
| `[SETA BAIXO]` | Diminuir a velocidade da simulação (-4 FPS) |
| `[V]` | Ligar / Desligar a exibição dos raios de percepção |
| `[R]` | Reiniciar a partida imediatamente |
| `[ESC]` | Encerrar e fechar a janela |

---

## Como Executar o Treinamento (`training.train`)

O módulo `training.train` gerencia o ciclo evolutivo populacional, avaliação de aptidão, seleção, mutação, geração de gráficos e salvamento periódico de checkpoints.

### 1. Retomar Treinamento Existente (Recomendado)

Para continuar o treinamento a partir do último checkpoint salvo (`checkpoints/checkpoint_latest.npz`):
```bash
python -m training.train --resume
```

### 2. Treinamento Contínuo (Modo Infinito)

Roda o treinamento indefinidamente, permitindo pausar com `Ctrl+C` a qualquer momento sem perder o progresso:
```bash
python -m training.train --resume --infinite
```

### 3. Iniciar Novo Treinamento do Zero

- **Utilizando GPU (NVIDIA CUDA)**:
  ```bash
  python -m training.train --pop 200 --gens 100 --device cuda
  ```
- **Utilizando CPU (Multiprocessamento Paralelo)**:
  ```bash
  python -m training.train --pop 200 --gens 100 --device cpu
  ```

### Argumentos de Treinamento

| Argumento | Padrão | Descrição |
| :--- | :---: | :--- |
| `--pop <N>` | `200` | Tamanho da população por geração. |
| `--gens <N>` | `50` | Quantidade de gerações a evoluir. |
| `--infinite` | `False` | Executa o treino continuamente até interrupção com `Ctrl+C`. |
| `--device <d>` | `cuda` | Dispositivo de cálculo (`cuda` ou `cpu`). |
| `--grid <N>` | `15` | Dimensão da grade quadrada (padrão: 15 para 15x15). |
| `--multi-grid` | `False` | Avalia cada cobra em dois tabuleiros (10x10 e 15x15) para evitar sobreajuste. |
| `--trials <N>` | `1` | Quantidade de partidas por indivíduo para amortizar o fator sorte. |
| `--save-every <N>` | `10` | Intervalo de gerações para salvar checkpoints periódicos completos. |
| `--resume [alvo]` | `None` | Retoma o treino (`latest`, número da geração ex: `50`, ou caminho de arquivo). |
| `--from-gen <N>` | `None` | Atalho para retomar a partir de uma geração específica. |
| `--mutation-rate <F>`| `0.035` | Taxa de mutação por gene (3.5%). |
| `--mutation-sigma <F>`| `0.15` | Desvio padrão da perturbação gaussiana $\mathcal{N}(0, \sigma^2)$. |
| `--decay-mutation` | `False` | Reduz a intensidade da mutação gradativamente ao longo do treino. |

### Salvamento Seguro contra Interrupções

Ao pressionar `Ctrl+C` durante o treino:
1. O processo conclui a geração corrente ou registra o último estado estável.
2. Salva automaticamente `checkpoints/checkpoint_interrupted.npz` e atualiza `checkpoint_latest.npz`.
3. Atualiza o gráfico de curvas em `checkpoints/fitness_curve.png` e o arquivo `checkpoints/training_stats.json`.

---

## Como Executar o Benchmark Estatístico (`benchmark.py`)

O script `benchmark.py` executa centenas de partidas consecutivas em modo *headless* (sem interface gráfica), coletando telemetria detalhada para análise científica e validação de consistência.

### 1. Benchmark do Modelo Campeão

Executa 100 partidas independentes e imprime relatório completo:
```bash
python benchmark.py -n 100
```

### 2. Teste de Generalização em Tabuleiro 20x20

```bash
python benchmark.py --width 20 --height 20 -n 100
```

### 3. Comparação Direta entre Dois Modelos

Compara lado a lado dois arquivos de pesos diferentes (ex.: geração 500 contra o modelo final):
```bash
python benchmark.py -c checkpoints/best_overall.npy --compare checkpoints/best_gen_0500.npy -n 50
```

### 4. Exportação dos Dados em JSON

```bash
python benchmark.py -n 100 -o benchmark_results.json
```

---

## Estrutura da Solução Genética

### 1. Codificação do Cromossomo (951 Genes)

Cada indivíduo é definido por um vetor unidimensional contínuo de **951 números de ponto flutuante** (`float32`), que correspondem a todos os pesos sinápticos e vieses da rede neural:

$$\text{MLP: } [32 \text{ entradas}] \xrightarrow{W_1, b_1} [20 \text{ Tanh}] \xrightarrow{W_2, b_2} [12 \text{ Tanh}] \xrightarrow{W_3, b_3} [3 \text{ Saídas (Argmax)}]$$

- **Camada 1**: $(32 \times 20) + 20 = 660$ genes
- **Camada 2**: $(20 \times 12) + 12 = 252$ genes
- **Camada 3**: $(12 \times 3) + 3 = 39$ genes
- **Total**: **951 parâmetros**

### 2. Vetor de Percepção (32 Sensores Espaciais)

A cobra não recebe pixels brutos nem posições absolutas, garantindo invariância de escala:
1. **8 Raios de Visão em 45°** ($8 \times 3 = 24$ entradas): Distâncias normalizadas $[0.0, 1.0]$ para a Parede, para o Próprio Corpo e para a Fruta em 8 orientações cardeais e intercardeais.
2. **Orientação e Ângulo Alvo** (4 entradas): Vetor de movimento atual $(dx, dy)$ e ângulo trigonométrico relativo da comida $(\sin\theta, \cos\theta)$.
3. **Flood-Fill BFS de Espaço Livre** (3 entradas): Fração de células contíguas acessíveis caso a cobra vire para a Esquerda, siga Reto ou vire para a Direita.
4. **Saturação do Tabuleiro** (1 entrada): Proporção do tabuleiro ocupado ($\text{tamanho} / N_{\text{células}}$).

### 3. Função de Avaliação (Fitness) e Solução de Vícios

Para eliminar comportamentos indesejados (como girar em círculos infinitos para acumular tempo), foram implementadas as seguintes diretrizes:
- **Crescimento Cúbico por Fruta**: Recompensa baseada em $\text{score}^3 \times 1000.0$. Uma fruta extra supera ordens de magnitude de qualquer ganho por passos vivos.
- **Trava de Farm de Passos**: Se a cobra não come nenhuma maçã ($\text{score} = 0$), a pontuação de sobrevivência é limitada ao teto de 20 pontos.
- **Punição por Canibalismo (50%)**: Colidir contra o próprio corpo corta o valor de todas as maçãs pela metade.
- **Detecção de Ciclos Repetidos**: Penalidade acumulativa de $-8.0$ por passar pela mesma célula na mesma direção.
- **Limite de Fome**: Morte imediata caso a cobra gaste mais de $150 + \text{comprimento}$ passos sem encontrar comida.

### 4. Operadores Genéticos

- **Seleção**: Torneio com $k = 5$ competidores.
- **Cruzamento (Crossover)**: Crossover uniforme com 50% de probabilidade de herança de cada pai por gene.
- **Mutação**: Mutação gaussiana aditiva $\Delta w \sim \mathcal{N}(0, 0.15^2)$ com taxa de 3.5% por gene.
- **Substituição / Elitismo**: Elitismo estrito com preservação direta do top 3% a 5% da população, garantindo convergência monótona dos recordes.

---

## Estrutura do Repositório

```text
snake_ga/
│
├── ai/
│   ├── ga.py              # Algoritmo Genético (seleção, crossover, mutação, elitismo)
│   ├── genome.py          # Classe do indivíduo (manipulação e persistência de genes)
│   ├── network.py         # Arquitetura da Rede Neural Feedforward PyTorch (SnakeNet)
│   └── perception.py      # Extração dos 32 sensores espaciais e aceleração Numba JIT
│
├── game/
│   ├── snake_env.py       # Lógica e regras do ambiente Snake
│   └── renderer.py        # Interface gráfica interativa em Pygame
│
├── training/
│   ├── fitness.py         # Função de avaliação de aptidão matemática
│   ├── evaluator.py       # Avaliador CPU (inferência C/Numba paralela)
│   ├── gpu_evaluator.py   # Avaliador GPU (inferência paralela com torch.bmm)
│   └── train.py           # Loop principal de treino e persistência de checkpoints
│
├── utils/
│   └── flood_fill.py      # Algoritmo BFS compilado em Numba para cálculo de área livre
│
├── checkpoints/           # Pesos do modelo (.npy), checkpoints (.npz) e estatísticas
│   ├── best_overall.npy   # Genoma do melhor indivíduo histórico (campeão)
│   ├── fitness_curve.png  # Gráfico da curva de evolução
│   └── training_stats.json# Histórico de fitness e scores por geração
│
├── play.py                # Script para assistir a IA jogando ao vivo
├── benchmark.py           # Script para testes em lote e compilação estatística
├── requirements.txt       # Lista de dependências Python
└── README.md              # Documentação completa do projeto
```

---

## Autores

- **Rick Ribeiro**
- **Vitória Milhomem**

---

## Licença

Projeto desenvolvido para fins acadêmicos e de pesquisa em Inteligência Artificial e Otimização Bioinspirada. Uso e distribuição permitidos para fins educacionais.
