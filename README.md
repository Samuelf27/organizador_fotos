# Organizador de Fotos com IA Local
Kit completo para organizar 50k+ fotos usando IA rodando localmente (sem nuvem).
## O que esse kit faz
1. **Analisa cada foto** do seu HD: extrai data/hora (EXIF), detecta e agrupa rostos, gera "impressão digital" visual (embedding CLIP) e descrição em linguagem natural.
2. **Descobre grupos** automaticamente: agrupa fotos parecidas em clusters. Você revisa e nomeia os grupos (ex: "reuniões 2023", "evento cliente X", "fotos de produto").
3. **Organiza em pastas**: copia as fotos do HD externo pro SSD em pastas nomeadas.
4. **App de busca**: interface web local pra buscar por rosto, descrição, data, etc.
## Requisitos confirmados
- ✅ Windows com RTX 3060/3070/3080 (8-12GB VRAM)
- ✅ Python já instalado
- ✅ Espaço sobrando no SSD interno
- ✅ HD externo com as fotos
## Antes de começar — EDITE O `config.py`
Abra o arquivo `config.py` e ajuste os caminhos:
```python
PASTA_FOTOS_ORIGEM = r"E:\fotos_trabalho"           # seu HD externo
PASTA_SAIDA = r"C:\fotos_organizadas"               # SSD interno
```
## Como executar (NA ORDEM)
### Passo 1 — Instalar dependências (uma vez só)
Clique duas vezes em `1_instalar.bat` e espere terminar. Vai instalar:
- Bibliotecas Python (PyTorch com CUDA, CLIP, InsightFace, etc.)
- Ollama (servidor local de IA)
- Modelo Llava 7B (~4GB) para descrever fotos
Tempo: 20-40 min dependendo da internet.
### Passo 2 — Analisar todas as fotos
```
python 2_analisar.py
```
Esse é o passo demorado. Pra 50k fotos numa RTX 3060/3070/3080, conte com **1-3 dias rodando em background**. Pode pausar e retomar — o script salva progresso e pula fotos já processadas.
Deixa rodando à noite. No final você terá um arquivo `fotos.db` (SQLite) com tudo.
### Passo 3 — Descobrir grupos e nomeá-los
```
python 3_descobrir_grupos.py
```
O script vai agrupar fotos similares e te mostrar uma amostra de cada grupo no navegador. Você dá um nome pra cada grupo ("reunião", "evento", "produto", "pessoal", "descartar", etc.). Tempo estimado: 1-2h de revisão sua.
### Passo 4 — Organizar em pastas
```
python 4_organizar.py
```
Copia as fotos do HD externo pro SSD em pastas com os nomes que você definiu. Tempo: depende da velocidade do HD (algumas horas pra 50k fotos).
### Passo 5 — App de busca (opcional, use quando quiser)
```
streamlit run 5_buscar.py
```
Abre uma página local no navegador onde você pode buscar por descrição, rosto, data, etc.
## Problemas comuns
**"CUDA out of memory"**: reduza `BATCH_SIZE` no `config.py` de 32 pra 16 ou 8.
**Ollama não inicia**: rode `ollama serve` num terminal separado antes do passo 2.
**Travou no meio**: relança o `2_analisar.py`, ele retoma de onde parou.
**Quero parar e voltar depois**: Ctrl+C no terminal. Tudo salvo no `fotos.db`.
## Estrutura final esperada
```
C:\fotos_organizadas\
├── reunioes\
│   ├── 2023-03-15_reuniao_001.jpg
│   └── ...
├── eventos\
├── produtos\
├── pessoal\
└── nao_classificadas\
```
