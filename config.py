"""
config.py — Configurações centrais do organizador.
EDITE OS CAMINHOS ABAIXO antes de rodar qualquer script.
"""

# ============================================================
# CAMINHOS — AJUSTE AQUI
# ============================================================

# HD externo onde estão as fotos originais (NÃO será modificado)
PASTA_FOTOS_ORIGEM = r"E:\fotos_trabalho"

# SSD interno onde as cópias organizadas serão salvas
PASTA_SAIDA = r"C:\fotos_organizadas"

# Onde salvar o banco de dados e arquivos temporários
PASTA_TRABALHO = r"C:\organizador_dados"

# ============================================================
# DESEMPENHO — AJUSTE SE TIVER PROBLEMAS DE MEMÓRIA
# ============================================================

# Tamanho do lote pra GPU. Diminua se der "CUDA out of memory".
# RTX 3060 (8GB): 16-24 | RTX 3070/3080 (10-12GB): 32-48
BATCH_SIZE = 24

# Quantas fotos processar antes de salvar progresso no banco
SALVAR_A_CADA = 100

# Extensões de imagem aceitas
EXTENSOES_IMAGEM = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tiff', '.tif', '.webp', '.bmp'}

# ============================================================
# MODELOS DE IA — não precisa mexer
# ============================================================

# Modelo CLIP pra embeddings visuais
MODELO_CLIP = "ViT-B-32"
PESOS_CLIP = "laion2b_s34b_b79k"

# Modelo Ollama pra descrições (llava 7B cabe folgado em 8GB VRAM)
MODELO_OLLAMA = "llava:7b"

# Quantos rostos no mínimo pra formar um cluster de pessoa
MIN_FOTOS_POR_PESSOA = 5

# Parâmetros do clustering de cenas
MIN_FOTOS_POR_GRUPO = 30  # grupos menores são considerados "outliers"

# ============================================================
# CAMINHOS DERIVADOS — não mexa
# ============================================================
import os

CAMINHO_BANCO = os.path.join(PASTA_TRABALHO, "fotos.db")
CAMINHO_THUMBS = os.path.join(PASTA_TRABALHO, "thumbs")
CAMINHO_LOGS = os.path.join(PASTA_TRABALHO, "logs")

os.makedirs(PASTA_TRABALHO, exist_ok=True)
os.makedirs(CAMINHO_THUMBS, exist_ok=True)
os.makedirs(CAMINHO_LOGS, exist_ok=True)
