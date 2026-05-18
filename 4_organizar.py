"""
4_organizar.py — Copia as fotos do HD externo pras pastas nomeadas no SSD.

Estrutura final:
  PASTA_SAIDA/
    reunioes/AAAA-MM-DD_originalname.jpg
    eventos/...
    produtos/...
    nao_classificadas/...  (grupos descartados ou outliers)
"""

import os
import sys
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import config


def parse_data_exif(data_str):
    """Tenta extrair AAAA-MM-DD da string EXIF (formato 'YYYY:MM:DD HH:MM:SS')."""
    if not data_str:
        return None
    try:
        return datetime.strptime(data_str[:10], '%Y:%m:%d').strftime('%Y-%m-%d')
    except Exception:
        return None


def nome_destino(caminho_origem, data, indice):
    """Gera nome legível pro arquivo de destino."""
    ext = Path(caminho_origem).suffix.lower()
    nome_orig = Path(caminho_origem).stem[:40]  # limita tamanho
    # Remove caracteres problemáticos
    nome_orig = ''.join(c if c.isalnum() or c in '-_' else '_' for c in nome_orig)
    prefixo = data or 'sem-data'
    return f"{prefixo}_{indice:05d}_{nome_orig}{ext}"


def main():
    if not os.path.exists(config.CAMINHO_BANCO):
        print(f"Banco não encontrado: {config.CAMINHO_BANCO}")
        sys.exit(1)

    conn = sqlite3.connect(config.CAMINHO_BANCO)
    c = conn.cursor()

    # Verifica se há grupos nomeados
    c.execute("SELECT COUNT(*) FROM grupos WHERE tipo='cena' AND nome NOT LIKE 'cena_%'")
    nomeados = c.fetchone()[0]
    if nomeados == 0:
        print("Nenhum grupo foi nomeado ainda. Rode primeiro: python 3_descobrir_grupos.py")
        sys.exit(1)
    print(f"{nomeados} grupos nomeados. Iniciando cópia...")

    # Cria pasta de saída
    os.makedirs(config.PASTA_SAIDA, exist_ok=True)
    os.makedirs(os.path.join(config.PASTA_SAIDA, 'nao_classificadas'), exist_ok=True)

    # Busca todas as fotos com seus grupos
    c.execute("""
        SELECT f.id, f.caminho, f.data_exif, g.nome
        FROM fotos f
        LEFT JOIN grupos g ON g.id = f.grupo_cena AND g.tipo = 'cena'
        WHERE f.erro IS NULL
        ORDER BY f.data_exif
    """)
    fotos = c.fetchall()
    print(f"Total a copiar: {len(fotos):,}")

    copiadas = 0
    puladas = 0
    erros = 0

    for idx, (foto_id, caminho, data_exif, nome_grupo) in enumerate(tqdm(fotos, unit="foto")):
        try:
            # Pasta destino baseada no nome do grupo
            if not nome_grupo or nome_grupo.startswith('cena_') or nome_grupo == 'descartar':
                pasta_grupo = 'nao_classificadas'
            else:
                pasta_grupo = nome_grupo

            pasta_destino = os.path.join(config.PASTA_SAIDA, pasta_grupo)
            os.makedirs(pasta_destino, exist_ok=True)

            data = parse_data_exif(data_exif)
            novo_nome = nome_destino(caminho, data, idx)
            destino = os.path.join(pasta_destino, novo_nome)

            if os.path.exists(destino):
                puladas += 1
                continue

            if not os.path.exists(caminho):
                erros += 1
                continue

            shutil.copy2(caminho, destino)
            copiadas += 1

        except KeyboardInterrupt:
            print("\nInterrompido. Você pode retomar rodando o script de novo.")
            break
        except Exception as e:
            erros += 1
            print(f"\nErro em {caminho}: {e}")

    print(f"\n=== Concluído ===")
    print(f"Copiadas: {copiadas:,}")
    print(f"Já existiam (puladas): {puladas:,}")
    print(f"Erros: {erros:,}")
    print(f"\nDestino: {config.PASTA_SAIDA}")
    print(f"Próximo passo (opcional): streamlit run 5_buscar.py")


if __name__ == "__main__":
    main()
