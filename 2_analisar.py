"""
2_analisar.py — Varre o HD, processa cada foto e salva tudo num banco SQLite.

Para cada foto extrai:
- Metadados EXIF (data, hora, câmera, GPS)
- Embedding visual (CLIP) — 512 dimensões
- Rostos detectados + embeddings faciais (InsightFace)
- Descrição em linguagem natural (Llava via Ollama)

É retomável: se interromper (Ctrl+C), na próxima execução pula o que já foi feito.
"""

import os
import sys
import sqlite3
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm
import exifread
import torch
import open_clip
import ollama
from insightface.app import FaceAnalysis
import pillow_heif

pillow_heif.register_heif_opener()  # suporte a HEIC do iPhone

import config

# ============================================================
# Logging
# ============================================================
log_file = os.path.join(config.CAMINHO_LOGS, f"analise_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ============================================================
# Banco de dados
# ============================================================
def criar_banco():
    """Cria as tabelas se ainda não existirem."""
    conn = sqlite3.connect(config.CAMINHO_BANCO)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS fotos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caminho TEXT UNIQUE NOT NULL,
            hash_md5 TEXT,
            largura INTEGER,
            altura INTEGER,
            tamanho_bytes INTEGER,
            data_exif TEXT,
            camera TEXT,
            gps_lat REAL,
            gps_lon REAL,
            embedding_clip BLOB,
            descricao TEXT,
            num_rostos INTEGER DEFAULT 0,
            grupo_cena INTEGER,
            processada_em TEXT,
            erro TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS rostos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            foto_id INTEGER REFERENCES fotos(id),
            embedding BLOB,
            bbox TEXT,
            qualidade REAL,
            grupo_pessoa INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS grupos (
            id INTEGER PRIMARY KEY,
            tipo TEXT,
            nome TEXT,
            total_fotos INTEGER
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_caminho ON fotos(caminho)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_grupo_cena ON fotos(grupo_cena)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_grupo_pessoa ON rostos(grupo_pessoa)")
    conn.commit()
    return conn


def ja_processada(conn, caminho):
    c = conn.cursor()
    c.execute("SELECT 1 FROM fotos WHERE caminho = ? AND embedding_clip IS NOT NULL", (caminho,))
    return c.fetchone() is not None


# ============================================================
# Extração de metadados EXIF
# ============================================================
def extrair_exif(caminho):
    """Retorna dict com data, câmera e GPS quando disponíveis."""
    info = {'data_exif': None, 'camera': None, 'gps_lat': None, 'gps_lon': None}
    try:
        with open(caminho, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='GPS GPSLongitude')

        if 'EXIF DateTimeOriginal' in tags:
            info['data_exif'] = str(tags['EXIF DateTimeOriginal'])
        elif 'Image DateTime' in tags:
            info['data_exif'] = str(tags['Image DateTime'])

        if 'Image Model' in tags:
            info['camera'] = str(tags['Image Model'])

        # GPS
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            info['gps_lat'] = _gps_para_decimal(tags['GPS GPSLatitude'], str(tags.get('GPS GPSLatitudeRef', 'N')))
            info['gps_lon'] = _gps_para_decimal(tags['GPS GPSLongitude'], str(tags.get('GPS GPSLongitudeRef', 'E')))
    except Exception as e:
        log.debug(f"EXIF falhou em {caminho}: {e}")
    return info


def _gps_para_decimal(coord, ref):
    try:
        d, m, s = [float(x.num) / float(x.den) for x in coord.values]
        decimal = d + m/60 + s/3600
        if ref in ('S', 'W'):
            decimal = -decimal
        return decimal
    except Exception:
        return None


# ============================================================
# CLIP (embeddings visuais)
# ============================================================
def carregar_clip(device):
    log.info(f"Carregando CLIP {config.MODELO_CLIP}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODELO_CLIP, pretrained=config.PESOS_CLIP
    )
    model = model.to(device).eval()
    return model, preprocess


def embedding_clip(imagem_pil, model, preprocess, device):
    """Retorna vetor numpy 512-d."""
    with torch.no_grad():
        tensor = preprocess(imagem_pil).unsqueeze(0).to(device)
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)[0]


# ============================================================
# InsightFace (rostos)
# ============================================================
def carregar_insightface():
    log.info("Carregando InsightFace (detecção + reconhecimento)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detectar_rostos(imagem_np, app):
    """Retorna lista de dicts com bbox, embedding (512-d) e qualidade."""
    try:
        faces = app.get(imagem_np)
        return [
            {
                'bbox': f.bbox.tolist(),
                'embedding': f.normed_embedding.astype(np.float32),
                'qualidade': float(f.det_score),
            }
            for f in faces if f.det_score > 0.5
        ]
    except Exception as e:
        log.debug(f"Detecção facial falhou: {e}")
        return []


# ============================================================
# Llava (descrição em linguagem natural via Ollama)
# ============================================================
def descrever_foto(caminho_thumb):
    """Pede uma descrição curta da foto ao Llava local."""
    try:
        r = ollama.chat(
            model=config.MODELO_OLLAMA,
            messages=[{
                'role': 'user',
                'content': (
                    'Descreva esta foto em até 30 palavras em português. '
                    'Foque em: tipo de cena (reunião, evento, retrato, produto, paisagem, documento, etc), '
                    'objetos principais, ambiente (interno/externo), número aproximado de pessoas. '
                    'Seja objetivo, sem floreio.'
                ),
                'images': [caminho_thumb]
            }],
            options={'temperature': 0.1, 'num_predict': 80}
        )
        return r['message']['content'].strip()
    except Exception as e:
        log.debug(f"Llava falhou: {e}")
        return None


# ============================================================
# Utilitários
# ============================================================
def hash_arquivo(caminho, chunk=65536):
    """MD5 dos primeiros megabytes pra detectar duplicatas rapidamente."""
    h = hashlib.md5()
    try:
        with open(caminho, 'rb') as f:
            for _ in range(16):  # ~1MB
                data = f.read(chunk)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except Exception:
        return None


def gerar_thumb(imagem_pil, caminho_destino, tamanho=512):
    """Salva uma versão pequena pra alimentar o Llava (mais rápido)."""
    img = imagem_pil.copy()
    img.thumbnail((tamanho, tamanho))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(caminho_destino, 'JPEG', quality=85)


def listar_fotos(pasta):
    """Gerador de caminhos de todas as imagens na pasta (recursivo)."""
    for raiz, _, arquivos in os.walk(pasta):
        for nome in arquivos:
            if Path(nome).suffix.lower() in config.EXTENSOES_IMAGEM:
                yield os.path.join(raiz, nome)


# ============================================================
# Loop principal
# ============================================================
def main():
    if not os.path.exists(config.PASTA_FOTOS_ORIGEM):
        log.error(f"Pasta de origem não existe: {config.PASTA_FOTOS_ORIGEM}")
        log.error("Edite o config.py e ajuste PASTA_FOTOS_ORIGEM")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        log.warning("CUDA não detectado! Vai rodar em CPU (MUITO mais lento).")
    else:
        log.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    conn = criar_banco()
    clip_model, clip_prep = carregar_clip(device)
    face_app = carregar_insightface()

    # Testa Ollama
    try:
        ollama.list()
        log.info("Ollama ok.")
    except Exception as e:
        log.error(f"Ollama não está rodando. Abra outro terminal e digite: ollama serve")
        log.error(f"Detalhe: {e}")
        sys.exit(1)

    log.info(f"Listando fotos em {config.PASTA_FOTOS_ORIGEM}...")
    todas = list(listar_fotos(config.PASTA_FOTOS_ORIGEM))
    log.info(f"Encontradas {len(todas):,} fotos no total.")

    # Filtra as já processadas
    pendentes = [p for p in todas if not ja_processada(conn, p)]
    log.info(f"Pendentes: {len(pendentes):,} (já processadas: {len(todas) - len(pendentes):,})")

    if not pendentes:
        log.info("Nada a fazer. Tudo já analisado!")
        return

    contador = 0
    erros = 0
    for caminho in tqdm(pendentes, desc="Analisando", unit="foto"):
        try:
            # Carrega
            try:
                img = Image.open(caminho)
                img.load()
            except Exception as e:
                _registrar_erro(conn, caminho, f"abrir: {e}")
                erros += 1
                continue

            # Converte pra RGB se necessário
            if img.mode != 'RGB':
                img_rgb = img.convert('RGB')
            else:
                img_rgb = img

            largura, altura = img_rgb.size
            tamanho = os.path.getsize(caminho)

            # EXIF
            exif = extrair_exif(caminho)

            # Embedding CLIP
            emb_clip = embedding_clip(img_rgb, clip_model, clip_prep, device)

            # Rostos
            img_np = np.array(img_rgb)
            rostos = detectar_rostos(img_np, face_app)

            # Thumb + descrição
            md5 = hash_arquivo(caminho)
            thumb_path = os.path.join(config.CAMINHO_THUMBS, f"{md5}.jpg")
            if not os.path.exists(thumb_path):
                gerar_thumb(img_rgb, thumb_path)
            descricao = descrever_foto(thumb_path)

            # Salva no banco
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO fotos
                (caminho, hash_md5, largura, altura, tamanho_bytes,
                 data_exif, camera, gps_lat, gps_lon,
                 embedding_clip, descricao, num_rostos, processada_em)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                caminho, md5, largura, altura, tamanho,
                exif['data_exif'], exif['camera'], exif['gps_lat'], exif['gps_lon'],
                emb_clip.tobytes(), descricao, len(rostos),
                datetime.now().isoformat()
            ))
            foto_id = c.lastrowid

            for r in rostos:
                c.execute("""
                    INSERT INTO rostos (foto_id, embedding, bbox, qualidade)
                    VALUES (?, ?, ?, ?)
                """, (foto_id, r['embedding'].tobytes(), json.dumps(r['bbox']), r['qualidade']))

            contador += 1
            if contador % config.SALVAR_A_CADA == 0:
                conn.commit()

        except KeyboardInterrupt:
            log.info("Interrompido pelo usuário. Salvando progresso...")
            conn.commit()
            sys.exit(0)
        except Exception as e:
            log.exception(f"Erro em {caminho}")
            _registrar_erro(conn, caminho, str(e))
            erros += 1

    conn.commit()
    log.info(f"FIM. Processadas: {contador:,} | Erros: {erros:,}")
    log.info("Próximo passo: python 3_descobrir_grupos.py")


def _registrar_erro(conn, caminho, msg):
    try:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO fotos (caminho, erro, processada_em)
            VALUES (?, ?, ?)
        """, (caminho, msg[:500], datetime.now().isoformat()))
        conn.commit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
