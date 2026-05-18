"""
5_buscar.py — Interface web local pra buscar fotos no banco.

Rode com:  streamlit run 5_buscar.py

Funcionalidades:
- Busca por palavra-chave nas descrições
- Filtrar por grupo (nome que você deu)
- Filtrar por pessoa (cluster de rosto)
- Filtrar por intervalo de data
- Busca semântica (digite uma frase, encontra fotos parecidas via CLIP)
"""

import os
import sqlite3
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

import config

st.set_page_config(page_title="Buscador de Fotos", layout="wide")

# ============================================================
# Cache de recursos pesados
# ============================================================
@st.cache_resource
def conectar():
    return sqlite3.connect(config.CAMINHO_BANCO, check_same_thread=False)


@st.cache_resource
def carregar_clip():
    """Carrega CLIP só se o usuário fizer busca semântica."""
    import torch
    import open_clip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms(
        config.MODELO_CLIP, pretrained=config.PESOS_CLIP
    )
    tokenizer = open_clip.get_tokenizer(config.MODELO_CLIP)
    model = model.to(device).eval()
    return model, tokenizer, device


# ============================================================
# Consultas
# ============================================================
def listar_grupos(conn):
    c = conn.cursor()
    c.execute("""
        SELECT nome, total_fotos FROM grupos
        WHERE tipo='cena' AND nome NOT LIKE 'cena_%'
        ORDER BY total_fotos DESC
    """)
    return c.fetchall()


def listar_pessoas(conn):
    c = conn.cursor()
    c.execute("""
        SELECT grupo_pessoa, COUNT(DISTINCT foto_id) as n
        FROM rostos
        WHERE grupo_pessoa IS NOT NULL
        GROUP BY grupo_pessoa
        HAVING n >= ?
        ORDER BY n DESC
    """, (config.MIN_FOTOS_POR_PESSOA,))
    return c.fetchall()


def buscar(conn, texto=None, grupo=None, pessoa=None, data_de=None, data_ate=None, limite=60):
    c = conn.cursor()
    where = ["f.erro IS NULL"]
    params = []

    if texto:
        where.append("f.descricao LIKE ?")
        params.append(f"%{texto}%")
    if grupo and grupo != "(todos)":
        where.append("g.nome = ?")
        params.append(grupo)
    if pessoa is not None:
        where.append("EXISTS (SELECT 1 FROM rostos r WHERE r.foto_id = f.id AND r.grupo_pessoa = ?)")
        params.append(pessoa)
    if data_de:
        where.append("substr(f.data_exif,1,10) >= ?")
        params.append(data_de.strftime("%Y:%m:%d"))
    if data_ate:
        where.append("substr(f.data_exif,1,10) <= ?")
        params.append(data_ate.strftime("%Y:%m:%d"))

    sql = f"""
        SELECT f.id, f.caminho, f.hash_md5, f.descricao, f.data_exif, g.nome
        FROM fotos f
        LEFT JOIN grupos g ON g.id = f.grupo_cena AND g.tipo='cena'
        WHERE {' AND '.join(where)}
        ORDER BY f.data_exif DESC
        LIMIT ?
    """
    params.append(limite)
    c.execute(sql, params)
    return c.fetchall()


def busca_semantica(conn, frase, limite=60):
    """Codifica a frase com CLIP, faz dot-product com todos os embeddings."""
    import torch
    model, tokenizer, device = carregar_clip()
    with torch.no_grad():
        toks = tokenizer([frase]).to(device)
        q = model.encode_text(toks)
        q = q / q.norm(dim=-1, keepdim=True)
        q = q.cpu().numpy().astype(np.float32)[0]

    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.caminho, f.hash_md5, f.descricao, f.data_exif, g.nome, f.embedding_clip
        FROM fotos f
        LEFT JOIN grupos g ON g.id = f.grupo_cena AND g.tipo='cena'
        WHERE f.embedding_clip IS NOT NULL AND f.erro IS NULL
    """)
    resultados = []
    for row in c.fetchall():
        emb = np.frombuffer(row[6], dtype=np.float32)
        sim = float(np.dot(q, emb))
        resultados.append((sim, row[:6]))
    resultados.sort(key=lambda x: -x[0])
    return [r[1] for r in resultados[:limite]]


# ============================================================
# UI
# ============================================================
st.title("🔍 Buscador de Fotos")

conn = conectar()

modo = st.radio(
    "Modo de busca",
    ["Filtros (rápido)", "Busca semântica por frase (mais lento, carrega CLIP)"],
    horizontal=True
)

with st.sidebar:
    st.header("Filtros")
    if modo.startswith("Filtros"):
        texto = st.text_input("Palavra na descrição", "")
        grupos = listar_grupos(conn)
        opcoes_grupo = ["(todos)"] + [g[0] for g in grupos]
        grupo_sel = st.selectbox("Grupo", opcoes_grupo)

        pessoas = listar_pessoas(conn)
        opcoes_pessoa = ["(todas)"] + [f"Pessoa #{p[0]} ({p[1]} fotos)" for p in pessoas]
        pessoa_idx = st.selectbox("Pessoa", range(len(opcoes_pessoa)), format_func=lambda i: opcoes_pessoa[i])
        pessoa_sel = pessoas[pessoa_idx - 1][0] if pessoa_idx > 0 else None

        col1, col2 = st.columns(2)
        with col1:
            data_de = st.date_input("De", value=None)
        with col2:
            data_ate = st.date_input("Até", value=None)
    else:
        frase = st.text_input("Descreva o que procura", placeholder="ex: reunião com café em sala de vidro")

    limite = st.slider("Máx. resultados", 12, 200, 60, 12)

if st.button("Buscar", type="primary"):
    if modo.startswith("Filtros"):
        resultados = buscar(conn, texto or None, grupo_sel, pessoa_sel, data_de, data_ate, limite)
    else:
        if not frase:
            st.warning("Digite uma frase pra busca semântica.")
            st.stop()
        with st.spinner("Codificando frase e comparando com todas as fotos..."):
            resultados = busca_semantica(conn, frase, limite)

    st.write(f"**{len(resultados)} resultado(s)**")

    cols = st.columns(4)
    for i, r in enumerate(resultados):
        foto_id, caminho, md5, descricao, data, grupo = r
        col = cols[i % 4]
        thumb = os.path.join(config.CAMINHO_THUMBS, f"{md5}.jpg")
        if os.path.exists(thumb):
            col.image(thumb, use_container_width=True)
        else:
            col.write("(thumb não encontrada)")
        col.caption(f"📁 {grupo or '?'} | 📅 {(data or '?')[:10]}")
        if descricao:
            col.caption(descricao[:150])
        col.caption(f"`{caminho}`")
else:
    st.info("Use os filtros à esquerda e clique em Buscar.")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM fotos WHERE erro IS NULL")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM rostos")
    rostos = c.fetchone()[0]
    st.metric("Fotos no banco", f"{total:,}")
    st.metric("Rostos detectados", f"{rostos:,}")
