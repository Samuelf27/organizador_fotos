"""
3_descobrir_grupos.py — Agrupa fotos similares e abre interface no navegador
para você revisar e nomear cada grupo.

Faz dois clusterings independentes:
1. Grupos de CENA (baseado em CLIP) — "reuniões", "produtos", "viagens", etc.
2. Grupos de PESSOA (baseado em rostos) — uma pessoa por grupo.

Resultado vai pra tabela `grupos` no banco. Os nomes que você der serão usados
pelo script 4_organizar.py pra criar as pastas.
"""

import os
import sys
import sqlite3
import json
import webbrowser
import threading
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs, unquote
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
import hdbscan

import config


# ============================================================
# Carregar dados do banco
# ============================================================
def carregar_embeddings_cena(conn):
    """Retorna (ids, embeddings) das fotos processadas."""
    c = conn.cursor()
    c.execute("""
        SELECT id, embedding_clip FROM fotos
        WHERE embedding_clip IS NOT NULL AND erro IS NULL
    """)
    ids, embs = [], []
    for foto_id, blob in c.fetchall():
        ids.append(foto_id)
        embs.append(np.frombuffer(blob, dtype=np.float32))
    return np.array(ids), np.stack(embs) if embs else np.empty((0, 512))


def carregar_embeddings_rosto(conn):
    """Retorna (ids_rosto, foto_ids, embeddings)."""
    c = conn.cursor()
    c.execute("""
        SELECT id, foto_id, embedding FROM rostos
        WHERE qualidade > 0.6
    """)
    ids, foto_ids, embs = [], [], []
    for rid, fid, blob in c.fetchall():
        ids.append(rid)
        foto_ids.append(fid)
        embs.append(np.frombuffer(blob, dtype=np.float32))
    return (np.array(ids), np.array(foto_ids),
            np.stack(embs) if embs else np.empty((0, 512)))


# ============================================================
# Clustering
# ============================================================
def cluster_cenas(embeddings):
    """
    HDBSCAN com cosine distance. Descobre o número de grupos sozinho.
    Retorna array de labels (-1 = outlier).
    """
    print(f"Agrupando {len(embeddings):,} fotos por cena (HDBSCAN)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.MIN_FOTOS_POR_GRUPO,
        min_samples=10,
        metric='euclidean',  # embeddings já estão normalizados, euclidean ~ cosine
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(embeddings.astype(np.float64))
    n = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  → {n} grupos encontrados ({(labels == -1).sum()} fotos sem grupo)")
    return labels


def cluster_pessoas(embeddings):
    """Mesma ideia, mas pra rostos. Threshold mais apertado."""
    print(f"Agrupando {len(embeddings):,} rostos por pessoa...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.MIN_FOTOS_POR_PESSOA,
        min_samples=3,
        metric='euclidean',
        cluster_selection_epsilon=0.5,
    )
    labels = clusterer.fit_predict(embeddings.astype(np.float64))
    n = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  → {n} pessoas identificadas")
    return labels


# ============================================================
# Salvar grupos no banco
# ============================================================
def salvar_grupos_cena(conn, foto_ids, labels):
    c = conn.cursor()
    for fid, lab in zip(foto_ids, labels):
        c.execute("UPDATE fotos SET grupo_cena = ? WHERE id = ?",
                  (int(lab) if lab >= 0 else None, int(fid)))

    # Registrar grupos
    for lab in set(labels):
        if lab < 0:
            continue
        total = int((labels == lab).sum())
        c.execute("""
            INSERT OR REPLACE INTO grupos (id, tipo, nome, total_fotos)
            VALUES (?, 'cena', ?, ?)
        """, (int(lab), f"cena_{lab}", total))
    conn.commit()


def salvar_grupos_pessoa(conn, rosto_ids, labels):
    c = conn.cursor()
    for rid, lab in zip(rosto_ids, labels):
        c.execute("UPDATE rostos SET grupo_pessoa = ? WHERE id = ?",
                  (int(lab) if lab >= 0 else None, int(rid)))
    conn.commit()


# ============================================================
# Interface HTML simples — servidor local
# ============================================================
def amostras_de_grupos(conn, n_por_grupo=12):
    """Retorna lista de grupos com amostras pra revisão."""
    c = conn.cursor()
    c.execute("SELECT id, total_fotos FROM grupos WHERE tipo='cena' ORDER BY total_fotos DESC")
    grupos = []
    for gid, total in c.fetchall():
        c.execute("""
            SELECT id, hash_md5, descricao FROM fotos
            WHERE grupo_cena = ? AND hash_md5 IS NOT NULL
            ORDER BY RANDOM() LIMIT ?
        """, (gid, n_por_grupo))
        amostras = [{'id': r[0], 'thumb': r[1], 'descricao': r[2] or ''} for r in c.fetchall()]

        # Palavras-chave mais comuns nas descrições
        c.execute("SELECT descricao FROM fotos WHERE grupo_cena = ? AND descricao IS NOT NULL", (gid,))
        palavras = Counter()
        for (desc,) in c.fetchall():
            for p in desc.lower().split():
                p = p.strip('.,;:!?()[]"\'')
                if len(p) > 3:
                    palavras[p] += 1
        top_palavras = [w for w, _ in palavras.most_common(8)]

        grupos.append({
            'id': gid,
            'total': total,
            'amostras': amostras,
            'palavras': top_palavras,
        })
    return grupos


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Nomear grupos</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; background: #f5f5f5; }
h1 { color: #333; }
.grupo { background: white; border-radius: 8px; padding: 1em; margin-bottom: 1em; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.grupo h2 { margin: 0 0 .5em; font-size: 1.1em; color: #555; }
.tags { color: #888; font-size: .9em; margin-bottom: .5em; }
.thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: .5em; margin-bottom: .8em; }
.thumbs img { width: 100%; height: 100px; object-fit: cover; border-radius: 4px; }
input[type=text] { width: 60%; padding: .5em; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; }
button { padding: .5em 1em; background: #2563eb; color: white; border: 0; border-radius: 4px; cursor: pointer; }
button:hover { background: #1d4ed8; }
.descartar { background: #dc2626; }
.descartar:hover { background: #b91c1c; }
.status { padding: .3em .6em; border-radius: 4px; font-size: .85em; }
.salvo { background: #dcfce7; color: #166534; }
.pendente { background: #fef3c7; color: #92400e; }
.barra-topo { position: sticky; top: 0; background: #f5f5f5; padding: 1em 0; z-index: 100; border-bottom: 1px solid #ddd; }
</style>
</head>
<body>
<div class="barra-topo">
<h1>Nomeie os grupos de cena</h1>
<p>Veja as amostras de cada grupo e dê um nome (ex: <code>reunioes</code>, <code>eventos</code>, <code>produtos</code>, <code>retratos</code>). 
Use <code>descartar</code> pra grupos que você não quer organizar (vão pra pasta "nao_classificadas").
Os nomes serão usados como nomes de pasta — evite acentos e espaços.</p>
<p><strong>Progresso: <span id="progresso">0</span> / __TOTAL__ grupos nomeados</strong></p>
</div>

__GRUPOS__

<script>
function nomear(gid, valor) {
    const status = document.getElementById('status-' + gid);
    status.textContent = 'salvando...';
    fetch('/nomear?id=' + gid + '&nome=' + encodeURIComponent(valor))
      .then(r => r.text())
      .then(t => {
          status.textContent = 'salvo: ' + valor;
          status.className = 'status salvo';
          atualizarProgresso();
      });
}
function descartar(gid) {
    nomear(gid, 'descartar');
    document.getElementById('input-' + gid).value = 'descartar';
}
function atualizarProgresso() {
    const salvos = document.querySelectorAll('.salvo').length;
    document.getElementById('progresso').textContent = salvos;
}
</script>
</body>
</html>
"""

HTML_GRUPO = """
<div class="grupo">
  <h2>Grupo #{id} — {total} fotos <span id="status-{id}" class="status pendente">pendente</span></h2>
  <div class="tags">Palavras frequentes: {palavras}</div>
  <div class="thumbs">{imgs}</div>
  <input id="input-{id}" type="text" placeholder="nome do grupo (ex: reunioes)" 
         onkeydown="if(event.key==='Enter'){{nomear({id}, this.value)}}">
  <button onclick="nomear({id}, document.getElementById('input-{id}').value)">Salvar</button>
  <button class="descartar" onclick="descartar({id})">Descartar</button>
</div>
"""


def montar_html(grupos):
    blocos = []
    for g in grupos:
        imgs = ''.join(
            f'<img src="/thumb/{a["thumb"]}.jpg" title="{a["descricao"][:120]}">'
            for a in g['amostras']
        )
        blocos.append(HTML_GRUPO.format(
            id=g['id'], total=g['total'],
            palavras=', '.join(g['palavras']) or '(sem descrições)',
            imgs=imgs
        ))
    return HTML_TEMPLATE.replace('__TOTAL__', str(len(grupos))).replace('__GRUPOS__', '\n'.join(blocos))


class Handler(http.server.BaseHTTPRequestHandler):
    conn = None
    html_cache = ''

    def log_message(self, *a, **k): pass  # silencia

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == '/':
            self._send(200, 'text/html; charset=utf-8', self.html_cache.encode('utf-8'))
        elif p.path.startswith('/thumb/'):
            nome = os.path.basename(p.path)
            caminho = os.path.join(config.CAMINHO_THUMBS, nome)
            if os.path.exists(caminho):
                with open(caminho, 'rb') as f:
                    self._send(200, 'image/jpeg', f.read())
            else:
                self._send(404, 'text/plain', b'nao encontrado')
        elif p.path == '/nomear':
            q = parse_qs(p.query)
            gid = int(q.get('id', [0])[0])
            nome = unquote(q.get('nome', [''])[0]).strip().lower().replace(' ', '_')
            if not nome:
                self._send(400, 'text/plain', b'nome vazio')
                return
            c = self.conn.cursor()
            c.execute("UPDATE grupos SET nome = ? WHERE id = ? AND tipo = 'cena'", (nome, gid))
            self.conn.commit()
            self._send(200, 'text/plain', b'ok')
        else:
            self._send(404, 'text/plain', b'404')

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def servir_interface(conn, porta=8765):
    grupos = amostras_de_grupos(conn)
    Handler.conn = conn
    Handler.html_cache = montar_html(grupos)
    with socketserver.TCPServer(("", porta), Handler) as httpd:
        url = f"http://localhost:{porta}/"
        print(f"\nAbra no navegador: {url}")
        print("Pressione Ctrl+C aqui no terminal quando terminar de nomear.\n")
        threading.Timer(1, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nEncerrando servidor.")


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.exists(config.CAMINHO_BANCO):
        print(f"Banco não encontrado: {config.CAMINHO_BANCO}")
        print("Rode antes: python 2_analisar.py")
        sys.exit(1)

    conn = sqlite3.connect(config.CAMINHO_BANCO)

    # Verifica se já há clustering feito
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM grupos WHERE tipo='cena'")
    ja_tem = c.fetchone()[0]

    if ja_tem == 0:
        print("=== Rodando clustering pela primeira vez ===")
        ids, embs = carregar_embeddings_cena(conn)
        if len(ids) == 0:
            print("Sem fotos no banco. Rode 2_analisar.py primeiro.")
            sys.exit(1)
        labels = cluster_cenas(embs)
        salvar_grupos_cena(conn, ids, labels)

        rids, _, rembs = carregar_embeddings_rosto(conn)
        if len(rids) > 0:
            rlabels = cluster_pessoas(rembs)
            salvar_grupos_pessoa(conn, rids, rlabels)
    else:
        print(f"Clustering já feito ({ja_tem} grupos). Abrindo interface de nomeação...")

    servir_interface(conn)


if __name__ == "__main__":
    main()
