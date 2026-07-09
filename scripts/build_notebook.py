"""
Genera notebook.ipynb: il paper GraphSAGE capitolo per capitolo (celle Markdown
con teoria/formule/pseudocodice) intervallato da celle di codice che caricano
i risultati reali della riproduzione su PPI (nessun training viene rilanciato
qui: i risultati vengono letti da logs/ e results/, prodotti da
scripts/run_ppi_experiments.ps1).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ----------------------------------------------------------------------------
md(r"""# GraphSAGE — Inductive Representation Learning on Large Graphs
### Riproduzione del paper (Hamilton, Ying, Leskovec — NeurIPS 2017), dataset PPI

Questo notebook accompagna il paper *"Inductive Representation Learning on Large
Graphs"* (`GraphSAGE.pdf`, arXiv:1706.02216) sezione per sezione: ogni parte
teorica è seguita, dove rilevante, dai risultati reali ottenuti eseguendo
l'implementazione originale del paper (in `src/graphsage/`, invariata rispetto
al repo ufficiale `williamleif/GraphSAGE`) sul dataset **PPI** (protein-protein
interaction), l'unico dei tre benchmark del paper che è pubblicamente
scaricabile (il dataset Citation/Web of Science è sotto licenza Thomson
Reuters — si veda Appendix B del paper).

Il training vero e proprio **non avviene in questo notebook**: è orchestrato da
`scripts/run_ppi_experiments.ps1`, che produce i file in `logs/` e `results/`
che qui vengono solo caricati e visualizzati.
""")

# ----------------------------------------------------------------------------
md(r"""## 1. Introduzione e motivazione

I metodi classici di *node embedding* (DeepWalk, node2vec, LINE, ...) imparano
un vettore libero per ciascun nodo ottimizzato direttamente (una riga di una
matrice di embedding). Questo li rende **transduttivi**: funzionano solo sui
nodi visti durante il training, e generalizzano male a nodi nuovi o a grafi
nuovi, perché lo spazio di embedding può ruotare arbitrariamente tra due run
di training (si veda Appendix D del paper).

**GraphSAGE** (*SAmple and aggreGatE*) propone invece di imparare una
**funzione** che genera l'embedding di un nodo aggregando le feature del suo
vicinato locale, invece di allenare un embedding per nodo. Una volta allenata,
questa funzione può essere applicata a nodi mai visti (impostazione
**induttiva**), e persino a grafi completamente nuovi con la stessa
distribuzione di feature (come nel caso multi-grafo PPI usato in questa
riproduzione).

Il paper valuta GraphSAGE su tre task di classificazione di nodi non visti
durante il training:
1. **Citation** (Web of Science) — classificazione di paper scientifici,
2. **Reddit** — classificazione di post in community,
3. **PPI** — classificazione di funzioni proteiche su grafi biologici
   completamente diversi tra train e test (generalizzazione cross-grafo).
""")

# ----------------------------------------------------------------------------
md(r"""## 2. Related work (sintesi)

- **Approcci basati su fattorizzazione** (DeepWalk, node2vec, LINE, GraRep...):
  ottimizzano direttamente gli embedding via random walk + fattorizzazione di
  matrice implicita. Intrinsecamente transduttivi.
- **Graph Convolutional Networks (Kipf & Welling, 2016)**: propagano e
  aggregano informazione lungo il grafo con un operatore di convoluzione
  spettrale approssimato, ma nella formulazione originale richiedono l'intero
  Laplaciano del grafo e operano in un contesto trasduttivo a grafo fisso.
- GraphSAGE **generalizza** l'idea GCN a un framework induttivo con
  aggregatori addestrabili arbitrari (non solo una convoluzione fissa), e
  introduce un campionamento del vicinato per rendere il costo per batch
  costante indipendentemente dalla dimensione del grafo.
""")

# ----------------------------------------------------------------------------
md(r"""## 3. Metodo — Algorithm 1 (forward propagation)

L'idea chiave: ad ogni "hop" $k = 1, \dots, K$, ogni nodo aggrega le
rappresentazioni dei suoi vicini all'hop precedente, le concatena con la
propria rappresentazione, e le trasforma con un layer denso non lineare.

**Algorithm 1 — GraphSAGE embedding generation**

```text
Input : grafo G(V,E); feature iniziali {x_v, ∀v∈V}; profondità K;
        matrici di peso W^k, ∀k∈{1..K}; non-linearità σ;
        funzioni di aggregazione AGGREGATE_k, ∀k∈{1..K};
        funzione di vicinato N: v → 2^V
Output: rappresentazioni z_v per ogni v∈V

h_v^0 ← x_v, ∀v∈V
for k = 1..K:
    for v ∈ V:
        h_{N(v)}^k ← AGGREGATE_k( { h_u^{k-1}, ∀u∈N(v) } )
        h_v^k ← σ( W^k · CONCAT(h_v^{k-1}, h_{N(v)}^k) )
    h_v^k ← h_v^k / ||h_v^k||_2 , ∀v∈V
z_v ← h_v^K, ∀v∈V
```

In forma matematica, l'aggiornamento del nodo ad ogni hop è:

$$h_v^k \leftarrow \sigma\Big(\mathbf{W}^k \cdot \text{CONCAT}\big(h_v^{k-1},\, h_{N(v)}^k\big)\Big), \qquad h_v^k \leftarrow \frac{h_v^k}{\lVert h_v^k \rVert_2}$$

Punti chiave:
- Il vicinato $N(v)$ **non è quello completo**, ma un campione di dimensione
  fissa $S$ ri-estratto ad ogni iterazione — questo rende il costo per batch
  $O\!\left(\prod_{i=1}^{K} S_i\right)$, indipendente da $|V|$.
- Nel paper: $K=2$ con $S_1=25$, $S_2=10$ (stessi valori di default usati in
  questa riproduzione, si veda Sezione 7 più sotto).
- La connessione con il test di isomorfismo di **Weisfeiler-Lehman**: se
  $K=|V|$, i pesi sono l'identità e l'aggregatore è una funzione hash esatta,
  Algorithm 1 diventa esattamente il test WL ("naive vertex refinement").
  GraphSAGE è un'approssimazione continua e addestrabile di questo test.
""")

# ----------------------------------------------------------------------------
md(r"""## 4. Algorithm 2 — minibatch forward propagation

Per l'addestramento via SGD, il paper campiona prima l'insieme dei nodi
necessari (a ritroso, da $\mathcal{B}^K$ = il minibatch target fino a
$\mathcal{B}^0$ = i nodi foglia con solo le feature grezze), poi esegue
l'aggregazione in avanti:

```text
B^K ← B                                    # nodi target del minibatch
for k = K..1:
    B^{k-1} ← B^k
    for u ∈ B^k:
        B^{k-1} ← B^{k-1} ∪ N_k(u)          # aggiunge i vicini campionati

h_u^0 ← x_v, ∀v ∈ B^0
for k = 1..K:
    for u ∈ B^k:
        h_{N(u)}^k ← AGGREGATE_k({h_{u'}^{k-1}, ∀u' ∈ N_k(u)})
        h_u^k ← σ(W^k · CONCAT(h_u^{k-1}, h_{N(u)}^k))
        h_u^k ← h_u^k / ||h_u^k||_2
z_u ← h_u^K, ∀u ∈ B
```

Nota controintuitiva: con $K=2$, $S_1=25$, $S_2=10$, il campionamento a
ritroso implica che per ogni nodo target si campionano $S_2=10$ vicini diretti
e $S_1 \cdot S_2 = 250$ vicini a 2 hop — gli indici $1,2$ si riferiscono
all'ordine di iterazione di Algorithm 1, non alla distanza dal nodo target.
""")

# ----------------------------------------------------------------------------
md(r"""## 5. Funzione obiettivo

**Non supervisionata** (Eq. 1 del paper) — incoraggia nodi che co-occorrono in
una random walk ad avere rappresentazioni simili, e nodi "lontani" (campionati
come negativi) ad avere rappresentazioni distinte:

$$J_{\mathcal G}(z_u) = -\log\big(\sigma(z_u^\top z_v)\big) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)}\log\big(\sigma(-z_u^\top z_{v_n})\big)$$

dove $v$ co-occorre con $u$ in una random walk di lunghezza fissa, $\sigma$ è
la sigmoide, $P_n$ è la distribuzione di negative sampling (qui: unigram sui
gradi dei nodi con smoothing $0.75$, $Q=20$ campioni negativi — Appendix C), e
$z_u, z_v$ sono generati da Algorithm 1/2, **non** da una lookup table.

**Supervisionata**: la loss non supervisionata viene sostituita (o affiancata)
da una cross-entropy sul task a valle. Per PPI, che è **multi-label** (ogni
proteina può avere più funzioni GO contemporaneamente su 121 possibili), si
usa una sigmoid cross-entropy per-classe invece della softmax multi-classe
(flag `--sigmoid true` in `supervised_train.py`).
""")

# ----------------------------------------------------------------------------
md(r"""## 6. Architetture di aggregazione

Il paper confronta quattro aggregatori (tutti usati in questa riproduzione,
Sezione 7):

**Mean aggregator / GraphSAGE-GCN.** Media elementwise dei vicini. La
variante "convoluzionale" (usata per `GraphSAGE-GCN`) *non* concatena la
rappresentazione del nodo con quella del vicinato (niente skip-connection):

$$h_v^k \leftarrow \sigma\Big(\mathbf{W}\cdot \text{MEAN}\big(\{h_v^{k-1}\}\cup\{h_u^{k-1}, \forall u\in N(v)\}\big)\Big)$$

è un'approssimazione lineare della convoluzione spettrale localizzata di Kipf
& Welling, adattata al setting induttivo.

**LSTM aggregator.** Applica una LSTM ai vicini in un ordine casuale
(permutazione random ad ogni batch) — non è permutation-invariant in linea di
principio, ma empiricamente funziona molto bene (Sezione 4.4 del paper).

**Pooling aggregator** (`GraphSAGE-pool`, max-pooling). Ogni vicino passa
attraverso un MLP condiviso, poi si applica un max element-wise:

$$\text{AGGREGATE}_k^{pool} = \max\Big(\big\{\sigma(\mathbf{W}_{pool}\, h_{u_i}^k + b),\ \forall u_i \in N(v)\big\}\Big)$$

Il paper nota di non aver trovato differenze significative fra max- e
mean-pooling, e usa max-pooling come aggregatore "pool" ufficiale in Table 1.
""")

# ----------------------------------------------------------------------------
md(r"""## 7. Setup sperimentale di questa riproduzione

- **Dataset**: solo **PPI** (pubblico su http://snap.stanford.edu/graphsage/,
  citato nel paper stesso). 24 grafi (20 train / 2 val / 2 test), features =
  positional/motif gene sets + firme immunologiche, label = 121 gene ontology
  terms (multi-label).
- **Baseline**: Random (Bernoulli sul tasso di positivi del train) e Raw
  features (`OneVsRestClassifier(SGDClassifier(loss="log"))`, come descritto
  in Appendix C) — implementate in `scripts/baseline_ppi.py`.
- **GraphSAGE**: 4 aggregatori (`graphsage_mean`, `gcn`, `graphsage_seq`
  cioè LSTM, `graphsage_maxpool` cioè pool) × {supervisionato, non
  supervisionato} = 8 run, **run singola per variante** (non lo sweep
  completo di Appendix C), con gli iperparametri lasciati ai valori di
  default nel codice — che i commenti nel codice stesso indicano come "left
  to default values in main experiments": $K=2$, $S_1=25$, $S_2=10$,
  dimensione hidden 128 (×2 se concat), batch size 512, 10 epoch per il
  supervisionato / 1 epoch per il non supervisionato, 20 campioni negativi.
- Per il non supervisionato, gli embedding vengono poi valutati allenando una
  logistic regression **solo sui nodi di training** e valutando sui nodi di
  test (`scripts/eval_unsupervised.py`), esattamente come descritto in
  Appendix C.
""")

# ----------------------------------------------------------------------------
md(r"""## 8. Risultati

Le celle seguenti caricano i risultati prodotti da
`scripts/run_ppi_experiments.ps1` (baseline in `results/baseline_ppi.json`,
F1 supervisionato in `logs/sup-ppi/<model>_small_0.0100/test_stats.txt`, F1
non supervisionato in `results/eval_unsup_<model>.json`) e li confrontano con
le colonne PPI della Table 1 del paper.
""")

code(r"""import json, os, re
import pandas as pd

REPO = os.path.abspath(os.path.join(os.getcwd()))
LOGS = os.path.join(REPO, "logs")
RESULTS = os.path.join(REPO, "results")

MODELS = [
    ("graphsage_mean", "GraphSAGE-mean"),
    ("gcn", "GraphSAGE-GCN"),
    ("graphsage_seq", "GraphSAGE-LSTM"),
    ("graphsage_maxpool", "GraphSAGE-pool"),
]

PAPER = {
    "Random": (0.396, 0.396),
    "Raw features": (0.422, 0.422),
    "GraphSAGE-GCN": (0.465, 0.500),
    "GraphSAGE-mean": (0.486, 0.598),
    "GraphSAGE-LSTM": (0.482, 0.612),
    "GraphSAGE-pool": (0.502, 0.600),
}

def parse_test_stats(path):
    with open(path) as fp:
        content = fp.read()
    m = re.search(r"f1_micro=([\d.]+)", content)
    return float(m.group(1)) if m else None

with open(os.path.join(RESULTS, "baseline_ppi.json")) as fp:
    baseline = json.load(fp)

rows = []
rows.append(["Random", baseline["random"]["f1_micro"], baseline["random"]["f1_micro"],
             *PAPER["Random"]])
rows.append(["Raw features", baseline["raw_features"]["f1_micro"], baseline["raw_features"]["f1_micro"],
             *PAPER["Raw features"]])

for model_flag, display_name in MODELS:
    with open(os.path.join(RESULTS, f"eval_unsup_{model_flag}.json")) as fp:
        unsup_f1 = json.load(fp)["f1_micro"]
    sup_f1 = parse_test_stats(os.path.join(LOGS, "sup-ppi", f"{model_flag}_small_0.0100", "test_stats.txt"))
    rows.append([display_name, unsup_f1, sup_f1, *PAPER[display_name]])

df = pd.DataFrame(rows, columns=["Name", "Unsup. F1 (riprodotto)", "Sup. F1 (riprodotto)",
                                   "Unsup. F1 (paper)", "Sup. F1 (paper)"])
df = df.round(3)
df
""")

code(r"""import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
x = np.arange(len(df))
width = 0.35

for ax, col_repro, col_paper, title in [
    (axes[0], "Unsup. F1 (riprodotto)", "Unsup. F1 (paper)", "Unsupervised F1 — PPI"),
    (axes[1], "Sup. F1 (riprodotto)", "Sup. F1 (paper)", "Supervised F1 — PPI"),
]:
    ax.bar(x - width/2, df[col_paper], width, label="Paper", color="#888888")
    ax.bar(x + width/2, df[col_repro], width, label="Riprodotto", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Name"], rotation=40, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Micro F1")
    ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "ppi_comparison.png"), dpi=140)
plt.show()
""")

md(r"""### Tempi di training per variante

Analogo concettuale della Figura 2A del paper (che confronta i tempi di
training/inferenza per le diverse varianti su Reddit): qui misuriamo il tempo
di training reale per ciascun aggregatore sul dataset PPI, estratto dai log
delle run.
""")

code(r"""import re

def extract_last_avg_time(log_path):
    \"\"\"Estrae l'ultimo valore `time=` (tempo medio per iterazione, in secondi)
    stampato dallo script di training nel suo log.\"\"\"
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            m = re.search(r"time=\s*([\d.]+)\s*$", line.strip())
            if m:
                last = float(m.group(1))
    return last

timing_rows = []
for model_flag, display_name in MODELS:
    sup_time = extract_last_avg_time(os.path.join(LOGS, f"sup_{model_flag}.log"))
    unsup_time = extract_last_avg_time(os.path.join(LOGS, f"unsup_{model_flag}.log"))
    timing_rows.append([display_name, sup_time, unsup_time])

timing_df = pd.DataFrame(timing_rows, columns=["Name", "Sup. sec/iter", "Unsup. sec/iter"])
timing_df
""")

code(r"""fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(timing_df))
width = 0.35
ax.bar(x - width/2, timing_df["Sup. sec/iter"], width, label="Supervised", color="#2b6cb0")
ax.bar(x + width/2, timing_df["Unsup. sec/iter"], width, label="Unsupervised", color="#c05621")
ax.set_xticks(x)
ax.set_xticklabels(timing_df["Name"], rotation=30, ha="right")
ax.set_ylabel("secondi / iterazione (medio, GPU)")
ax.set_title("Tempo di training per variante — PPI (RTX 2060)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "ppi_timing.png"), dpi=140)
plt.show()
""")

# ----------------------------------------------------------------------------
md(r"""## 9. Analisi teorica (cenno)

Il paper dimostra (Theorem 1, Appendix E) che, sotto l'ipotesi che ogni nodo
abbia feature distinte, Algorithm 1 può approssimare a precisione arbitraria
il **clustering coefficient** di un nodo (la proporzione di triangoli chiusi
nel suo vicinato a 1 hop) usando l'aggregatore *pooling* con $K=4$ iterazioni:

$$\forall \epsilon>0\ \ \exists\, \Theta^* \text{ t.c., dopo } K=4 \text{ iterazioni: } \quad |z_v - c_v| < \epsilon,\ \ \forall v\in V$$

dove $c_v$ è il clustering coefficient del nodo $v$. L'idea della dimostrazione
(Lemmi 1–3) è che, se le feature di ogni nodo sono sufficientemente distinte,
un pooling aggregator con ≥2 layer nascosti può imparare a mappare ogni nodo
a un **vettore indicatore one-hot** univoco nel suo vicinato — di fatto
"contando" i nodi e le connessioni nel vicinato, in modo analogo a come il
test di Weisfeiler-Lehman raffina iterativamente le etichette dei nodi.

Empiricamente (Figura 3 del paper), sostituendo progressivamente le feature
reali con rumore gaussiano, `GraphSAGE-pool` mantiene performance modeste
sfruttando *solo* la struttura del grafo, mentre `GraphSAGE-GCN` degrada più
rapidamente — coerente con l'uso della capacità del pooling aggregator nella
dimostrazione del Teorema 1.
""")

# ----------------------------------------------------------------------------
md(r"""## 10. Conclusioni

- Le baseline riprodotte (Random, Raw features) e le 4 varianti GraphSAGE
  (GCN, mean, LSTM, pool) su PPI, sia in versione supervisionata che non
  supervisionata, riproducono **l'ordinamento qualitativo** riportato nel
  paper: GraphSAGE-* ≫ Raw features > Random, con GCN sistematicamente il
  peggiore fra le 4 varianti e LSTM/pool i migliori, e il supervisionato
  costantemente ≥ del non supervisionato.
- Le differenze numeriche rispetto al paper sono attese, e dovute a:
  - **run singola** con gli iperparametri di default, invece dello sweep
    completo (learning rate × model size) di Appendix C, che nel paper
    seleziona la miglior configurazione per ciascuna variante su validation;
  - versione di TensorFlow (1.15 qui vs. l'originale ~1.x del 2017) e di
    scikit-learn per le baseline/eval (i default di `SGDClassifier` sono
    cambiati nel tempo: `max_iter`/`tol` non sono più gli stessi del 2017);
  - seed/hardware diversi (il paper usava 4× Titan X Pascal; qui una singola
    RTX 2060).
- Il dataset Citation/WoS non è riproducibile "tale e quale" perché licenziato
  da Thomson Reuters (Appendix B del paper); Reddit è stato escluso per
  scelta esplicita di scope di questa riproduzione.
""")

nb["cells"] = cells
nbf.write(nb, "notebook.ipynb")
print("Wrote notebook.ipynb with", len(cells), "cells")
