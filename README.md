# RAG LLMOps

Projet RAG local : il indexe un PDF, retrouve des passages pertinents avec une recherche vectorielle et BM25, puis génère une réponse avec Ollama. L'API expose le pipeline, le tableau de bord présente les évaluations et Arize Phoenix affiche les traces.

## Prérequis

- Python 3.10 ou plus récent
- [Ollama](https://ollama.com/) installé
- Un PDF contenant du texte à placer dans `data/` (l'OCR des PDF scannés n'est pas configuré)

## Installation

Depuis la racine du projet :

```bash
cd rag-llmops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sous Windows PowerShell, active l'environnement avec :

```powershell
.venv\Scripts\Activate.ps1
```

Place un PDF dans `data/`. L'application utilise le premier fichier trouvé dans ce dossier. Pour éviter toute ambiguïté, garde un seul PDF à cet endroit.

## Démarrage

Laisse chaque service tourner dans son propre terminal. Exécute les commandes depuis la racine du projet.

### 1. Démarrer Ollama et télécharger les modèles

Si Ollama ne tourne pas déjà en arrière-plan, démarre le service :

```bash
ollama serve
```

Dans un autre terminal :

```bash
ollama pull qwen2.5-coder:7b
```

Ce modèle est utilisé pour les réponses et l'évaluation. Le nom du modèle de réponse peut être modifié dans `app.py`.

### 2. Démarrer Phoenix pour les traces (facultatif)

```bash
phoenix serve
```

Phoenix est alors accessible sur [http://localhost:6006](http://localhost:6006). Le pipeline envoie ses traces à cette adresse. Si Phoenix n'est pas lancé, l'API peut fonctionner, mais les traces ne seront pas visibles dans son interface.

### 3. Démarrer l'API RAG

```bash
uvicorn app:app --host 0.0.0.0 --port 8006 --reload
```

- API : [http://localhost:8006](http://localhost:8006)
- Documentation interactive : [http://localhost:8006/docs](http://localhost:8006/docs)
- Route de question : `POST /ask`

Exemple de requête :

```bash
curl -X POST http://localhost:8006/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelles sont les compétences décrites dans le document ?"}'
```

La réponse contient le statut, le texte généré, les pages citées, les passages utilisés et des métriques de latence.

### 4. Lancer une évaluation

Avec l'API toujours active, ouvre un autre terminal et exécute :

```bash
python eval_pipeline.py
```

Le script envoie son jeu de questions à l'API, utilise Ollama comme évaluateur, puis ajoute les résultats à `eval_history.jsonl`.

### 5. Ouvrir le tableau de bord

```bash
streamlit run dashboard_llmops.py
```

Streamlit affiche l'adresse locale dans le terminal, généralement [http://localhost:8501](http://localhost:8501). Le tableau de bord contient :

- **Benchmarks & Drift** : scores de fidélité et de pertinence, latence et détails des évaluations enregistrées dans `eval_history.jsonl`. Lance d'abord `python eval_pipeline.py` pour remplir cet historique.
- **Live Playground & Traces** : envoi d'une question à l'API et lien vers l'interface Phoenix.

## Fichiers principaux

| Fichier ou dossier | Rôle |
|---|---|
| `app.py` | API FastAPI et route `POST /ask`. |
| `rag_engine.py` | Extraction du PDF, découpage, indexation hybride, reranking, guardrails, cache et génération de réponse. |
| `llmops_layer.py` | Détection de motifs de prompt injection, masquage de certaines données personnelles et cache sémantique. |
| `prompts.yaml` | Versions et modèles de prompts utilisés par le pipeline. |
| `eval_pipeline.py` | Évaluation du pipeline et enregistrement des résultats. |
| `dashboard_llmops.py` | Interface Streamlit pour les benchmarks et le test en direct. |
| `eval_history.jsonl` | Historique des exécutions d'évaluation. |
| `data/` | Documents PDF indexés au démarrage de l'API. |
| `requirements.txt` | Dépendances Python du projet. |
| `DEPENDENCIES.md` | Explication du rôle des bibliothèques. |

## Configuration et dépannage

- Lance les commandes depuis la racine du projet : les chemins `data/`, `prompts.yaml` et `eval_history.jsonl` sont relatifs au dossier courant.
- Le nom du modèle Ollama de l'API est défini dans `app.py`; celui de l'évaluateur est défini dans `eval_pipeline.py`. Télécharge chaque modèle configuré avec `ollama pull <nom-du-modèle>`.
- La version active du prompt est définie par `active_version` dans `prompts.yaml`.
- Si l'API indique qu'aucun PDF n'a été trouvé, vérifie que le fichier est directement dans `data/` et possède l'extension `.pdf`.
- Si le tableau de bord ne montre aucun benchmark, exécute `python eval_pipeline.py` après avoir démarré l'API.
- Si aucune trace n'apparaît dans Phoenix, vérifie que Phoenix écoute sur le port `6006`.
