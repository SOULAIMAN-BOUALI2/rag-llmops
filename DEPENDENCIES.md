# Rôle des dépendances

Cette commande installe les bibliothèques listées ci-dessous :

```bash
pip install rank_bm25 sentence-transformers pydantic pyyaml streamlit plotly pandas
```

| Bibliothèque | Rôle |
|---|---|
| `rank_bm25` | Recherche des passages pertinents par mots-clés avec l’algorithme BM25. |
| `sentence-transformers` | Transforme les textes et les questions en vecteurs (embeddings) pour la recherche sémantique. |
| `pydantic` | Valide et structure les données, par exemple les paramètres et les réponses de l’application. |
| `pyyaml` | Lit et écrit des fichiers de configuration au format YAML. |
| `streamlit` | Crée une interface web interactive pour utiliser le système RAG. |
| `plotly` | Produit des graphiques interactifs, par exemple pour visualiser des résultats ou des métriques. |
| `pandas` | Organise et manipule des données tabulaires, notamment pour l’analyse et l’affichage. |

Dans un système RAG, `rank_bm25` peut compléter la recherche sémantique de `sentence-transformers` : BM25 repère les correspondances lexicales, tandis que les embeddings rapprochent les textes par leur sens.

Pour installer toutes les dépendances utilisées par le projet (API, Ollama, Phoenix, évaluation et tableau de bord), utilise plutôt :

```bash
pip install -r requirements.txt
```
