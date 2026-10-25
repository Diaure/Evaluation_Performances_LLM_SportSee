# Assistant RAG avec Mistral + SQL & Evaluation des performances LLM

Ce projet consiste à développer et à évaluer un assistant virtuel capable de répondre à des questions sur les statistiques NBA et les opinions exprimées par les fans dans des discussions Reddit.

L'assistant combine deux sources de connaissances complémentaires:

- **PostgreSQL / SQL**: récupération de données structurées sur les équipes, les joueurs, les matchs et leurs statistiques

- **RAG (Retrieval-Augmented Generation)**: recherche sémantique dans des documents textuels à l'aide d'**`embeddings et de FAISS`**, notamment pour exploiter des discussions Reddit.

L'objectif est double: fournir des réponses pertinentes et fiables, puis mesurer l'impact des améliorations apportées au prototype initial.

Le projet compare de ce fait deux versions du prototype:

- **Prototype initial**: version de référence, sans les contrôles et transformations ajoutés au prototype amélioré

- **Prototype amélioré**: version intégrant une validation structurée avec Pydantic, un contrôle sémantique avec Pydantic AI et des traitements spécifiques destinés à fiabiliser les réponses.

L'évaluation compare les deux versions sur un jeu de validation commun à l'aide de métriques RAGAS.

## Technologies utilisées
| Technologie	| Rôle |
|---------------|:---------------------------------|
| Python 3.10.8 | Environnement de développement |
| Mistral AI | Génération des réponses et embeddings |
| PostgreSQL | Stockage des données NBA structurées |
| SQL | Extraction et calcul des statistiques |
| RAG | Recherche d'informations dans les documents textuels |
| FAISS | Recherche vectorielle des passages pertinents |
| LangChain | Composants de traitement et de recherche documentaire |
| Pydantic | Validation des structures de données |
| Pydantic AI | Contrôle sémantique des réponses |
| Logfire| Observabilité et diagnostic des exécutions |
| RAGAS | Évaluation automatique des performances |
| Streamlit | Interface conversationnelle |


## Prérequis

- **Python 3.10.8**: version utilisée lors du développement et de l'évaluation; et recommandée afin de limiter les problèmes de compatibilité avec les dépendances
- Clé API Mistral valide (obtenue sur [console.mistral.ai](https://console.mistral.ai/))
- **PostgreSQL** accessible


## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
py -3.10 -m venv nom_environnement

# Activation de l'environnement virtuel
# Sur Windows
nom_environnement\Scripts\activate
# Sur macOS/Linux
source nom_environnement/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer clé API et connexion postgresql**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```bash
# Clé API
MISTRAL_API_KEY=votre_clé_api_mistral
```

```bash
# Connexion postresql
DB_USER = nom utilisateur
DB_PASSWORD = mot de passe
DB_HOST = adresse du serveur
DB_PORT = port (généralement 5432)
DB_NAME = nom de de la base
```

## Structure du projet

![Structure projet](https://raw.githubusercontent.com/Diaure/Evaluation_Performances_LLM_SportSee/main/inputs/Capture.png)


## Utilisation

### 1. Préparation des documents

Les documents/chunks ont déjà été préparés aussi bien pour le prototype initial que celui amélioré:
```bash
# Initial
vector_db/documents_chunks.pkl
vector_db/faiss_index.idx

# Amélioré
vector_db/documents_chunks_new.pkl
vector_db/faiss_index_new.idx
```

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS:

```bash
# Prototype initial
python indexer.py

# Protype amélioré
python indexer_new.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Création et alimentation de la base PostgreSQL

Exécutez les étapes tel que suit:
- Créer une base de données (*nba_stats*) `PostgreSQL` depuis `pgAdmin`, puis configurer les variables d'environnement nécessaires à la connexion dans le fichier `.env`
- Exécuter le script `load_excel_to_db.py`: Ce script lit les données du fichier *`inputs/regular NBA.xlsx`*, les prépare et les valide à l'aide des modèles Pydantic, puis crée les tables nécessaires et y insère les données.

    **Attention**: les tables gérées par le script sont supprimées puis recréées à chaque exécution. Les données précédemment présentes dans ces tables sont donc remplacées.

- Le module `sql_tool.py` assure l'accès aux données structurées de PostgreSQL: 

    - il transforme la question de l'utilisateur en requête SQL, 
    - contrôle la requête, 
    - l'exécute sur la base de données,
    - retourne les résultats au prototype pour la génération de la réponse.

**Architecture accès aux données**

![Accès données](https://raw.githubusercontent.com/Diaure/Evaluation_Performances_LLM_SportSee/main/inputs/Picture1.png)

### 4. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


### 5. Evaluation de l'application

```bash
# Prototype initial
python evaluation initiale.py

# Prototype amélioré
python evaluate_prototypes_pydantic.py

# Comparaison des deux prototypes
python resume_evaluation.py
```
Les résultats de l'évaluation seront stockés dans le dossier `ragas_results`. 

## Pipeline global de fonctionnement

![Pipeline global](https://raw.githubusercontent.com/Diaure/Evaluation_Performances_LLM_SportSee/main/inputs/pipeline_global.png)

**Étapes du pipeline**
1. **Routage**: le système détermine si la question nécessite une recherche SQL, une recherche documentaire RAG ou les deux
2. **Récupération**: `PostgreSQL` fournit les données statistiques; `FAISS` retrouve les passages textuels pertinents
3. **Génération**: `Mistral` formule une réponse à partir de la question et des informations disponibles
4.	**Validation**: `Pydantic` vérifie la structure des données; `Pydantic AI` contrôle la réponse selon les règles définies
5.	**Restitution**: la réponse finale est affichée dans l'interface `Streamlit`
6. **Évaluation**: `RAGAS` mesure les performances sur le jeu de validation. Les résultats des deux prototypes sont ensuite comparés.

`Pydantic Logfire` accompagne les étapes instrumentées du pipeline pour faciliter le suivi des exécutions et le diagnostic des erreurs. Il ne remplace ni la validation ni l'évaluation des réponses.


## Évaluation comparative avec RAGAS

### Jeu de validation
Le benchmark utilise un jeu de validation de 31 questions couvrant différents types de demandes:

- statistiques individuelles et collectives
- classements et comparaisons
- questions nécessitant des informations issues de documents textuels
- questions combinant SQL et RAG
- questions pour lesquelles les données peuvent être insuffisantes.

Chaque question est associée à:
- une réponse de référence (**ground_truth**)
- un type de question et son niveau de difficulté (**type, dificulty**)
- une source de donnée expectée (**source_expected**).

### Métriques utilisées
| Métrique	| Ce qu'elle mesure |
|---------------|:---------------------------------|
| **answer_relevancy**| La pertinence de la réponse par rapport à la question |
| **answer_correctness** | La conformité de la réponse à la référence, sur le fond et les faits |
| **context_precision** | La pertinence des contextes récupérés, en tenant compte de leur classement |
| **context_recall** | La couverture des informations attendues dans les contextes récupérés |
| **faithfulness** | Le degré auquel les affirmations de la réponse sont étayées par le contexte |
| **answer_similarity** | La proximité sémantique entre la réponse produite et la référence |
| **context_entity_recall** | La couverture des entités présentes dans la référence par les contextes |

Les métriques sont complémentaires. Une réponse peut être pertinente sans être correcte, ou fidèle au contexte sans couvrir toutes les informations nécessaires.

Un `score de 0,00` signifie que la métrique n'a attribué aucun point selon ses critères. 

Un `score absent (NaN)` signifie qu'aucun score exploitable n'a été produit. 

### Résultats globaux et interprétation
Les résultats ci-dessous proviennent du tableau de synthèse comparative fourni dans le projet. Ils sont présentés sur une échelle allant de 0 à 1.

| Métriques	| Prototype initial | Prototype amélioré | Interprétation |
:---------------|:---------------|:------------------|:------------|
| **answer_relevancy**| 0,446 | 0.723 | Les réponses sont davantage en adéquation avec les questions |
| **answer_correctness** | 0,255 | 0,729 | La conformité des réponses à la référence progresse fortement |
| **context_precision** | 0,140 | 0,902 | Les contextes récupérés sont beaucoup plus pertinents selon l'évaluation |
| **context_recall** | 0,583 | 0,936 | Les informations attendues sont davantage couvertes dans les contextes évalués |
| **faithfulness** | 0,535 | 0,994 | Les réponses sont mieux étayées par les contextes fournis |
| **answer_similarity** | 0,839 | 0,948 | Les réponses se rapprochent sémantiquement des références |
| **context_entity_recall** | 0,362 | 0,501 | La couverture des entités attendues s'améliore, mais reste perfectible |

### Principales évolutions constatées
**Meilleure pertinence des réponses**

La progression de `answer_relevancy` indique que le prototype amélioré répond plus directement aux demandes. Le gain est particulièrement important sur les questions auxquelles le prototype initial apportait une réponse hors sujet ou insuffisamment adaptée.

**Amélioration de la correction factuelle**

Le score moyen de `answer_correctness` progresse de ***0,255 à 0,729***. Cela traduit une meilleure conformité aux réponses de référence sur le jeu évalué, notamment pour les questions statistiques et les classements.

**Amélioration de la récupération**

La progression de `context_precision` et de `context_recall` indique que les contextes utilisés pour répondre sont globalement plus pertinents et couvrent mieux les informations attendues. 

Les limites de ces métriques doivent néanmoins être considérées pour les questions SQL, dont les contextes peuvent être constitués de résultats tabulaires plutôt que de chunks textuels.

**Réduction des affirmations non étayées**

Le score moyen de `faithfulness` atteint presque 1,0 dans le tableau comparatif. Cela indique que, selon l'évaluation RAGAS, les réponses produites sont étayées par les contextes fournis.

**Couverture des entités encore perfectible**
`context_entity_recall` progresse de 0,362 à 0,501, mais reste sensiblement inférieur aux autres métriques. Le système peut donc encore manquer certaines entités attendues dans les contextes.

### Limites de l'évaluation
- Les résultats dépendent du jeu de validation, de la qualité des réponses de référence et du comportement du modèle évaluateur

- Une réponse refusant de conclure faute de données peut être plus fiable qu'une réponse chiffrée inventée, même si certaines métriques automatiques ne valorisent pas parfaitement ce refus

- Les `questions mixtes SQL + RAG` restent plus complexes, car elles nécessitent de combiner des données quantitatives avec des opinions issues de documents textuels. La qualité de la récupération et l'attribution correcte des opinions demeurent des axes d'amélioration.

Le détail des résultats, les scores par question et les commentaires personnalisés sont disponibles dans `Comparaison_prototypes.xlsx`.

## Structure du projet

![Structure projet](https://raw.githubusercontent.com/Diaure/Evaluation_Performances_LLM_SportSee/main/inputs/Capture.png)

## Conclusion
Ce projet met en œuvre une architecture hybride combinant SQL, recherche sémantique et génération de réponses par un LLM.

Les améliorations introduites visent à:
- mieux structurer et contrôler les réponses, 
- respecter les données disponibles,
- limiter les affirmations non justifiées.

L'évaluation comparative permet de mesurer ces évolutions et d'identifier les limites qui persistent.

Les scores moyens donnent une vue d'ensemble des performances, tandis que l'analyse par question permet de comprendre les différences entre les deux prototypes.







