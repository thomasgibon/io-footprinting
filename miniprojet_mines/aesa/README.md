# AESA — limites planétaires et partage du budget carbone

Ce dépôt est un support pédagogique en français destiné à des étudiantes et étudiants de master. Il introduit l'évaluation absolue de la soutenabilité environnementale (AESA), puis accompagne la construction et la discussion de trajectoires de décarbonation obtenues par descente d'échelle d'une trajectoire mondiale.

La terminologie française suit en priorité la publication du Service des données et études statistiques (SDES), [*La France face aux neuf limites planétaires*](https://www.statistiques.developpement-durable.gouv.fr/edition-numerique/la-france-face-aux-neuf-limites-planetaires/partie3-quelles-utilisations-du-cadre-des-limites).

## Parcours du semestre

1. `01_introduction_aesa.ipynb` — limites planétaires, logique de l'AESA et trajectoires mondiales C1/C3 du GIEC.
2. `02_principes_de_partage.ipynb` — calcul et comparaison de budgets nationaux selon quatre principes de partage.
3. Notebook à venir — comparaison, analyse de sensibilité et préparation du poster.

Le poster final devra expliciter la question étudiée (pays, secteur ou individus), comparer plusieurs règles de partage, discuter leurs présupposés et leurs conséquences distributives, puis formuler une recommandation argumentée.

## Démarrage

Depuis la racine du dépôt :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Ouvrir ensuite [`notebooks/01_introduction_aesa.ipynb`](notebooks/01_introduction_aesa.ipynb). Le notebook fonctionne hors ligne : les données requises sont déjà dans `data/raw`.

## Organisation

```text
aesa/
├── data/raw/       # trajectoires C1/C3 et émissions historiques
├── doc/            # rapport EEA/FOEN de référence
├── figures/        # graphiques exportés par les notebooks
├── notebooks/      # supports et exercices
├── README.md
└── requirements.txt
```

Les fichiers de `data/raw` sont conservés tels quels. Leur provenance et les choix d'interprétation sont documentés dans les notebooks et dans `data/README.md`.
