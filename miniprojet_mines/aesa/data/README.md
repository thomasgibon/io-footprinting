# Données fournies

Le dossier `raw` permet d'exécuter les notebooks sans téléchargement ni accès à une API.

| Fichier | Contenu | Unité / structure | Provenance dans ce dépôt |
|---|---|---|---|
| `c1_pathway_IPCC.csv` | trajectoires mondiales de catégorie C1, médiane, p25 et p75 | Gt CO₂-éq/an, 2026–2100 | jeu pédagogique préparé à partir des scénarios de l'AR6 du GIEC |
| `c3_pathway_IPCC.csv` | trajectoires mondiales de catégorie C3, médiane, p25 et p75 | Gt CO₂-éq/an, 2026–2100 | jeu pédagogique préparé à partir des scénarios de l'AR6 du GIEC |
| `greenhouse-gas-emissions.csv` | émissions historiques de GES par territoire et agrégat mondial, usage des sols inclus | tonnes CO₂-éq/an | export Our World in Data |
| `population_owid.csv` | population par pays | personnes | Our World in Data |
| `pib_ppa_banque_mondiale_owid.csv` | PIB en parité de pouvoir d'achat | dollars internationaux constants de 2021 | Banque mondiale, redistribution Our World in Data |
| `pib_ppa_par_habitant_banque_mondiale_owid.csv` | PIB par habitant en parité de pouvoir d'achat | dollars internationaux constants de 2021 par personne | Banque mondiale, redistribution Our World in Data |

## Point de vigilance

Les fichiers C1/C3 sont des **données préparées à partir de scénarios du GIEC**, et non les bases brutes complètes de l'AR6. Avant une publication externe, il faudra compléter leur fiche de provenance avec la requête, les variables, la version de la base et les transformations ayant servi à les produire. Cette limite de traçabilité est volontairement visible : elle peut servir à discuter en classe de la reproductibilité d'une AESA.

Les fichiers de `raw` ne doivent pas être modifiés par les notebooks. Toute donnée transformée ou tout résultat tabulaire ultérieur devra être écrit dans un dossier `data/processed`.
