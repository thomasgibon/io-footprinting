"""Calcule les comptes climatiques régionaux d'une année EXIOBASE."""

from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pymrio as mr


def construire_facteurs_prg100(index_emissions):
    facteurs = pd.Series(0.0, index=index_emissions, name="PRG100")
    noms = index_emissions.to_series().astype(str)
    co2 = noms.str.startswith("CO2 -") & ~noms.str.contains(
        "biogenic", case=False
    )
    methane = noms.str.startswith(("CH4 -", "CH4_bio -"))
    methane_fossile = noms.str.contains(
        "Extraction/production|Mining of|Oil refinery", regex=True
    )
    n2o = noms.str.startswith(("N2O -", "N2O_bio -"))

    facteurs.loc[co2] = 1.0
    facteurs.loc[methane] = 27.0
    facteurs.loc[methane & methane_fossile] = 29.8
    facteurs.loc[n2o] = 273.0
    if "SF6 - air" in facteurs.index:
        facteurs.loc["SF6 - air"] = 25_200.0
    return facteurs


def calculer(annee, archive, population, empreinte_sha256, version, systeme):
    debut = time.perf_counter()
    io = mr.parse_exiobase3(path=archive)
    regions = io.get_regions().tolist()

    facteurs = construire_facteurs_prg100(io.air_emissions.F.index)
    f_climat = facteurs @ io.air_emissions.F.fillna(0)
    fy_climat = facteurs @ io.air_emissions.F_Y.fillna(0)
    emissions_finales = fy_climat.groupby(level="region").sum().reindex(regions)
    production_observee = (
        f_climat.groupby(level="region").sum().reindex(regions)
        + emissions_finales
    )
    pib_prix_courants = (
        io.factor_inputs.F.fillna(0)
        .sum(axis=0)
        .groupby(level="region")
        .sum()
        .reindex(regions)
    )

    x = io.x.iloc[:, 0]
    s_climat = f_climat.div(x.replace(0, np.nan)).fillna(0)
    demande_regionale = io.Y.T.groupby(level="region").sum().T
    production_attribuable = (
        (s_climat * x).groupby(level="region").sum().reindex(regions)
        + emissions_finales
    )

    demande_valeurs = demande_regionale.to_numpy()
    demande_colonnes = demande_regionale.columns
    a = io.Z.to_numpy(copy=False)
    x_valeurs = x.to_numpy()
    colonnes_nulles = x_valeurs == 0
    np.divide(a, x_valeurs, out=a, where=~colonnes_nulles)
    a[:, colonnes_nulles] = 0
    a *= -1
    indices = np.arange(a.shape[0])
    a[indices, indices] += 1

    multiplicateur = np.linalg.solve(a.T, s_climat.to_numpy())
    empreinte_consommation = pd.Series(
        multiplicateur @ demande_valeurs, index=demande_colonnes
    ).reindex(regions) + emissions_finales

    resultat = pd.DataFrame(
        {
            "ghg_cba_kgco2e": empreinte_consommation,
            "ghg_pba_observe_kgco2e": production_observee,
            "ghg_pba_attribuable_kgco2e": production_attribuable,
            "ghg_final_direct_kgco2e": emissions_finales,
            "pib_exiobase_meur_courants": pib_prix_courants,
            "population": population.loc[annee, regions],
        },
        index=pd.Index(regions, name="region"),
    )
    resultat.insert(0, "annee", annee)
    resultat["version_exiobase"] = version
    resultat["systeme"] = systeme
    resultat["archive_sha256"] = empreinte_sha256
    resultat["temps_calcul_s"] = time.perf_counter() - debut
    return resultat.reset_index().set_index(["annee", "region"])


def lire_arguments():
    analyseur = ArgumentParser()
    analyseur.add_argument("--annee", type=int, required=True)
    analyseur.add_argument("--archive", type=Path, required=True)
    analyseur.add_argument("--population", type=Path, required=True)
    analyseur.add_argument("--sortie", type=Path, required=True)
    analyseur.add_argument("--sha256", required=True)
    analyseur.add_argument("--version", required=True)
    analyseur.add_argument("--systeme", required=True)
    return analyseur.parse_args()


if __name__ == "__main__":
    arguments = lire_arguments()
    donnees_population = pd.read_csv(
        arguments.population, sep="\t", index_col="Year"
    )
    comptes = calculer(
        arguments.annee,
        arguments.archive,
        donnees_population,
        arguments.sha256,
        arguments.version,
        arguments.systeme,
    )
    comptes.to_csv(arguments.sortie)
    print(
        f"{arguments.annee} : {len(comptes)} régions calculées en "
        f"{comptes['temps_calcul_s'].iloc[0]:.1f} s",
        flush=True,
    )
