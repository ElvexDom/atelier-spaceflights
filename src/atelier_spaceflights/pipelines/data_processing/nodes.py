"""Nodes du pipeline 'data_processing' : nettoyage et table d'apprentissage.

Fonctions pures : chaque node reçoit un DataFrame et en renvoie un nouveau,
sans toucher à l'entrée et sans lire ni écrire de fichier (c'est le rôle du
catalogue).
"""

import pandas as pd

# Seules valeurs acceptées pour un indicateur texte ('t' = vrai, 'f' = faux).
_INDICATEURS = {"t": True, "f": False}


def _to_boolean(serie: pd.Series) -> pd.Series:
    """Convertit une colonne 't'/'f' en booléens.

    Une valeur manquante reste manquante (dtype 'boolean', qui accepte <NA>).
    Toute autre valeur lève une ValueError plutôt que d'être convertie en
    silence : une donnée inattendue doit se voir tout de suite.
    """
    invalides = serie.dropna()[lambda s: ~s.isin(_INDICATEURS)].unique()
    if len(invalides) > 0:
        exemples = ", ".join(repr(v) for v in invalides[:5])
        raise ValueError(
            f"Indicateur invalide dans '{serie.name}' (attendu 't' ou 'f') : {exemples}"
        )
    return serie.map(_INDICATEURS).astype("boolean")


def _percentage_to_float(serie: pd.Series) -> pd.Series:
    """Convertit '90%' en 0.9. Une valeur manquante reste manquante."""
    return serie.str.replace("%", "", regex=False).astype(float) / 100


def _money_to_float(serie: pd.Series) -> pd.Series:
    """Convertit '$1,715.0' en 1715.0. Une valeur manquante reste manquante."""
    texte = serie.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return texte.astype(float)


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Prétraite les compagnies : note en proportion, agrément IATA en booléen.

    Args:
        companies: DataFrame brut (company_rating en '90%', iata_approved en 't'/'f').

    Returns:
        Une copie avec company_rating en float (0.9) et iata_approved en booléen.
    """
    companies = companies.copy()
    companies["company_rating"] = _percentage_to_float(companies["company_rating"])
    companies["iata_approved"] = _to_boolean(companies["iata_approved"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Prétraite les navettes : prix en nombre, contrôles en booléens.

    Args:
        shuttles: DataFrame brut (price en '$800', d_check_complete et
            moon_clearance_complete en 't'/'f').

    Returns:
        Une copie avec price en float (800.0) et les deux contrôles en booléens.
    """
    shuttles = shuttles.copy()
    shuttles["price"] = _money_to_float(shuttles["price"])
    shuttles["d_check_complete"] = _to_boolean(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _to_boolean(
        shuttles["moon_clearance_complete"]
    )
    return shuttles


# Colonnes dont l'absence rend une ligne inutilisable pour l'apprentissage :
# la cible (price) et les variables d'entrée du modèle (A5). Les autres colonnes
# sont conservées telles quelles, même si elles contiennent des valeurs vides.
_COLONNES_OBLIGATOIRES = [
    "passenger_capacity",
    "company_rating",
    "review_scores_rating",
    "price",
    "engines",
    "crew",
]


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Construit la table d'apprentissage : navettes + avis + compagnies.

    Les jointures sont à gauche (aucune navette n'est perdue par la jointure) et
    leur cardinalité est vérifiée : un avis dupliqué ou une compagnie dupliquée
    lève une erreur au lieu de multiplier des lignes sans prévenir.

    Args:
        shuttles: navettes prétraitées (clé : id, lien vers la compagnie : company_id).
        companies: compagnies prétraitées (clé : id).
        reviews: avis bruts (clé : shuttle_id, un seul avis par navette).

    Returns:
        Table dont les lignes incomplètes sur les colonnes obligatoires sont
        retirées, avec un index réinitialisé.

    Raises:
        pandas.errors.MergeError: si la cardinalité attendue n'est pas respectée.
    """
    table = shuttles.merge(
        reviews,
        left_on="id",
        right_on="shuttle_id",
        how="left",
        validate="one_to_one",
    )
    companies = companies.rename(columns={"id": "company_id"})
    table = table.merge(companies, on="company_id", how="left", validate="many_to_one")
    table = table.dropna(subset=_COLONNES_OBLIGATOIRES)
    return table.reset_index(drop=True)
