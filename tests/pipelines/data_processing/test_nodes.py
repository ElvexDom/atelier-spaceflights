"""Tests unitaires des nodes de data_processing.

Les données sont construites en mémoire : ces tests ne dépendent ni du
catalogue, ni des fichiers, ni de Kedro. Une erreur ici est donc une erreur de
calcul, pas de configuration.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError
from pandas.testing import assert_frame_equal, assert_series_equal

from atelier_spaceflights.pipelines.data_processing.nodes import (
    create_model_input_table,
    preprocess_companies,
    preprocess_shuttles,
)


@pytest.fixture
def raw_companies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "company_rating": ["90%", "100%", None],
            "iata_approved": ["t", "f", None],
        }
    )


@pytest.fixture
def raw_shuttles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 11],
            "price": ["$800", "$1,715.0"],
            "d_check_complete": ["t", "f"],
            "moon_clearance_complete": ["f", "f"],
        }
    )


# --- Cas 1 : conversions, sans modifier l'entrée ---------------------------


def test_preprocess_companies_convertit_sans_modifier_l_entree(raw_companies):
    avant = raw_companies.copy(deep=True)

    resultat = preprocess_companies(raw_companies)

    assert_series_equal(
        resultat["company_rating"],
        pd.Series([0.9, 1.0, np.nan], name="company_rating"),
    )
    # La valeur manquante reste manquante (elle n'est pas devenue False).
    assert_series_equal(
        resultat["iata_approved"],
        pd.Series([True, False, pd.NA], dtype="boolean", name="iata_approved"),
    )
    assert resultat["id"].tolist() == [1, 2, 3]
    assert resultat is not raw_companies
    assert_frame_equal(raw_companies, avant)


def test_preprocess_shuttles_convertit_sans_modifier_l_entree(raw_shuttles):
    avant = raw_shuttles.copy(deep=True)

    resultat = preprocess_shuttles(raw_shuttles)

    assert_series_equal(resultat["price"], pd.Series([800.0, 1715.0], name="price"))
    assert_series_equal(
        resultat["d_check_complete"],
        pd.Series([True, False], dtype="boolean", name="d_check_complete"),
    )
    assert_series_equal(
        resultat["moon_clearance_complete"],
        pd.Series([False, False], dtype="boolean", name="moon_clearance_complete"),
    )
    assert resultat["id"].tolist() == [10, 11]
    assert_frame_equal(raw_shuttles, avant)


# --- Cas 2 : indicateur invalide -------------------------------------------


@pytest.mark.parametrize("valeur", ["x", "T", "true", "1", ""])
def test_indicateur_invalide_des_navettes_leve_value_error(raw_shuttles, valeur):
    raw_shuttles.loc[0, "d_check_complete"] = valeur

    with pytest.raises(ValueError, match="Indicateur invalide"):
        preprocess_shuttles(raw_shuttles)


def test_indicateur_invalide_des_compagnies_nomme_la_colonne(raw_companies):
    raw_companies.loc[0, "iata_approved"] = "oui"

    with pytest.raises(ValueError, match="iata_approved"):
        preprocess_companies(raw_companies)


# --- Cas 3 : jointure -------------------------------------------------------


@pytest.fixture
def shuttles() -> pd.DataFrame:
    """Navettes prétraitées. La 2 n'a pas d'avis chiffré, la 4 n'a pas d'équipage,
    la 5 appartient à une compagnie inconnue : seules la 1 et la 3 sont complètes."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "company_id": [10, 10, 20, 20, 30],
            "price": [100.0, 200.0, 300.0, 400.0, 500.0],
            "passenger_capacity": [2, 4, 6, 8, 10],
            "engines": [1.0, 2.0, 3.0, 4.0, 5.0],
            "crew": [1.0, 1.0, 2.0, np.nan, 3.0],
        }
    )


@pytest.fixture
def companies() -> pd.DataFrame:
    """Compagnies prétraitées. `company_location` est vide pour la compagnie 20 :
    cette colonne n'est pas obligatoire, la navette 3 doit donc être conservée."""
    return pd.DataFrame(
        {
            "id": [10, 20],
            "company_rating": [0.9, 0.5],
            "company_location": ["Mars", None],
            "iata_approved": pd.array([True, False], dtype="boolean"),
        }
    )


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shuttle_id": [1, 2, 3, 4, 5],
            "review_scores_rating": [90.0, np.nan, 80.0, 70.0, 60.0],
        }
    )


def test_jointure_conserve_les_bonnes_lignes(shuttles, companies, reviews):
    resultat = create_model_input_table(shuttles, companies, reviews)

    assert resultat["id"].tolist() == [1, 3]
    assert resultat["company_rating"].tolist() == [0.9, 0.5]
    assert resultat["review_scores_rating"].tolist() == [90.0, 80.0]
    assert resultat["price"].tolist() == [100.0, 300.0]
    # Index réinitialisé, et colonnes non obligatoires conservées sans suffixe.
    assert resultat.index.equals(pd.RangeIndex(2))
    assert set(resultat.columns) == {
        "id",
        "company_id",
        "price",
        "passenger_capacity",
        "engines",
        "crew",
        "shuttle_id",
        "review_scores_rating",
        "company_rating",
        "company_location",
        "iata_approved",
    }


def test_jointure_ne_modifie_pas_ses_entrees(shuttles, companies, reviews):
    avant = [df.copy(deep=True) for df in (shuttles, companies, reviews)]

    create_model_input_table(shuttles, companies, reviews)

    for apres, initial in zip((shuttles, companies, reviews), avant, strict=True):
        assert_frame_equal(apres, initial)


def test_avis_duplique_leve_une_erreur(shuttles, companies, reviews):
    reviews = pd.concat([reviews, reviews.iloc[[0]]], ignore_index=True)

    with pytest.raises(MergeError, match="one-to-one"):
        create_model_input_table(shuttles, companies, reviews)


def test_compagnie_dupliquee_leve_une_erreur(shuttles, companies, reviews):
    companies = pd.concat([companies, companies.iloc[[0]]], ignore_index=True)

    with pytest.raises(MergeError, match="many-to-one"):
        create_model_input_table(shuttles, companies, reviews)
