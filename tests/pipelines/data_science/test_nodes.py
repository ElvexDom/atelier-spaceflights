"""Tests unitaires des nodes de data_science.

Aucun fichier, aucun catalogue : les tables sont construites en mémoire, avec
des valeurs choisies pour que le résultat attendu se calcule à la main.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from atelier_spaceflights.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)

FEATURES = ["engines", "passenger_capacity", "crew"]


@pytest.fixture
def table() -> pd.DataFrame:
    """19 lignes, comme le mini-jeu du tutoriel : 19 -> 15 apprentissage + 4 test."""
    n = 19
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "engines": np.arange(1, n + 1, dtype=float),
            "passenger_capacity": np.arange(n) % 5 + 1,
            "crew": (np.arange(n) % 3 + 1).astype(float),
            "price": np.arange(n) * 10.0 + 100.0,
        }
    )


@pytest.fixture
def options() -> dict:
    return {"test_size": 0.2, "random_state": 3, "features": list(FEATURES)}


# --- Cas 4 : découpage ------------------------------------------------------


def test_split_data_decoupe_19_lignes_en_15_et_4(table, options):
    X_train, X_test, y_train, y_test = split_data(table, options)

    # L'ordre de retour compte : deux tables, puis deux séries.
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series) and isinstance(y_test, pd.Series)
    assert (len(X_train), len(X_test), len(y_train), len(y_test)) == (15, 4, 15, 4)
    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert y_train.name == "price" and y_test.name == "price"
    # Aucune ligne perdue, aucune ligne partagée, et X reste aligné sur y.
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(table.index)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_split_data_est_reproductible_avec_la_meme_graine(table, options):
    premier = split_data(table, options)
    second = split_data(table, options)

    for a, b in zip(premier, second, strict=True):
        assert a.equals(b)
    # Une autre graine donne un autre découpage.
    autre = split_data(table, {**options, "random_state": 4})
    assert not premier[1].index.equals(autre[1].index)


# --- Cas 5 : feature inexistante --------------------------------------------


def test_feature_inexistante_leve_key_error(table, options):
    options["features"] = ["engines", "n_existe_pas"]

    with pytest.raises(KeyError, match="n_existe_pas"):
        split_data(table, options)


# --- Entraînement ------------------------------------------------------------


def test_train_model_retrouve_une_relation_lineaire_exacte():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 0, 2, 1, 3, 2]})
    y = 3 * X["a"] + 2 * X["b"] + 5

    regressor = train_model(X, y)

    assert isinstance(regressor, LinearRegression)
    assert regressor.coef_ == pytest.approx([3, 2])
    assert regressor.intercept_ == pytest.approx(5)


# --- Cas 6 : évaluation d'un modèle simple connu ----------------------------


@pytest.fixture
def regressor_connu() -> LinearRegression:
    """Modèle y = 2x + 1, appris sur des points exacts."""
    return LinearRegression().fit(
        pd.DataFrame({"x": [0, 1, 2, 3]}), pd.Series([1, 3, 5, 7])
    )


def test_evaluate_model_metriques_et_predictions(regressor_connu):
    # Le modèle prédit 21 et 23 ; les vraies valeurs sont 21 et 24.
    # MAE = (0 + 1) / 2 = 0,5 ; R² = 1 - 1 / 4,5 (somme des carrés totale = 4,5).
    X_test = pd.DataFrame({"x": [10, 11]}, index=[5, 9])
    y_test = pd.Series([21.0, 24.0], index=[5, 9], name="price")

    metrics, predictions = evaluate_model(regressor_connu, X_test, y_test)

    assert set(metrics) == {"mae", "r2", "n_test"}
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(1 - 1 / 4.5)
    assert metrics["n_test"] == 2
    # Types Python natifs, donc sérialisables en JSON.
    assert type(metrics["mae"]) is float
    assert type(metrics["r2"]) is float
    assert type(metrics["n_test"]) is int
    json.dumps(metrics)

    assert list(predictions.columns) == ["actual", "prediction"]
    assert predictions["actual"].tolist() == [21.0, 24.0]
    assert predictions["prediction"].tolist() == pytest.approx([21.0, 23.0])
    # L'index d'origine (5, 9) n'est pas réutilisé comme identifiant.
    assert predictions.index.equals(pd.RangeIndex(2))
