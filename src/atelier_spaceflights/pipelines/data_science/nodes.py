"""Nodes du pipeline 'data_science' : découpage, entraînement, évaluation.

Fonctions pures : elles reçoivent des objets et en renvoient d'autres. Aucune
ne lit un fichier ni n'écrit un résultat ; c'est le catalogue qui s'en charge.
"""

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Colonne à prédire.
_CIBLE = "price"


def split_data(
    table: pd.DataFrame, options: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare la table en jeux d'apprentissage et de test.

    Args:
        table: table d'apprentissage (sortie de data_processing).
        options: dictionnaire `model_options` avec `features` (liste de colonnes),
            `test_size` (proportion du jeu de test) et `random_state` (graine).

    Returns:
        X_train, X_test, y_train, y_test, dans cet ordre.

    Raises:
        KeyError: si une feature demandée n'existe pas dans la table.
    """
    X = table[options["features"]]
    y = table[_CIBLE]
    return train_test_split(
        X,
        y,
        test_size=options["test_size"],
        random_state=options["random_state"],
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Entraîne une régression linéaire et la renvoie."""
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Évalue le modèle sur le jeu de test.

    Returns:
        metrics: `mae`, `r2` et `n_test`, en nombres Python natifs. Les conversions
            sont défensives : selon la version, une métrique peut sortir en type
            NumPy, et un entier NumPy (`np.int64`) n'est pas sérialisable en JSON.
        predictions: DataFrame `actual` / `prediction`, avec un index neuf
            (l'index de la table d'origine n'est pas un identifiant à conserver).
    """
    y_pred = regressor.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
    }
    predictions = pd.DataFrame({"actual": y_test.to_numpy(), "prediction": y_pred})
    return metrics, predictions
