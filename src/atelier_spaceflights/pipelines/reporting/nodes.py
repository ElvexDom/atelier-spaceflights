"""Nodes du pipeline 'reporting' : visualisation des résultats.

Fonction pure : elle reçoit le DataFrame produit par l'évaluation et renvoie une
figure ; elle n'écrit aucun fichier (c'est le catalogue qui sauvegarde la figure).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_predictions(predictions: pd.DataFrame) -> go.Figure:
    """Nuage de points : prix réel (x) et prix prédit (y) du jeu de test.

    Args:
        predictions: DataFrame avec les colonnes `actual` et `prediction`.

    Returns:
        Une figure Plotly (`go.Figure`) avec un point par ligne du jeu de test.
    """
    return px.scatter(
        predictions,
        x="actual",
        y="prediction",
        title="Prix prédit en fonction du prix réel",
        labels={"actual": "Prix réel", "prediction": "Prix prédit"},
    )
