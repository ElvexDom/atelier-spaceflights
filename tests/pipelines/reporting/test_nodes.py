"""Tests unitaires du node de reporting.

Données en mémoire, aucun fichier : on vérifie que la fonction renvoie une figure
correcte sans rien écrire (l'écriture est l'affaire du catalogue).
"""

import json

import pandas as pd
import plotly.graph_objects as go
import pytest
from pandas.testing import assert_frame_equal

from atelier_spaceflights.pipelines.reporting.nodes import plot_predictions


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # Volontairement non triés : un node qui trierait l'entrée sur place
            # serait ainsi détecté par le test de non-modification.
            "actual": [300.0, 100.0, 400.0, 200.0],
            "prediction": [320.0, 110.0, 380.0, 190.0],
        }
    )


def test_renvoie_une_figure_plotly(predictions):
    assert isinstance(plot_predictions(predictions), go.Figure)


def test_un_seul_nuage_avec_un_point_par_prediction(predictions):
    figure = plot_predictions(predictions)

    assert len(figure.data) == 1
    trace = figure.data[0]
    assert trace.type == "scatter"
    assert trace.mode == "markers"
    assert len(trace.x) == len(trace.y) == 4


def test_reel_en_x_et_prediction_en_y(predictions):
    trace = plot_predictions(predictions).data[0]

    assert list(trace.x) == [300.0, 100.0, 400.0, 200.0]
    assert list(trace.y) == [320.0, 110.0, 380.0, 190.0]


def test_ne_modifie_pas_l_entree_et_n_ecrit_aucun_fichier(
    predictions, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    avant = predictions.copy(deep=True)

    plot_predictions(predictions)

    assert_frame_equal(predictions, avant)
    assert list(tmp_path.iterdir()) == []


def test_la_figure_tient_dans_un_json(predictions):
    # C'est ce que fait plotly.JSONDataset au moment de sauvegarder.
    contenu = json.loads(plot_predictions(predictions).to_json())

    assert contenu["data"][0]["x"] == [300.0, 100.0, 400.0, 200.0]
    assert contenu["data"][0]["y"] == [320.0, 110.0, 380.0, 190.0]


def test_colonne_manquante_leve_une_erreur(predictions):
    with pytest.raises(ValueError, match="prediction"):
        plot_predictions(predictions.drop(columns=["prediction"]))
