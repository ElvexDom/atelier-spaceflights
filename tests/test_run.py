"""Tests d'assemblage du projet : les pipelines sont enregistrés et bien câblés.

Aucun nœud n'est exécuté et aucun fichier de données n'est lu ni écrit : on
vérifie seulement la structure (noms, tags, entrées, sorties, ordre). Une erreur
ici signale un mauvais câblage, pas un problème de calcul.

Ces tests remplacent le test de démonstration généré par le modèle Kedro, qui
attendait qu'un projet sans pipeline échoue au lancement.
"""

from pathlib import Path

import pytest
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

RACINE = Path(__file__).resolve().parents[1]

# Nom du nœud -> tags attendus.
TAGS_ATTENDUS = {
    "preprocess_companies_node": {"prepare"},
    "preprocess_shuttles_node": {"prepare"},
    "create_model_input_table_node": {"prepare"},
    "split_data_node": {"train"},
    "train_model_node": {"train"},
    "evaluate_model_node": {"evaluate"},
}


@pytest.fixture(scope="module")
def registre():
    bootstrap_project(RACINE)
    return pipelines


def noeuds(pipeline) -> dict:
    """Nœuds d'un pipeline, indexés par nom."""
    return {n.name: n for n in pipeline.nodes}


def test_les_pipelines_sont_enregistres(registre):
    assert set(registre) == {"__default__", "data_processing", "data_science"}


def test_le_pipeline_par_defaut_contient_les_six_noeuds(registre):
    assert set(noeuds(registre["__default__"])) == set(TAGS_ATTENDUS)
    assert len(registre["data_processing"].nodes) == 3
    assert len(registre["data_science"].nodes) == 3


def test_chaque_noeud_porte_le_bon_tag(registre):
    tous = noeuds(registre["__default__"])
    assert {nom: set(n.tags) for nom, n in tous.items()} == TAGS_ATTENDUS


def test_entrees_de_la_table_dans_l_ordre_des_arguments(registre):
    # L'ordre compte : Kedro passe les entrées à la fonction par position.
    noeud = noeuds(registre["data_processing"])["create_model_input_table_node"]
    assert noeud.inputs == [
        "preprocessed_shuttles",
        "preprocessed_companies",
        "reviews",
    ]
    assert noeud.outputs == ["model_input_table"]


def test_sorties_du_decoupage_dans_l_ordre(registre):
    noeud = noeuds(registre["data_science"])["split_data_node"]
    assert noeud.inputs == ["model_input_table", "params:model_options"]
    assert noeud.outputs == ["X_train", "X_test", "y_train", "y_test"]


def test_evaluation_produit_les_metriques_et_les_predictions(registre):
    noeud = noeuds(registre["data_science"])["evaluate_model_node"]
    assert noeud.inputs == ["regressor", "X_test", "y_test"]
    assert noeud.outputs == ["metrics", "predictions"]


def test_entrees_exterieures_de_chaque_pipeline(registre):
    # Ce que chaque pipeline doit trouver dans le catalogue ou la configuration.
    assert registre["data_processing"].inputs() == {"companies", "shuttles", "reviews"}
    assert registre["data_science"].inputs() == {
        "model_input_table",
        "params:model_options",
    }


def test_ordre_d_execution_deduit_des_donnees(registre):
    ordre = [n.name for n in registre["__default__"].nodes]
    position = ordre.index

    assert position("preprocess_companies_node") < position(
        "create_model_input_table_node"
    )
    assert position("preprocess_shuttles_node") < position(
        "create_model_input_table_node"
    )
    assert position("create_model_input_table_node") < position("split_data_node")
    assert position("split_data_node") < position("train_model_node")
    assert position("train_model_node") < position("evaluate_model_node")
