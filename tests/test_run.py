"""Tests d'assemblage du projet : les pipelines sont enregistrés et bien câblés.

Aucun nœud n'est exécuté et aucune donnée n'est lue ni écrite : on vérifie la
structure (noms, tags, entrées, sorties, ordre), la présence des paramètres de
chaque branche et la déclaration au catalogue des sorties à conserver. Une erreur
ici signale un mauvais câblage, pas un problème de calcul.

Depuis la séance C, le modèle (split, entraînement, évaluation) est instancié une
fois par branche grâce aux namespaces : `baseline.split_data_node`,
`candidate.split_data_node`... Pour ajouter une branche, il suffit de l'ajouter à
BRANCHES.

Ces tests remplacent le test de démonstration généré par le modèle Kedro, qui
attendait qu'un projet sans pipeline échoue au lancement.
"""

from pathlib import Path

import pytest
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

RACINE = Path(__file__).resolve().parents[1]
BRANCHES = ("baseline", "candidate")

# Nom du nœud -> tags attendus.
TAGS_ATTENDUS = {
    "preprocess_companies_node": {"prepare"},
    "preprocess_shuttles_node": {"prepare"},
    "create_model_input_table_node": {"prepare"},
    "baseline.plot_predictions_node": {"reporting"},
}
for _branche in BRANCHES:
    TAGS_ATTENDUS[f"{_branche}.split_data_node"] = {"train"}
    TAGS_ATTENDUS[f"{_branche}.train_model_node"] = {"train"}
    TAGS_ATTENDUS[f"{_branche}.evaluate_model_node"] = {"evaluate"}


@pytest.fixture(scope="module")
def registre():
    bootstrap_project(RACINE)
    return pipelines


@pytest.fixture(scope="module")
def contexte(registre):
    with KedroSession.create(RACINE) as session:
        yield session.load_context()


def noeuds(pipeline) -> dict:
    """Nœuds d'un pipeline, indexés par nom."""
    return {n.name: n for n in pipeline.nodes}


def test_les_pipelines_sont_enregistres(registre):
    attendus = {"__default__", "data_processing", "data_science", "reporting"}
    assert set(registre) == attendus | set(BRANCHES)


def test_le_pipeline_par_defaut_contient_tous_les_noeuds(registre):
    assert set(noeuds(registre["__default__"])) == set(TAGS_ATTENDUS)
    assert len(registre["data_processing"].nodes) == 3
    assert len(registre["data_science"].nodes) == 3 * len(BRANCHES)
    assert len(registre["reporting"].nodes) == 1
    for branche in BRANCHES:
        assert len(registre[branche].nodes) == 3


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


@pytest.mark.parametrize("branche", BRANCHES)
def test_sorties_du_decoupage_dans_l_ordre(registre, branche):
    noeud = noeuds(registre[branche])[f"{branche}.split_data_node"]
    # La table est partagée (non préfixée) ; les paramètres et les sorties sont préfixés.
    assert noeud.inputs == ["model_input_table", f"params:{branche}.model_options"]
    assert noeud.outputs == [
        f"{branche}.X_train",
        f"{branche}.X_test",
        f"{branche}.y_train",
        f"{branche}.y_test",
    ]


@pytest.mark.parametrize("branche", BRANCHES)
def test_evaluation_produit_les_metriques_et_les_predictions(registre, branche):
    noeud = noeuds(registre[branche])[f"{branche}.evaluate_model_node"]
    assert noeud.inputs == [
        f"{branche}.regressor",
        f"{branche}.X_test",
        f"{branche}.y_test",
    ]
    assert noeud.outputs == [f"{branche}.metrics", f"{branche}.predictions"]


def test_le_graphique_lit_les_predictions_et_ecrit_sous_baseline(registre):
    # Le namespace doit préfixer l'entrée ET la sortie : sans le préfixe de sortie,
    # le graphique n'est plus déclaré au catalogue et n'est pas écrit sur disque.
    noeud = noeuds(registre["reporting"])["baseline.plot_predictions_node"]
    assert noeud.inputs == ["baseline.predictions"]
    assert noeud.outputs == ["baseline.prediction_plot"]


def test_entrees_exterieures_de_chaque_pipeline(registre):
    # Ce que chaque pipeline doit trouver dans le catalogue ou la configuration.
    assert registre["data_processing"].inputs() == {"companies", "shuttles", "reviews"}
    assert registre["data_science"].inputs() == {"model_input_table"} | {
        f"params:{b}.model_options" for b in BRANCHES
    }
    for branche in BRANCHES:
        assert registre[branche].inputs() == {
            "model_input_table",
            f"params:{branche}.model_options",
        }
    assert registre["reporting"].inputs() == {"baseline.predictions"}


def test_les_branches_reutilisent_les_memes_fonctions(registre):
    # Aucune duplication de code : chaque branche appelle exactement les mêmes fonctions.
    tous = noeuds(registre["__default__"])
    for nom in ("split_data_node", "train_model_node", "evaluate_model_node"):
        fonctions = {tous[f"{b}.{nom}"].func for b in BRANCHES}
        assert len(fonctions) == 1, f"{nom} est dupliqué entre les branches"


def test_les_branches_partagent_une_seule_table(registre):
    tous = registre["__default__"].nodes
    producteurs = [n.name for n in tous if "model_input_table" in n.outputs]
    consommateurs = sorted(n.name for n in tous if "model_input_table" in n.inputs)
    assert producteurs == ["create_model_input_table_node"]
    assert consommateurs == sorted(f"{b}.split_data_node" for b in BRANCHES)


def test_ordre_d_execution_deduit_des_donnees(registre):
    ordre = [n.name for n in registre["__default__"].nodes]
    position = ordre.index

    assert position("preprocess_companies_node") < position(
        "create_model_input_table_node"
    )
    assert position("preprocess_shuttles_node") < position(
        "create_model_input_table_node"
    )
    for branche in BRANCHES:
        assert position("create_model_input_table_node") < position(
            f"{branche}.split_data_node"
        )
        assert position(f"{branche}.split_data_node") < position(
            f"{branche}.train_model_node"
        )
        assert position(f"{branche}.train_model_node") < position(
            f"{branche}.evaluate_model_node"
        )
    assert position("baseline.evaluate_model_node") < position(
        "baseline.plot_predictions_node"
    )


@pytest.mark.parametrize("branche", BRANCHES)
def test_chaque_branche_a_ses_parametres(contexte, branche):
    options = contexte.params[branche]["model_options"]
    assert set(options) == {"test_size", "random_state", "features"}
    assert 0 < options["test_size"] < 1
    assert len(options["features"]) >= 1


def test_les_sorties_a_conserver_sont_declarees_au_catalogue(contexte):
    # Une sortie non déclarée serait gardée en mémoire puis perdue, sans erreur.
    catalogue = contexte.catalog
    attendues = [
        f"{b}.{nom}"
        for b in BRANCHES
        for nom in ("regressor", "metrics", "predictions")
    ]
    attendues += ["baseline.prediction_plot", "model_input_table"]
    manquantes = [nom for nom in attendues if nom not in catalogue]
    assert manquantes == []
