"""Pipelines du projet : enregistrement explicite (séance C, C2 et C4).

Le pipeline `data_science` sert de modèle réutilisable (split, entraînement,
évaluation). On en tire une instance par configuration grâce aux namespaces :
les fonctions ne sont jamais dupliquées, seuls les noms des jeux de données et
des paramètres changent (`baseline.regressor`, `params:candidate.model_options`...).

Pour ajouter une branche : l'ajouter à BRANCHES et décrire ses options dans
`conf/base/parameters_data_science.yml`. Le catalogue n'a pas à changer : ses
factories (`{namespace}.regressor`...) couvrent déjà tous les namespaces.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from atelier_spaceflights.pipelines import data_processing, data_science, reporting

BRANCHES = ("baseline", "candidate", "capacity_rating")


def register_pipelines() -> dict[str, Pipeline]:
    """Enregistre les pipelines du projet.

    Returns:
        Un dictionnaire nom -> `Pipeline`.
    """
    preparation = data_processing.create_pipeline()
    modele = data_science.create_pipeline()

    # `model_input_table` n'est pas préfixée : toutes les branches partagent la même table.
    branches = {
        nom: Pipeline(modele.nodes, inputs={"model_input_table"}, namespace=nom)
        for nom in BRANCHES
    }
    modelisation = sum(branches.values())

    # Le graphique est celui de la branche baseline : le namespace préfixe à la fois
    # son entrée (`baseline.predictions`) et sa sortie (`baseline.prediction_plot`).
    rapport = Pipeline(reporting.create_pipeline().nodes, namespace="baseline")

    return {
        "__default__": preparation + modelisation + rapport,
        "data_processing": preparation,
        "data_science": modelisation,
        "reporting": rapport,
        **branches,
    }
