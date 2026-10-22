"""Pipelines du projet : enregistrement explicite (séance C, C2).

Le pipeline `data_science` sert de modèle réutilisable (split, entraînement,
évaluation). On en tire une instance par configuration grâce aux namespaces :
les fonctions ne sont jamais dupliquées, seuls les noms des jeux de données et
des paramètres changent (`baseline.regressor`, `params:candidate.model_options`...).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from atelier_spaceflights.pipelines import data_processing, data_science, reporting


def register_pipelines() -> dict[str, Pipeline]:
    """Enregistre les pipelines du projet.

    Returns:
        Un dictionnaire nom -> `Pipeline`.
    """
    preparation = data_processing.create_pipeline()
    modele = data_science.create_pipeline()

    # `model_input_table` n'est pas préfixée : les branches partagent la même table.
    baseline = Pipeline(
        modele.nodes, inputs={"model_input_table"}, namespace="baseline"
    )
    candidate = Pipeline(
        modele.nodes, inputs={"model_input_table"}, namespace="candidate"
    )
    modelisation = baseline + candidate

    # Le graphique est celui de la branche baseline : le namespace préfixe à la fois
    # son entrée (`baseline.predictions`) et sa sortie (`baseline.prediction_plot`).
    rapport = Pipeline(reporting.create_pipeline().nodes, namespace="baseline")

    return {
        "__default__": preparation + modelisation + rapport,
        "data_processing": preparation,
        "data_science": modelisation,
        "baseline": baseline,
        "candidate": candidate,
        "reporting": rapport,
    }
