"""Pipeline 'reporting' : graphique des prédictions."""

from kedro.pipeline import Node, Pipeline

from .nodes import plot_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=plot_predictions,
                inputs="predictions",
                outputs="prediction_plot",
                name="plot_predictions_node",
                tags=["reporting"],
            ),
        ]
    )
