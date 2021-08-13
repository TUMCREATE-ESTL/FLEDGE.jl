"""Plots module."""

import cv2
import itertools
from multimethod import multimethod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.electric_grid_models
import mesmo.thermal_grid_models
import mesmo.utils

if mesmo.config.config['plots']['add_basemap']:
    # Basemap requires `contextily`, which is an optional dependency, due to needing installation through `conda`.
    import contextily as ctx

logger = mesmo.config.get_logger(__name__)


class ElectricGridGraph(nx.DiGraph):
    """Electric grid graph object."""

    edge_by_line_name: pd.Series
    transformer_nodes: list
    node_positions: dict
    node_labels: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain electric grid data.
        electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)

        self.__init__(
            electric_grid_data
        )

    @multimethod
    def __init__(
            self,
            electric_grid_data: mesmo.data_interface.ElectricGridData
    ):

        # Create electric grid graph.
        super().__init__()
        self.add_nodes_from(
            electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].tolist()
        )
        self.add_edges_from(
            electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
        )

        # Obtain edges indexed by line name.
        self.edge_by_line_name = (
            pd.Series(
                electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False),
                index=electric_grid_data.electric_grid_lines.loc[:, 'line_name']
            )
        )

        # Obtain transformer nodes (secondary nodes of transformers).
        self.transformer_nodes = (
            electric_grid_data.electric_grid_transformers.loc[:, 'node_2_name'].tolist()
        )

        # Obtain node positions / labels.
        if pd.notnull(electric_grid_data.electric_grid_nodes.loc[:, ['longitude', 'latitude']]).any().any():
            self.node_positions = (
                electric_grid_data.electric_grid_nodes.loc[:, ['longitude', 'latitude']].T.to_dict('list')
            )
        else:
            # If latitude / longitude are not defined, generate node positions based on networkx layout.
            self.node_positions = (
                nx.spring_layout(self)
            )
            # Only keep positions for nodes with line connections.
            # Only keep positions for nodes with line connections.
            for node_name in self.node_positions:
                if (
                        node_name not in
                        electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].values.ravel()
                ):
                    self.node_positions[node_name] = [np.nan, np.nan]
        self.node_labels = (
            electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].to_dict()
        )


class ThermalGridGraph(nx.DiGraph):
    """Thermal grid graph object."""

    edge_by_line_name: pd.Series
    node_positions: dict
    node_labels: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = mesmo.data_interface.ThermalGridData(scenario_name)

        self.__init__(
            thermal_grid_data
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_data: mesmo.data_interface.ThermalGridData
    ):

        # Create thermal grid graph.
        super().__init__()
        self.add_nodes_from(
            thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name'].tolist()
        )
        self.add_edges_from(
            thermal_grid_data.thermal_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
        )

        # Obtain edges indexed by line name.
        self.edge_by_line_name = (
            pd.Series(
                thermal_grid_data.thermal_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False),
                index=thermal_grid_data.thermal_grid_lines.loc[:, 'line_name']
            )
        )

        # Obtain node positions / labels.
        if pd.notnull(thermal_grid_data.thermal_grid_nodes.loc[:, ['longitude', 'latitude']]).any().any():
            self.node_positions = (
                thermal_grid_data.thermal_grid_nodes.loc[:, ['longitude', 'latitude']].T.to_dict('list')
            )
        else:
            # If latitude / longitude are not defined, generate node positions based on networkx layout.
            self.node_positions = (
                nx.spring_layout(self)
            )
            # Only keep positions for nodes with line connections.
            for node_name in self.node_positions:
                if (
                        node_name not in
                        thermal_grid_data.thermal_grid_lines.loc[:, ['node_1_name', 'node_2_name']].values.ravel()
                ):
                    self.node_positions[node_name] = [np.nan, np.nan]
        self.node_labels = (
            thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name'].to_dict()
        )


def create_video(
        name: str,
        labels: pd.Index,
        results_path: str
):

    # Obtain images / frames based on given name / labels.
    images = []
    for label in labels:
        if type(label) is pd.Timestamp:
            filename = f"{name}_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
            images.append(cv2.imread(os.path.join(results_path, filename)))
        if len(images) == 0:
            raise FileNotFoundError(f"No images / frames found for video of '{name}'. Check if given labels are valid timesteps.")

    # Setup video.
    video_writer = (
        cv2.VideoWriter(
            os.path.join(results_path, f'{name}.avi'),  # Filename.
            cv2.VideoWriter_fourcc(*'XVID'),  # Format.
            2.0,  # FPS.
            images[0].shape[1::-1]  # Size.
        )
    )

    # Write frames to video.
    for image in images:
        video_writer.write(image)

    # Cleanup.
    video_writer.release()
    cv2.destroyAllWindows()


@multimethod
def plot_line_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_line_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='line_utilization',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_line_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit='W',
        horizontal_line_value=None
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax
    if horizontal_line_value is not None:
        vmin = min(vmin, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)
        vmax = max(vmax, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)

    # Obtain values for plotting.
    if isinstance(grid_graph, ElectricGridGraph):
        # Take only lines & mean across all phases.
        values = (
            value_vector.loc[grid_model.lines].mean(level='branch_name')
        )
    else:
        values = value_vector

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename.
    if label is not None:
        title = f"Line utilization: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"line_utilization_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = f"Line utilization"
        filename = "line_utilization.png"
    y_label = f"Utilization [{value_unit}]"

    # Create plot.
    plt.figure()
    plt.title(title)
    plt.bar(
        range(len(grid_model.line_names)),
        values
    )
    if horizontal_line_value is not None:
        plt.hlines(horizontal_line_value, -0.5, len(grid_model.line_names) - 0.5, colors='red')
    plt.xticks(
        range(len(grid_model.line_names)),
        grid_model.line_names,
        rotation=45,
        ha='right'
    )
    plt.ylim([vmin, vmax])
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    # plt.show()
    plt.close()


def wrapper_plot_line_utilization(*args, **kwargs):

    plot_line_utilization(*args, **kwargs)


@multimethod
def plot_transformer_utilization(
        grid_model: mesmo.electric_grid_models.ElectricGridModel,
        grid_graph: ElectricGridGraph,
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_transformer_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='transformer_utilization',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_transformer_utilization(
        grid_model: mesmo.electric_grid_models.ElectricGridModel,
        grid_graph: ElectricGridGraph,
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit='W',
        horizontal_line_value=None
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax
    if horizontal_line_value is not None:
        vmin = min(vmin, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)
        vmax = max(vmax, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)

    # Obtain values for plotting.
    if isinstance(grid_graph, ElectricGridGraph):
        # Take only transformers & mean across all phases.
        values = (
            value_vector.loc[grid_model.transformers].mean(level='branch_name')
        )
    else:
        values = value_vector

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename.
    if label is not None:
        title = f"Transformer utilization: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"transformer_utilization_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = f"Transformer utilization"
        filename = "transformer_utilization.png"
    y_label = f"Utilization [{value_unit}]"

    # Create plot.
    plt.figure()
    plt.title(title)
    plt.bar(
        range(len(grid_model.transformer_names)),
        values
    )
    if horizontal_line_value is not None:
        plt.hlines(horizontal_line_value, -0.5, len(grid_model.transformer_names) - 0.5, colors='red')
    plt.xticks(
        range(len(grid_model.transformer_names)),
        grid_model.transformer_names,
        rotation=45,
        ha='right'
    )
    plt.ylim([vmin, vmax])
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    # plt.show()
    plt.close()


def wrapper_plot_transformer_utilization(*args, **kwargs):

    plot_transformer_utilization(*args, **kwargs)


@multimethod
def plot_node_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_node_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='node_voltage' if isinstance(grid_graph, ElectricGridGraph) else 'node_head',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_node_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit=None,
        suffix=None,
        horizontal_line_value=None
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax
    if horizontal_line_value is not None:
        vmin = min(vmin, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)
        vmax = max(vmax, 1.05 * horizontal_line_value, 0.95 * horizontal_line_value)

    # Obtain values for plotting.
    if isinstance(grid_graph, ElectricGridGraph):
        # Take mean across all phases.
        values = (
            value_vector.mean(level='node_name')
        )
    else:
        values = value_vector

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename / unit.
    if isinstance(grid_graph, ElectricGridGraph):
        title = 'Node voltage' + f' {suffix}' if suffix is not None else ''
        filename = 'node_voltage'
        y_label = 'Voltage' + f' {suffix}' if suffix is not None else ''
        value_unit = 'V' if value_unit is None else value_unit
    else:
        title = 'Node head' + f' {suffix}' if suffix is not None else ''
        filename = 'node_head'
        y_label = 'Head' + f' {suffix}' if suffix is not None else ''
        value_unit = 'm' if value_unit is None else value_unit
    if label is not None:
        title = f"{title}: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"{filename}_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = f"{title}"
        filename = f"{filename}.png"

    # Create plot.
    plt.figure()
    plt.title(title)
    plt.bar(
        range(len(grid_model.node_names)),
        values
    )
    if horizontal_line_value is not None:
        plt.hlines(horizontal_line_value, -0.5, len(grid_model.node_names) - 0.5, colors='red')
    plt.xticks(
        range(len(grid_model.node_names)),
        grid_model.node_names,
        rotation=45,
        ha='right'
    )
    plt.ylim([vmin, vmax])
    plt.ylabel(f'{y_label} [{value_unit}]')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    # plt.show()
    plt.close()


def wrapper_plot_node_utilization(*args, **kwargs):

    plot_node_utilization(*args, **kwargs)


@multimethod
def plot_grid_line_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_grid_line_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='grid_line_utilization',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_grid_line_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit='W'
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Obtain edge color values.
    if isinstance(grid_graph, ElectricGridGraph):
        # Take only lines & mean across all phases.
        values = (
            value_vector.loc[grid_model.lines].mean(level='branch_name')
        )
    else:
        values = value_vector

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename.
    if label is not None:
        title = f"Line utilization: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"grid_line_utilization_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = "Line utilization"
        filename = "grid_line_utilization.png"

    # Create plot.
    plt.figure()
    plt.title(title)
    if isinstance(grid_graph, ElectricGridGraph):
        # Highlight transformer nodes.
        nx.draw(
            grid_graph,
            nodelist=grid_graph.transformer_nodes,
            edgelist=[],
            pos=grid_graph.node_positions,
            node_size=100.0,
            node_color='red'
        )
    nx.draw(
        grid_graph,
        edgelist=grid_graph.edge_by_line_name.loc[grid_model.line_names].tolist(),
        pos=grid_graph.node_positions,
        node_size=10.0,
        node_color='black',
        arrows=False,
        width=5.0,
        edge_vmin=vmin,
        edge_vmax=vmax,
        edge_color=values.tolist()
    )
    # Add colorbar.
    sm = (
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=vmin,
                vmax=vmax
            )
        )
    )
    cb = plt.colorbar(sm, shrink=0.9)
    cb.set_label(f'Utilization [{value_unit}]')

    if mesmo.config.config['plots']['add_basemap']:
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add contextual basemap layer for orientation.
        ctx.add_basemap(
            plt.gca(),
            crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
            source=ctx.providers.CartoDB.Positron,
            attribution=mesmo.config.config['plots']['show_basemap_attribution']
        )

    # Store / show / close figure.
    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight')
    # plt.show()
    plt.close()


def wrapper_plot_grid_line_utilization(*args, **kwargs):

    plot_grid_line_utilization(*args, **kwargs)


@multimethod
def plot_grid_transformer_utilization(
        grid_model: mesmo.electric_grid_models.ElectricGridModel,
        grid_graph: ElectricGridGraph,
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_grid_transformer_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='grid_transformer_utilization',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_grid_transformer_utilization(
        grid_model: mesmo.electric_grid_models.ElectricGridModel,
        grid_graph: ElectricGridGraph,
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit='W'
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Obtain edge color values.
    # - Take only transformers & mean across all phases.
    values = (
        value_vector.loc[grid_model.transformers].mean(level='branch_name')
    )

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename.
    if label is not None:
        title = f"Transformer utilization: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"grid_transformer_utilization_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = "Transformer utilization"
        filename = "grid_transformer_utilization.png"

    # Create plot.
    plt.figure()
    plt.title(title)
    # Plot nodes all nodes, but with node size 0.0, just to get appropriate map extent.
    nx.draw(
        grid_graph,
        edgelist=[],
        pos=grid_graph.node_positions,
        node_size=0.0
    )
    nx.draw(
        grid_graph,
        nodelist=grid_graph.transformer_nodes,
        edgelist=[],
        pos=grid_graph.node_positions,
        node_size=200.0,
        node_color=values.tolist(),
        vmin=vmin,
        vmax=vmax,
        edgecolors='black'
    )
    # Add colorbar.
    sm = (
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=vmin,
                vmax=vmax
            )
        )
    )
    cb = plt.colorbar(sm, shrink=0.9)
    cb.set_label(f'Utilization [{value_unit}]')

    if mesmo.config.config['plots']['add_basemap']:
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add contextual basemap layer for orientation.
        ctx.add_basemap(
            plt.gca(),
            crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
            source=ctx.providers.CartoDB.Positron,
            attribution=mesmo.config.config['plots']['show_basemap_attribution']
        )

    # Store / show / close figure.
    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight')
    # plt.show()
    plt.close()


def wrapper_plot_grid_transformer_utilization(*args, **kwargs):

    plot_grid_transformer_utilization(*args, **kwargs)


@multimethod
def plot_grid_node_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.DataFrame,
        results_path: str,
        vmin=None,
        vmax=None,
        make_video=False,
        **kwargs
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Create plot for each column in `value_vector`.
    mesmo.utils.starmap(
        wrapper_plot_grid_node_utilization,
        zip(
            itertools.repeat(grid_model),
            itertools.repeat(grid_graph),
            [row[1] for row in value_vector.iterrows()],
            itertools.repeat(results_path)
        ),
        dict(
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    )

    # Stitch images to video.
    if make_video:
        create_video(
            name='grid_node_voltage' if isinstance(grid_graph, ElectricGridGraph) else 'grid_node_head',
            labels=value_vector.index,
            results_path=results_path
        )


@multimethod
def plot_grid_node_utilization(
        grid_model: typing.Union[
            mesmo.electric_grid_models.ElectricGridModel,
            mesmo.thermal_grid_models.ThermalGridModel
        ],
        grid_graph: typing.Union[
            ElectricGridGraph,
            ThermalGridGraph
        ],
        value_vector: pd.Series,
        results_path: str,
        vmin=None,
        vmax=None,
        label=None,
        value_unit=None,
        suffix=None
):

    # Obtain colorscale minimum / maximum value.
    vmin = value_vector.values.ravel().min() if vmin is None else vmin
    vmax = value_vector.values.ravel().max() if vmax is None else vmax

    # Obtain edge color values.
    if isinstance(grid_graph, ElectricGridGraph):
        # Take mean across all phases.
        values = (
            value_vector.mean(level='node_name')
        )
    else:
        values = value_vector

    # Obtain label.
    label = value_vector.name if label is None else label

    # Obtain plot title / filename / unit.
    if isinstance(grid_graph, ElectricGridGraph):
        title = 'Node voltage' + f' {suffix}' if suffix is not None else ''
        filename = 'grid_node_voltage'
        colorbar_label = f'Voltage {suffix}'
        value_unit = 'V' if value_unit is None else value_unit
    else:
        title = 'Node head' + f' {suffix}' if suffix is not None else ''
        filename = 'grid_node_head'
        colorbar_label = 'Head' + f' {suffix}' if suffix is not None else ''
        value_unit = 'm' if value_unit is None else value_unit
    if label is not None:
        title = f"{title}: {label.strftime('%H:%M:%S') if type(label) is pd.Timestamp else label}"
        filename = f"{filename}_{mesmo.utils.get_alphanumeric_string(f'{label}')}.png"
    else:
        title = f"{title}"
        filename = f"{filename}.png"

    # Create plot.
    plt.figure()
    plt.title(title)
    if isinstance(grid_graph, ElectricGridGraph):
        # Highlight transformer nodes.
        nx.draw(
            grid_graph,
            nodelist=grid_graph.transformer_nodes,
            edgelist=[],
            pos=grid_graph.node_positions,
            node_size=100.0,
            node_color='red'
        )
    nx.draw(
        grid_graph,
        nodelist=grid_model.node_names.tolist(),
        pos=grid_graph.node_positions,
        node_size=50.0,
        arrows=False,
        vmin=vmin,
        vmax=vmax,
        node_color=values.tolist(),
        edgecolors='black',
    )
    # Add colorbar.
    sm = (
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=vmin,
                vmax=vmax
            )
        )
    )
    cb = plt.colorbar(sm, shrink=0.9)
    cb.set_label(f'{colorbar_label} [{value_unit}]')

    if mesmo.config.config['plots']['add_basemap']:
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add contextual basemap layer for orientation.
        ctx.add_basemap(
            plt.gca(),
            crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
            source=ctx.providers.CartoDB.Positron,
            attribution=mesmo.config.config['plots']['show_basemap_attribution']
        )

    # Store / show / close figure.
    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight')
    # plt.show()
    plt.close()


def wrapper_plot_grid_node_utilization(*args, **kwargs):

    plot_grid_node_utilization(*args, **kwargs)


def plot_total_active_power(
        values_dict: dict,
        results_path: str
):

    # Pre-process values.
    for key in values_dict:
        values_dict[key] = values_dict[key].sum(axis='columns') / 1e6
        values_dict[key].loc[:] = np.abs(np.real(values_dict[key]))

    # Obtain plot title / labels / filename.
    title = 'Total active power'
    filename = 'total_active_power_timeseries'
    y_label = 'Active power'
    value_unit = 'MW'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            fill='tozeroy',
            line=go.scatter.Line(shape='hv')
        ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{mesmo.config.config['plots']['file_format']}"))


def plot_line_utilization_histogram(
    values_dict: dict,
    results_path: str,
    histogram_minimum=0.0,
    histogram_maximum=1.0,
    histogram_bin_count=100
):

    # Obtain histogram bins.
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)

    # Pre-process values.
    for key in values_dict:
        # Obtain maximum utilization for all lines.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'line'].max()
        )
        # Set over-utilized lines to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > histogram_maximum] = histogram_maximum
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins)
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = 'Lines'
    filename = 'line_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Bar(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key
        ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{mesmo.config.config['plots']['file_format']}"))


def plot_line_utilization_histogram_cumulative(
        values_dict: dict,
        results_path: str,
        histogram_minimum=0.0,
        histogram_maximum=1.0,
        histogram_bin_count=100
):

    # Obtain histogram bins.
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)

    # Pre-process values.
    for key in values_dict:
        # Obtain maximum utilization for all lines.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'line'].max()
        )
        # Set over-utilized lines to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > histogram_maximum] = histogram_maximum
        # Obtain cumulative histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins).cumsum()
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = 'Lines'
    filename = 'line_utilization_histogram_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            line=go.scatter.Line(shape='hv')
        ))
    # Add horizontal line at 90%.
    figure.add_shape(go.layout.Shape(
        x0=0,
        x1=1,
        xref='paper',
        y0=0.9,
        y1=0.9,
        yref='y',
        type='line',
        line=go.layout.shape.Line(width=2)
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Cumulative proportion',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{mesmo.config.config['plots']['file_format']}"))


def plot_transformer_utilization_histogram(
        values_dict: dict,
        results_path: str,
        selected_columns=None,
        histogram_minimum=0.0,
        histogram_maximum=1.0,
        histogram_bin_count=100
):

    # Obtain histogram bins.
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)

    # Pre-process values.
    for key in values_dict:
        # Only use selected columns.
        values_dict[key] = (
            values_dict[key].loc[:, selected_columns] if selected_columns is not None else values_dict[key]
        )
        # Obtain maximum utilization for all transformers.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'transformer'].max()
        )
        # Set over-utilized transformers to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > histogram_maximum] = histogram_maximum
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins)
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Bar(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key
        ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{mesmo.config.config['plots']['file_format']}"))


def plot_transformer_utilization_histogram_cumulative(
        values_dict: dict,
        results_path: str,
        selected_columns=None,
        histogram_minimum=0.0,
        histogram_maximum=1.0,
        histogram_bin_count=100
):

    # Obtain histogram bins.
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)

    # Pre-process values.
    for key in values_dict:
        # Only use selected columns.
        values_dict[key] = (
            values_dict[key].loc[:, selected_columns] if selected_columns is not None else values_dict[key]
        )
        # Obtain maximum utilization for all transformers.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'transformer'].max()
        )
        # Set over-utilized transformers to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > histogram_maximum] = histogram_maximum
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins).cumsum()
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            line=go.scatter.Line(shape='hv')
        ))
    # Add horizontal line at 90%.
    figure.add_shape(go.layout.Shape(
        x0=0,
        x1=1,
        xref='paper',
        y0=0.9,
        y1=0.9,
        yref='y',
        type='line',
        line=go.layout.shape.Line(width=2)
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Cumulative proportion',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{mesmo.config.config['plots']['file_format']}"))


def plot_histogram_cumulative_branch_utilization(
        results_dict: mesmo.problems.ResultsDict,
        results_path: str,
        branch_type: str = 'line',
        filename_base: str = 'branch_utilization_',
        filename_suffix: str = '',
        plot_title: str = None,
        histogram_minimum: float = 0.0,
        histogram_maximum: float = 1.0,
        histogram_bin_count: int = 100,
        vertical_line: float = None,
        horizontal_line: float = None,
        x_tick_interval: float = 0.1
):

    # Obtain histogram bins.
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)

    # Pre-process values.
    values_dict = dict.fromkeys(results_dict.keys())
    box_values_dict = dict.fromkeys(results_dict.keys())
    for key in values_dict:
        # Obtain branch power in p.u. values.
        values_dict[key] = (
            (
                results_dict[key].branch_power_magnitude_vector_1_per_unit
                + results_dict[key].branch_power_magnitude_vector_2_per_unit
            ) / 2
        )
        # Select branch type.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == branch_type]
        )
        # Obtain maximum utilization.
        values_dict[key] = values_dict[key].max()
        # Keep these values for boxplot.
        box_values_dict[key] = values_dict[key]
        # Obtain cumulative histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins).cumsum()
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = None
    filename = f'{filename_base}{branch_type}{filename_suffix}'
    value_label = f'Maximum {branch_type} utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = (
        plotly.subplots.make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.2, 0.8],
            vertical_spacing=0.0,
            shared_xaxes=True
        )
    )
    for key in box_values_dict:
        figure.add_trace(go.Box(
            x=box_values_dict[key].values,
            boxmean=True,
            name=key,
            showlegend=False
        ), row=1, col=1)
    for index, key in enumerate(values_dict):
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            line=go.scatter.Line(shape='hv', color=plotly.colors.qualitative.D3[index])
        ), row=2, col=1)
    # Add vertical line.
    if vertical_line is not None:
        figure.add_shape(go.layout.Shape(
            x0=vertical_line,
            x1=vertical_line,
            xref='x2',
            y0=0.0,
            y1=1.0,
            yref='paper',
            type='line',
            line=go.layout.shape.Line(width=2)
        ))
        for trace in figure.data:
            if type(trace) is go.Scatter:
                key = trace['name']
                value = np.interp(vertical_line, values_dict[key].index, values_dict[key].values)
                figure.add_shape(go.layout.Shape(
                    x0=histogram_minimum,
                    x1=vertical_line,
                    xref='x2',
                    y0=value,
                    y1=value,
                    yref='y2',
                    type='line',
                    line=go.layout.shape.Line(width=2, color=trace['line']['color']),
                    layer='below'
                ))
    # Add horizontal line.
    if horizontal_line is not None:
        figure.add_shape(go.layout.Shape(
            x0=0.0,
            x1=1.0,
            xref='paper',
            y0=horizontal_line,
            y1=horizontal_line,
            yref='y2',
            type='line',
            line=go.layout.shape.Line(width=2)
        ))
        for trace in figure.data:
            if type(trace) is go.Scatter:
                key = trace['name']
                value = np.interp(horizontal_line, values_dict[key].values, values_dict[key].index)
                figure.add_shape(go.layout.Shape(
                    x0=value,
                    x1=value,
                    xref='x2',
                    y0=0.0,
                    y1=horizontal_line,
                    yref='y2',
                    type='line',
                    line=go.layout.shape.Line(width=2, color=trace['line']['color']),
                    layer='below'
                ))
    figure.update_layout(
        title=title,
        yaxis1_showticklabels=False,
        xaxis1_side='top',
        xaxis1_dtick=x_tick_interval,
        xaxis1_showticklabels=True,
        xaxis2_range=[histogram_minimum, histogram_maximum],
        xaxis2_dtick=x_tick_interval,
        xaxis2_title=f'{value_label} [{value_unit}]',
        yaxis2_dtick=0.1,
        yaxis2_title='Cumulative proportion',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.05, yanchor='auto')
    )
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))


def plot_histogram_node_utilization(
        results_dict: mesmo.problems.ResultsDict,
        results_path: str,
        filename_base: str = 'node_utilization',
        filename_suffix: str = '',
        plot_title: str = None,
        histogram_bin_count: int = 30,
        x_tick_interval: float = None
):

    # Pre-process values.
    values_dict = dict.fromkeys(results_dict.keys())
    box_values_dict = dict.fromkeys(results_dict.keys())
    for key in values_dict:
        # Obtain node voltage in p.u. values.
        values_dict[key] = (
            results_dict[key].node_voltage_magnitude_vector_per_unit
        )
        # Obtain maximum voltage drop.
        values_dict[key] = 1.0 - values_dict[key].min()
        # Keep these values for boxplot.
        box_values_dict[key] = values_dict[key]
    # Obtain histogram bins.
    histogram_maximum = pd.DataFrame(values_dict).max().max()
    histogram_minimum = pd.DataFrame(values_dict).min().min()
    histogram_interval = (histogram_maximum - histogram_minimum) / histogram_bin_count
    histogram_bins = np.arange(histogram_minimum, histogram_maximum + histogram_interval, histogram_interval)
    # Obtain cumulative histogram values.
    for key in values_dict:
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins)
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = plot_title
    filename = f'{filename_base}{filename_suffix}'
    value_label = 'Maximum voltage drop'
    value_unit = 'p.u.'

    # Create plot.
    figure = (
        plotly.subplots.make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.2, 0.8],
            vertical_spacing=0.0,
            shared_xaxes=True
        )
    )
    for key in box_values_dict:
        figure.add_trace(go.Box(
            x=box_values_dict[key].values,
            boxmean=True,
            name=key,
            showlegend=False
        ), row=1, col=1)
    for index, key in enumerate(values_dict):
        figure.add_trace(go.Bar(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            marker=go.bar.Marker(color=plotly.colors.qualitative.D3[index])
        ), row=2, col=1)
    figure.update_layout(
        title=title,
        yaxis1_showticklabels=False,
        xaxis1_side='top',
        xaxis1_dtick=x_tick_interval,
        xaxis1_showticklabels=True,
        xaxis2_title=f'{value_label} [{value_unit}]',
        xaxis2_dtick=x_tick_interval,
        yaxis2_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.05, yanchor='auto')
    )
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))


def plot_aggregate_timeseries_der_power(
        results_dict: mesmo.problems.ResultsDict,
        results_path: str,
        filename_base: str = 'der_power',
        filename_suffix: str = '',
        plot_title: str = None,
        value_factor: float = -1.0,
        value_unit_label: str = None,
        der_type_labels: dict = None
):

    # Pre-process values.
    values_dict = dict.fromkeys(results_dict.keys())
    value_minimum = 0.0
    value_maximum = 0.0
    for key in values_dict:
        # Obtain values.
        # TODO: Multiply by base power value. (Needs to be added to results object first.)
        values_dict[key] = results_dict[key].der_active_power_vector
        values_dict[key] = value_factor * values_dict[key].groupby('der_type', axis='columns').sum()
        # Obtain value range.
        value_minimum = min(value_minimum, values_dict[key].sum(axis='columns').min())
        value_maximum = max(value_maximum, values_dict[key].sum(axis='columns').max())

    # Obtain plot title / labels / filename.
    title = plot_title
    filename = f'{filename_base}{filename_suffix}'
    value_label = 'Active power'
    value_unit = f' [{value_unit_label}]' if value_unit_label is not None else ''

    # Create plot.
    figure = (
        plotly.subplots.make_subplots(
            rows=len(values_dict),
            cols=1,
            vertical_spacing=0.08,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=tuple(values_dict.keys()),
            y_title=f'{value_label}{value_unit}'
        )
    )
    legend_values = list()
    for index, key in enumerate(values_dict):
        for column in values_dict[key].columns:
            if column not in legend_values:
                legend_values.append(column)
                showlegend = True
            else:
                showlegend = False
            color_index = legend_values.index(column)
            figure.add_trace(go.Scatter(
                x=values_dict[key].index,
                y=values_dict[key].loc[:, column].values,
                name=der_type_labels[column] if der_type_labels is not None else column,
                showlegend=showlegend,
                line=go.scatter.Line(shape='hv', color=plotly.colors.qualitative.D3[color_index]),
                fill='tozeroy' if column == values_dict[key].columns[0] else 'tonexty',
                stackgroup='one'
            ), row=index + 1, col=1)
    figure.update_layout(
        title=title,
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=1.05, yanchor='auto'),
        height=mesmo.config.config['plots']['plotly_figure_height'] * 1.25,
        **{f'xaxis{len(values_dict)}_tickformat': '%H:%M'},
        **{f'yaxis{index}_range': [value_minimum, value_maximum] for index in range(1, len(values_dict) + 1)}
    )
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))
