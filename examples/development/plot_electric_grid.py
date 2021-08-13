"""Script to generate an network plot of the electric grid for given scenario.

- This script depends on `contextily`, which is not included in the package dependencies, but can be installed
  under Anaconda via `conda install -c conda-forge contextily`.
"""

import contextily as ctx
import numpy as np
import os
import pandas as pd
import PIL.Image
import plotly.express as px
import plotly.graph_objects as go
import rasterio.warp

import mesmo


def main():

    # Settings.
    scenario_name = 'singapore_all'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain electric grid data / graph.
    electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)
    electric_grid_graph = mesmo.plots.ElectricGridGraph(scenario_name)

    # Obtain substation nodes.
    # - This identification is based on the assumption that all nodes with no DER connected are substation nodes,
    #   which is true for the Singapore synthetic grid test case.
    nodes_substation = (
        electric_grid_data.electric_grid_transformers.loc[
            electric_grid_data.electric_grid_transformers.loc[:, 'transformer_name'].str.contains('66kV'),
            'node_2_name'
        ].values
    )

    # Obtain values.
    nodes_values = (
        pd.DataFrame(electric_grid_graph.node_positions).T.rename({0: 'longitude', 1: 'latitude'}, axis='columns')
    )
    lines = pd.DataFrame(electric_grid_graph.edges)
    lines_values = pd.DataFrame(index=range(3 * len(lines)), columns=['longitude', 'latitude'])
    lines_values.loc[range(0, 3 * len(lines), 3), :] = nodes_values.reindex(lines.iloc[:, 0]).values
    lines_values.loc[range(1, 3 * len(lines), 3), :] = nodes_values.reindex(lines.iloc[:, 1]).values
    lines_values.loc[range(2, 3 * len(lines), 3), :] = np.full((len(lines), 2), np.nan)

    # Obtain latitude / longitude bounds.
    longitude_max = nodes_values.loc[:, 'longitude'].max()
    longitude_min = nodes_values.loc[:, 'longitude'].min()
    latitude_max = nodes_values.loc[:, 'latitude'].max()
    latitude_min = nodes_values.loc[:, 'latitude'].min()

    # Obtain zoom / center for interactive plot.
    zoom_longitude = np.log2(360 * 2.0 / (longitude_max - longitude_min))
    zoom_latitude = np.log2(360 * 2.0 / (latitude_max - latitude_min))
    zoom_correction = 1.035
    zoom = zoom_correction * np.min([zoom_longitude, zoom_latitude])
    center = dict(lon=0.5 * (longitude_max + longitude_min), lat=0.5 * (latitude_max + latitude_min))

    # Obtain background map image for static plot.
    image_bounds = (longitude_min, latitude_min, longitude_max, latitude_max)
    image, image_bounds = (
        ctx.bounds2img(
            *image_bounds,
            zoom=14,
            source=ctx.providers.CartoDB.Positron,
            ll=True
        )
    )
    # Reorder image bounds, because it's jumbled up in bounds2img. # TODO: Raise issue.
    image_bounds = (image_bounds[0], image_bounds[2], image_bounds[1], image_bounds[3])
    image_bounds = (
        rasterio.warp.transform_bounds(
            {'init': 'epsg:3857'},
            {'init': 'epsg:4326'},
            *image_bounds
        )
    )
    width, height = image.shape[1], image.shape[0]
    image = PIL.Image.fromarray(image)

    # Create static plot.
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=lines_values.loc[:, 'longitude'],
        y=lines_values.loc[:, 'latitude'],
        line=dict(color='black', width=0.25),
        hoverinfo='none',
        mode='lines'
    ))
    figure.add_trace(go.Scatter(
        x=nodes_values.loc[:, 'longitude'],
        y=nodes_values.loc[:, 'latitude'],
        text=list(electric_grid_graph.nodes),
        mode='markers+text',
        hoverinfo='none',
        marker=dict(color='lightskyblue', size=4),
        textfont=dict(size=1)
    ))
    figure.add_trace(go.Scatter(
        x=nodes_values.loc[nodes_substation, 'longitude'],
        y=nodes_values.loc[nodes_substation, 'latitude'],
        text=nodes_substation,
        mode='markers+text',
        hoverinfo='none',
        marker=dict(color='lightpink', size=4),
        textfont=dict(size=1)
    ))
    figure.add_layout_image(
        dict(
            source=image,
            xref='x',
            yref='y',
            x=image_bounds[0],
            y=image_bounds[1],
            sizex=image_bounds[2] - image_bounds[0],
            sizey=image_bounds[3] - image_bounds[1],
            xanchor='left',
            yanchor='bottom',
            sizing='stretch',
            opacity=1.0,
            layer='below'
        )
    )
    figure.update(layout=go.Layout(
        width=width, height=height,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=go.layout.XAxis(
            showgrid=False, zeroline=False, showticklabels=False, ticks='',
            range=(image_bounds[0], image_bounds[2])
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, ticks='',
            scaleanchor='x', scaleratio=1,
            range=(image_bounds[1], image_bounds[3])
        )
    ))
    figure.write_image(os.path.join(results_path, 'electric_grid.pdf'))
    mesmo.utils.launch(os.path.join(results_path, 'electric_grid.pdf'))

    # Create interactive plot.
    figure = go.Figure()
    figure.add_trace(go.Scattermapbox(
        lon=lines_values.loc[:, 'longitude'],
        lat=lines_values.loc[:, 'latitude'],
        line=dict(color='black', width=0.5),
        hoverinfo='none',
        mode='lines'
    ))
    figure.add_trace(go.Scattermapbox(
        lon=nodes_values.loc[:, 'longitude'],
        lat=nodes_values.loc[:, 'latitude'],
        text=('Node: ' + np.array(electric_grid_graph.nodes, dtype=object)),
        mode='markers',
        hoverinfo='text',
        marker=dict(color='royalblue')
    ))
    figure.add_trace(go.Scattermapbox(
        lon=nodes_values.loc[nodes_substation, 'longitude'],
        lat=nodes_values.loc[nodes_substation, 'latitude'],
        text=('Substation: ' + nodes_substation),
        mode='markers',
        hoverinfo='text',
        marker=dict(color='crimson')
    ))
    figure.update(layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        mapbox=go.layout.Mapbox(style='carto-positron', zoom=zoom, center=center),
        xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks='')
    ))
    figure.write_html(os.path.join(results_path, 'electric_grid.html'))
    mesmo.utils.launch(os.path.join(results_path, 'electric_grid.html'))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
