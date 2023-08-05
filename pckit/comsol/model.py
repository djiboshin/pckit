import mph
from abc import ABCMeta
from typing import Any, AnyStr, Dict, Sequence

import pandas as pd
import numpy as np

from ..models import Model
from .._typevars import Task


def make_unique(labels: Sequence) -> list:
    """Gets labels list and makes all labels unique by adding '(number of inclusion)' postfix

    :param labels: List of labels
    :return: List of renamed labels
    """
    new_labels = []
    _, real_index, counts = np.unique(labels, return_counts=True, return_inverse=True)
    for index in range(len(labels)):
        for count in range(counts[real_index[index]]):
            new_label = labels[index] + (f'({count})' if count != 0 else '')
            if new_label not in new_labels:
                new_labels.append(new_label)
                break

    return new_labels


class ComsolModel(mph.Model, Model, metaclass=ABCMeta):
    def __init__(self):
        super().__init__(None)

    def add_circle(
            self,
            name: str,
            x_i: float,
            y_j: float,
            geometry: mph.Node,
            r: float,
            alpha: float = 1.1
    ):
        node = geometry.create("Circle", name=name)
        node.property("r", str(r))
        node.property("pos", [str(x_i), str(y_j)])

        node_sel = (self/'selections').create('Box', name=name)
        node_sel.property('entitydim', 2)

        modified_r = r * alpha
        node_sel.property('xmin', f'+{x_i}-{modified_r}')
        node_sel.property('xmax', f'+{x_i}+{modified_r}')
        node_sel.property('ymin', f'+{y_j}-{modified_r}')
        node_sel.property('ymax', f'+{y_j}+{modified_r}')
        node_sel.property('condition', 'inside')

        return node, node_sel

    def add_square(
            self, name: str, x_i: float, y_j: float, geometry: mph.Node, width: float, alpha: float = 1.1):
        node = geometry.create("Square", name=name)
        node.property("size", str(width))
        node.property("base", "center")
        node.property("pos", [str(x_i), str(y_j)])

        node_sel = (self/'selections').create('Box', name=name)
        node_sel.property('entitydim', 2)

        modified_width = width / 2 * alpha
        node_sel.property('xmin', f'+{x_i}-{modified_width}')
        node_sel.property('xmax', f'+{x_i}+{modified_width}')
        node_sel.property('ymin', f'+{y_j}-{modified_width}')
        node_sel.property('ymax', f'+{y_j}+{modified_width}')
        node_sel.property('condition', 'inside')

        return node, node_sel

    def add_cylinder(self, name: str,
                     x_i: float, y_j: float, z_k: float,
                     geometry: mph.Node,
                     h: float,
                     r: float, alpha: float = 1.1):

        node = geometry.create("Cylinder", name=name)
        node.property("r", str(r))
        node.property("pos", [str(x_i), str(y_j), str(z_k)])
        node.property("h", str(h))

        node_sel = (self/'selections').create('Box', name=name)
        node_sel.property('entitydim', 3)

        modified_r = r * alpha
        modified_h = h * alpha
        node_sel.property('xmin', f'+{x_i}-{modified_r}')
        node_sel.property('xmax', f'+{x_i}+{modified_r}')
        node_sel.property('ymin', f'+{y_j}-{modified_r}')
        node_sel.property('ymax', f'+{y_j}+{modified_r}')
        node_sel.property('zmin', f'+{z_k}-{modified_h - h}')
        node_sel.property('zmax', f'+{z_k}+{modified_h}')
        node_sel.property('condition', 'inside')

        return node, node_sel

    @staticmethod
    def _clean(node: mph.Node, tag: str):
        for c in node.children():
            if tag in c.path[-1]:
                c.remove()

    def clean_geometry(self, geometry: mph.Node, tag):
        self._clean(geometry, tag)
        self._clean(self/'selections', tag)
        self.build(geometry)

    @staticmethod
    def global_evaluation(dataset: mph.Node, evaluation: mph.Node) -> pd.DataFrame:
        #  https://github.com/MPh-py/MPh/blob/2b967b77352f9ce7effcd50ad4774bf5eaf731ea/mph/model.py#L425
        evaluation.property('data', dataset)
        java = evaluation.java
        real, imag = java.computeResult()
        results = (np.array(real) + 1j * np.array(imag)) if imag is not None else np.array(real)
        return pd.DataFrame(data=results, columns=make_unique(evaluation.property('descr')))

    def clear(self):
        # super().clear()
        super().reset()

    def export_image(self, source: mph.Node, filepath: AnyStr, props: Dict = None):
        exports = self / 'exports'
        image = exports.create('Image')
        default_props = {
            'size': 'manualweb',
            'unit': 'px',
            'height': '720',
            'width': '720'
        }
        for prop in default_props:
            image.property(prop, default_props[prop])
        if props is not None:
            for prop in props:
                image.property(prop, props[prop])

        image.property('sourceobject', source)
        image.property('filename', filepath)

        self.export()
        image.remove()

    def plot2d(self, expr: AnyStr, filepath: AnyStr, props: Dict = None):
        plots = self / 'plots'
        plots.java.setOnlyPlotWhenRequested(True)
        plot = plots.create('PlotGroup2D')

        surface = plot.create('Surface', name='plot2d')
        surface.property('resolution', 'normal')
        surface.property('expr', expr)

        self.export_image(plot, filepath, props)
        plot.remove()

    def getLastComputationTime(self):
        # TODO ADD ANY STUDY SUPPORT

        studies = (self / 'studies').children()
        return -1 if not len(studies) else int(studies[-1].java.getLastComputationTime())

    def configure(self) -> Any:
        pass

    def pre_build(self, task: Task):
        pass

    def pre_solve(self, task: Task):
        pass

    def pre_clear(self, task: Task):
        pass
