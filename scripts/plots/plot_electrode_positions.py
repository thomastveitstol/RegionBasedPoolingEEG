"""
Script for plotting electrode positions
"""
from matplotlib import pyplot

from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem, Reduced1CleanedChildChannelSystem
from src.data.line_separation.utils import project_head_shape, plot_2d_projection


def main() -> None:
    plot_3d = False

    if plot_3d:
        CleanedChildChannelSystem().plot_electrode_positions()
        Reduced1CleanedChildChannelSystem().plot_electrode_positions()
    else:
        plot_2d_projection(project_head_shape(CleanedChildChannelSystem().get_electrode_positions()))
        plot_2d_projection(project_head_shape(Reduced1CleanedChildChannelSystem().get_electrode_positions()))

    pyplot.show()


if __name__ == "__main__":
    main()
