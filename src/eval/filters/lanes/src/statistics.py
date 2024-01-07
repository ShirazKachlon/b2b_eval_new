import pandas as pd


def _plot_decision_percentage(decision_perc: pd.Series):
    import matplotlib.pyplot as plt

    decision_perc.plot(kind='pie', title='Decision percentage',
                       figsize=(6, 6), autopct='%1.1f%%', startangle=90)
    plt.show()


def calculate_lanes_statistics(detections: pd.DataFrame, plot=True):
    """Calculate statistics for lanes"""

    detections.loc[detections.is_outside_boundaries.isna(),
                   "is_outside_boundaries"] = "None"
    total_applied_on = detections.is_outside_boundaries.value_counts().sum()
    decision_perc = detections.is_outside_boundaries.value_counts() / total_applied_on

    # Plot the results
    if plot:
        _plot_decision_percentage(decision_perc)
