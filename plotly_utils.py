
import pandas as pd
import plotly.express as px


def create_time_line_plot(data, x, group_column, y, agg_func, freq, title, y_label, x_label='Date'):
    """
    Generalized function to create a line plot with grouping and aggregation.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        date_column (str): The column containing date values.
        group_column (str): The column to group by (e.g., 'hotel').
        agg_column (str): The column to aggregate (e.g., 'children_count').
        agg_func (str): The aggregation function (e.g., 'sum', 'mean').
        freq (str): The frequency for grouping (e.g., 'W' for weekly, 'M' for monthly).
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        x_label (str): The label for the x-axis (default is 'Date').

    Returns:
        None: Displays the plot.
    """
    # Ensure the date column is in datetime format
    data[x] = pd.to_datetime(data[x])

    # Group and aggregate the data
    grouped_data = data.groupby(
        [pd.Grouper(key=x, freq=freq), group_column]
    )[y].agg(agg_func).reset_index(name=y)

    # Create the line plot
    fig = px.line(
        grouped_data,
        x=x,
        y=y,
        color=group_column,
        title=title,
        labels={y: y_label, x: x_label}
    )

    # Show the plot
    fig.show()


def create_children_rate_plot(data, x, group_column, y, freq, title, y_label, x_label='Date'):
    """
    Create a line plot showing the children rate over time.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        date_column (str): The column containing date values.
        group_column (str): The column to group by (e.g., 'hotel').
        children_column (str): The column indicating the presence of children (e.g., 'has_children').
        freq (str): The frequency for grouping (e.g., 'W' for weekly, 'M' for monthly).
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        x_label (str): The label for the x-axis (default is 'Date').

    Returns:
        None: Displays the plot.
    """
    # Ensure the date column is in datetime format
    data[x] = pd.to_datetime(data[x])

    # Group and calculate total bookings and bookings with children
    grouped_data = data.groupby(
        [pd.Grouper(key=x, freq=freq), group_column]
    ).agg(
        total_bookings=('hotel', 'size'),
        children_bookings=(y, 'sum')
    ).reset_index()

    # Calculate the children rate
    grouped_data['children_rate'] = grouped_data['children_bookings'] / grouped_data['total_bookings']

    # Create the line plot
    fig = px.line(
        grouped_data,
        x=x,
        y='children_rate',
        color=group_column,
        title=title,
        labels={'children_rate': y_label, x: x_label}
    )

    # Show the plot
    fig.show()