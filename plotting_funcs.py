def fix_plotly_grid_lines(fig, y_axis = True, x_axis = True):
    if x_axis:
        fig.update_xaxes(minor=dict(ticklen=6, tickcolor="black", showgrid=True))
        fig.update_xaxes(gridcolor='black', zeroline = True, zerolinecolor='black', rangemode = 'tozero')
    if y_axis:
        fig.update_yaxes(gridcolor='black', minor_griddash="dot")
        fig.update_yaxes(zeroline = True, zerolinecolor='black')
    return fig