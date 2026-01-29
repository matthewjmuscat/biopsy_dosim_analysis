from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import copy
import seaborn as sns
import numpy as np
import pandas as pd 
from pathlib import Path
from statsmodels.nonparametric.kernel_regression import KernelReg
import misc_tools
import plotly.express as px
import plotting_funcs
import kaleido # imported for exporting image files, although not referenced it is required
import math
import warnings
import random
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, truncnorm, lognorm, gamma, weibull_min, expon, pareto, rice, gengamma, kstest, gennorm, skewnorm
import os
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from typing import Iterable, Optional, Tuple, Literal, Sequence, Dict
import statsmodels.api as sm
from scipy.stats import spearmanr
import re 
from matplotlib.ticker import StrMethodFormatter



# for high-quality vector graphics output
mpl.rcParams["pdf.fonttype"] = 42   # TrueType fonts in PDF
mpl.rcParams["ps.fonttype"] = 42

# Global font / math settings for all figures in this module
MPL_FONT_RC = {
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.unicode_minus": True,
}

# Keep a clean, white canvas with black axes (match GPR_production_plots style)
MPL_FACE_RC = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

mpl.rcParams.update(MPL_FONT_RC | MPL_FACE_RC)

# (Optional) make seaborn respect these too
sns.set_theme(style="white", rc=MPL_FONT_RC | MPL_FACE_RC)

# Shared colors for consistent styling across production figures
PRIMARY_LINE_COLOR = "#0b3b8a"  # deep blue
OVERLAY_LINE_COLOR = "#4a4a4a"   # neutral gray
GRID_COLOR = "#b8b8b8"



plt.ioff()

def production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                      dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                                                                                      dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                                                                                 patientUID,
                                                                                 bx_struct_roi,
                                                                                 bx_struct_ind,
                                                                                 bx_ref,
                                                                                 value_col_key,
                                                                                 patient_sp_output_figures_dir,
                                                                                 general_plot_name_string,
                                                                                 num_rand_trials_to_show,
                                                                                 y_axis_label,
                                                                                 custom_fig_title,
                                                                                 trial_annotation_style = 'number'     # ← new: 'arrow' or 'number'
                                                                                 ):
    # plotting function
    def plot_quantile_regression_and_more_corrected(df, 
                                                    df_voxelized, 
                                                    sp_patient_all_structure_shifts_pandas_data_frame, 
                                                    patientUID, 
                                                    bx_id, 
                                                    bx_struct_ind, 
                                                    bx_ref, 
                                                    trial_annotation_style=trial_annotation_style,       # ← new: 'arrow' or 'number'
                                                    linestyle_regressions='-',
                                                    line_width_regressions=2,
                                                    linestyle_regressions_trials='--',
                                                    line_width_regressions_trials=1,
                                                    linestyle_quantiles=':',
                                                    line_width_quantiles=1
                                                    ):
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Placeholder dictionaries for regression results
        y_regressions = {}

        # Function to perform and plot kernel regression
        def perform_and_plot_kernel_regression(x, y, x_range, label, color, annotation_text = None, target_offset=0 ,linestyle='-', linewidth=2):
            kr = KernelReg(endog=y, exog=x, var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            plotted_line, = plt.plot(x_range, y_kr, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
            
            """
            if annotation_text != None:
                plt.annotate(annotation_text, xy=(x_range[0], y_kr[0]), xytext=(x_range[0], y_kr[0]))
            """
            # Add annotation if provided
            if annotation_text is not None:
                # Determine the total number of points
                total_points = len(x_range)
                
                # Calculate the target index based on the offset, with wrapping
                target_index = (total_points // 5 * target_offset) % total_points
                
                # Target point coordinates
                target_x = x_range[target_index]
                target_y = y_kr[target_index]
                
                # Add annotation with an arrow
                plt.annotate(
                    annotation_text, 
                    xy=(target_x, target_y),  # Point to annotate
                    xytext=(target_x + 1, target_y + 1),  # Offset for text
                    arrowprops=dict(
                        arrowstyle="->",  # Arrow style
                        color=color,      # Arrow color
                        lw=1.5            # Line width
                    ),
                    fontsize=10,        # Font size of annotation
                    color=color,        # Color of annotation text
                    bbox=dict(
                        boxstyle="round,pad=0.3",  # Text box style
                        edgecolor=color,          # Edge color of box
                        facecolor="white",        # Background color of box
                        alpha=0.8                 # Transparency of box
                    )
                )

            return y_kr, plotted_line


        # Perform kernel regression for each quantile and store the y-values
        for quantile in [0.05, 0.25, 0.75, 0.95]:
            q_df = df.groupby('Z (Bx frame)')[value_col_key].quantile(quantile).reset_index()
            kr = KernelReg(endog=q_df[value_col_key], exog=q_df['Z (Bx frame)'], var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            y_regressions[quantile] = y_kr

        # Filling the space between the quantile lines
        fill_1 = plt.fill_between(x_range, y_regressions[0.05], y_regressions[0.25], color='springgreen', alpha=1)
        fill_2 = plt.fill_between(x_range, y_regressions[0.25], y_regressions[0.75], color='dodgerblue', alpha=1)
        fill_3 = plt.fill_between(x_range, y_regressions[0.75], y_regressions[0.95], color='springgreen', alpha=1)
        
        plt.plot(x_range, y_regressions[0.05], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.25], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.75], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.95], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')

        # Additional plot enhancements
        # Plot line for MC trial num = 0
        # Kernel regression for MC trial num = 0 subset
        
        mc_trial_0 = df[df['MC trial num'] == 0]
        _ , nominal_line = perform_and_plot_kernel_regression(mc_trial_0['Z (Bx frame)'], mc_trial_0[value_col_key], x_range, 'Nominal', 'red', linestyle=linestyle_regressions, linewidth=line_width_regressions)
        

        # KDE and mean dose per Original pt index
        kde_max_doses = []
        mean_doses = []
        z_vals = []
        for z_val in df['Z (Bx frame)'].unique():
            pt_data = df[df['Z (Bx frame)'] == z_val]
            kde = gaussian_kde(pt_data[value_col_key])
            kde_doses = np.linspace(pt_data[value_col_key].min(), pt_data[value_col_key].max(), 500)
            max_density_dose = kde_doses[np.argmax(kde(kde_doses))]
            kde_max_doses.append(max_density_dose)
            mean_doses.append(pt_data[value_col_key].mean())
            z_vals.append(z_val)
        
        _ , max_density_line = perform_and_plot_kernel_regression(z_vals, kde_max_doses, x_range, 'KDE Max Density Dose', 'magenta', linestyle=linestyle_regressions, linewidth=line_width_regressions)
        _ , mean_line = perform_and_plot_kernel_regression(z_vals, mean_doses, x_range, 'Mean Dose', 'orange', linestyle=linestyle_regressions, linewidth=line_width_regressions)

        



        ## Instead want to show regressions of random trials so that we can appreciate structure
        # For "arrow" style only
        annotation_offset_index = 0
        # For “number” style only
        annotation_lines = []

        for trial in range(1,num_rand_trials_to_show + 1): # +1 because we start at 1 in range()
            mc_trial_shift_vec_df = sp_patient_all_structure_shifts_pandas_data_frame[(sp_patient_all_structure_shifts_pandas_data_frame["Trial"] == trial) &
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) & 
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)].reset_index(drop=True)
            mc_trial = df[df['MC trial num'] == trial]
            mc_trial_voxelized = df_voxelized[df_voxelized['MC trial num'] == trial]

            

            x_dist = mc_trial_shift_vec_df.at[0,'Shift (X)']
            y_dist = mc_trial_shift_vec_df.at[0,'Shift (Y)']
            z_dist = mc_trial_shift_vec_df.at[0,'Shift (Z)']
            d_tot = math.sqrt(x_dist**2 + y_dist**2 + z_dist**2)


            if trial_annotation_style == 'arrow':

                annotation_text_for_trial = f"({x_dist:.1f},{y_dist:.1f},{z_dist:.1f}), d = {d_tot:.1f} mm"
            
                perform_and_plot_kernel_regression(mc_trial['Z (Bx frame)'], 
                                                mc_trial[value_col_key], 
                                                x_range, 
                                                f"Trial: {trial}", 
                                                'black', 
                                                annotation_text = annotation_text_for_trial, 
                                                target_offset=annotation_offset_index,
                                                linestyle=linestyle_regressions_trials,
                                                linewidth=line_width_regressions_trials)
                
                plt.scatter(
                    mc_trial_voxelized['Z (Bx frame)'], 
                    mc_trial_voxelized[value_col_key], 
                    color='grey', 
                    alpha=0.1, 
                    s=10,  # Size of dots, adjust as needed
                    zorder=1.1
                )

                annotation_offset_index += 1

            
            else: # ‘number’ style

                # 1) Build the per‐trial annotation line
                line_str = f"{trial}: ({x_dist:.1f}, {y_dist:.1f}, {z_dist:.1f}), d = {d_tot:.1f} mm"
                annotation_lines.append(line_str)

                # 2) plot and grab the y-values at x_range
                y_kr, _ = perform_and_plot_kernel_regression(
                            mc_trial['Z (Bx frame)'],
                            mc_trial[value_col_key],
                            x_range,
                            str(trial),      # legend entry is now just the number
                            'black',
                            linestyle=linestyle_regressions_trials,
                            linewidth=line_width_regressions_trials
                            )
            
                plt.scatter(
                    mc_trial_voxelized['Z (Bx frame)'], 
                    mc_trial_voxelized[value_col_key], 
                    color='grey', 
                    alpha=0.1, 
                    s=10,  # Size of dots, adjust as needed
                    zorder=1.1
                )

                # —— 3) label each line end with its number
                plt.text(
                    x_range[-1] + 0.1,    # 0.1 mm further from the right edge
                    y_kr[-1],           # the line’s final y-value
                    str(trial),
                    fontsize=14,
                    color='black',
                    ha='left', va='center'
                )




        ax = plt.gca() 

        # Final plot adjustments
        ax.set_title(f'{patientUID} - {bx_id} - {custom_fig_title}',
             fontsize=16,      # Increase the title font size
             #fontname='Arial' # Set the title font family
            )
        ax.set_xlabel("Biopsy Axial Dimension (mm)",
              fontsize=16,    # sets the font size
              #fontname='Arial'
               )   # sets the font family

        ax.set_ylabel(y_axis_label,
                    fontsize=16,
                    #fontname='Arial'
                    )



        # always supply the same quantile + nominal/density/mean entries —
        # then either “Trial: X” (arrow mode) or simply “1”, “2”, … (number mode)
        legend_handles = [fill_1, fill_2, fill_3] + [nominal_line, max_density_line, mean_line]

        quantile_labels = ['5th–25th Q','25th–75th Q','75th–95th Q']
        main_labels     = ['Nominal','Max Density','Mean']


        leg = ax.legend(legend_handles,
            quantile_labels + main_labels ,
            loc='best', facecolor='white',
            prop={
                'size': 14,        # font size
                #'family': 'serif', # optional: font family
                #'weight': 'bold'   # optional: font weight
            }
        )

        plt.tight_layout(rect=[0.01, 0.01, 1, 1])  # Adjust layout to fit the legend


        fig.canvas.draw()  # needed to compute text/legend positions
        renderer = fig.canvas.get_renderer()

        # 3) Get the frame (background patch) bbox in display coords:
        leg_frame = leg.get_frame()
        bbox_disp = leg_frame.get_window_extent(renderer)
    
        # 4) Convert that to figure coords:
        inv_fig = fig.transFigure.inverted()
        # top-left corner of the frame:
        frame_x0, frame_y0 = inv_fig.transform((bbox_disp.x0, bbox_disp.y0))
    
        # 5) Compute just below the legend frame:
        text_x = frame_x0
        text_y = frame_y0 - 0.02  # 2% of figure height below the legend
    
        # 6) Draw in FIGURE coordinates so it lines up perfectly:
        annotation_text = "\n".join(annotation_lines)
        fig.text(
            text_x, text_y, annotation_text,
            transform=fig.transFigure,
            ha='left', va='top',
            multialignment='left',
            fontsize=14, color='black',
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.5,
                boxstyle='round'
            )
        )



        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes

        plt.grid(True, which='major', linestyle='--', linewidth=0.5)


        
        
        return fig
    
    # plotting loop
    #patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    #bx_struct_roi = specific_bx_structure["ROI"]
    #bx_struct_ind = specific_bx_structure["Index number"]

    #sp_patient_all_structure_shifts_pandas_data_frame = pydicom_item[all_ref]["Multi-structure MC simulation output dataframes dict"]["All MC structure transformation values"]
    
    #dose_output_nominal_and_all_MC_trials_pandas_data_frame = specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]

    #dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame = specific_bx_structure["Output data frames"]["Voxel-wise dose output by MC trial number"]

    fig = plot_quantile_regression_and_more_corrected(dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                                                        dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                                                        sp_patient_all_structure_shifts_pandas_data_frame,
                                                        patientUID, 
                                                        bx_struct_roi,
                                                        bx_struct_ind,
                                                        bx_ref)

    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

    fig.savefig(svg_dose_fig_file_path, format='svg')

    # clean up for memory
    plt.close(fig)
            

def production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib_v2(
    sp_patient_all_structure_shifts_pandas_data_frame,
    dose_output_nominal_and_all_MC_trials_pandas_data_frame,
    dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    value_col_key,
    patient_sp_output_figures_dir,
    general_plot_name_string,
    num_rand_trials_to_show,
    y_axis_label,          # kept for backward compatibility; if None we auto-pick from value_col_key
    custom_fig_title,
    trial_annotation_style='number',      # 'arrow' or 'number'  (unchanged)
    # --- NEW options ---
    axis_label_fontsize: int = 15,        # x/y label font size
    tick_labelsize: int = 14,             # tick label size
    show_x_direction_arrow: bool = False, # draw direction arrow + phrase under x-axis
    x_direction_label: str = "To biopsy needle tip / patient superior",
    use_latex_labels: bool = True,         # render labels with mathtext/LaTeX
    show_title: bool = True,

):
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.kernel_regression import KernelReg
    from scipy.stats import gaussian_kde

    # plotting function (internals unchanged except for label sizes/latex + arrow)
    def plot_quantile_regression_and_more_corrected(
        df,
        df_voxelized,
        sp_patient_all_structure_shifts_pandas_data_frame,
        patientUID,
        bx_id,
        bx_struct_ind,
        bx_ref,
        trial_annotation_style=trial_annotation_style,
        linestyle_regressions='-',
        line_width_regressions=2,
        linestyle_regressions_trials='--',
        line_width_regressions_trials=1,
        linestyle_quantiles=':',
        line_width_quantiles=1
    ):
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))

        # Common x-range
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Placeholder for quantile regressions
        y_regressions = {}

        # Kernel regression helper (unchanged rendering, now reused for annotations)
        def perform_and_plot_kernel_regression(x, y, x_range, label, color,
                                               annotation_text=None, target_offset=0,
                                               linestyle='-', linewidth=2):
            kr = KernelReg(endog=y, exog=x, var_type='c', bw=[1])
            y_kr, _ = kr.fit(x_range)
            plotted_line, = plt.plot(x_range, y_kr, label=label,
                                     color=color, linewidth=linewidth, linestyle=linestyle)

            # Optional arrow annotation on line (only used in 'arrow' style)
            if annotation_text is not None:
                total_points = len(x_range)
                target_index = (total_points // 5 * target_offset) % total_points
                target_x = x_range[target_index]
                target_y = y_kr[target_index]
                plt.annotate(
                    annotation_text,
                    xy=(target_x, target_y),
                    xytext=(target_x + 1, target_y + 1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    fontsize=10,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                              facecolor="white", alpha=0.8)
                )
            return y_kr, plotted_line

        # Quantile curves (unchanged)
        for quantile in [0.05, 0.25, 0.50, 0.75, 0.95]:
            q_df = df.groupby('Z (Bx frame)')[value_col_key].quantile(quantile).reset_index()
            kr = KernelReg(endog=q_df[value_col_key], exog=q_df['Z (Bx frame)'], var_type='c', bw=[1])
            y_kr, _ = kr.fit(x_range)
            y_regressions[quantile] = y_kr

        fill_1 = plt.fill_between(x_range, y_regressions[0.05], y_regressions[0.25], color='springgreen', alpha=0.5)
        fill_2 = plt.fill_between(x_range, y_regressions[0.25], y_regressions[0.75], color='dodgerblue', alpha=0.5)
        fill_3 = plt.fill_between(x_range, y_regressions[0.75], y_regressions[0.95], color='springgreen', alpha=0.5)

        plt.plot(x_range, y_regressions[0.05], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.25], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.75], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_range, y_regressions[0.95], linestyle=linestyle_quantiles, linewidth=line_width_quantiles, color='black')
        
        # Median line
        q50_line, = plt.plot(x_range, y_regressions[0.50], linestyle='-', linewidth=2.0, color='black', label='Median (Q50)')

        # Nominal (trial 0)
        mc_trial_0 = df[df['MC trial num'] == 0]
        _, nominal_line = perform_and_plot_kernel_regression(
            mc_trial_0['Z (Bx frame)'], mc_trial_0[value_col_key], x_range,
            'Nominal', 'red', linestyle='-', linewidth=2
        )

        # KDE max-density & mean per unique Z (unchanged)
        kde_max_doses, mean_doses, z_vals = [], [], []
        for z_val in df['Z (Bx frame)'].unique():
            pt_data = df[df['Z (Bx frame)'] == z_val]
            kde = gaussian_kde(pt_data[value_col_key])
            kde_doses = np.linspace(pt_data[value_col_key].min(), pt_data[value_col_key].max(), 500)
            max_density_dose = kde_doses[np.argmax(kde(kde_doses))]
            kde_max_doses.append(max_density_dose)
            mean_doses.append(pt_data[value_col_key].mean())
            z_vals.append(z_val)

        _, max_density_line = perform_and_plot_kernel_regression(
            z_vals, kde_max_doses, x_range, 'Mode (KDE)', 'magenta',
            linestyle='-', linewidth=2
        )
        _, mean_line = perform_and_plot_kernel_regression(
            z_vals, mean_doses, x_range, 'Mean Dose', 'orange',
            linestyle='-', linewidth=2
        )

        # Random trial regressions + scatter (unchanged logic)
        annotation_offset_index = 0
        annotation_lines = []  # used only for 'number' style

        for trial in range(1, num_rand_trials_to_show + 1):
            mc_trial_shift_vec_df = sp_patient_all_structure_shifts_pandas_data_frame[
                (sp_patient_all_structure_shifts_pandas_data_frame["Trial"] == trial) &
                (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
                (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
            ].reset_index(drop=True)

            mc_trial = df[df['MC trial num'] == trial]
            mc_trial_voxelized = df_voxelized[df_voxelized['MC trial num'] == trial]

            x_dist = mc_trial_shift_vec_df.at[0, 'Shift (X)']
            y_dist = mc_trial_shift_vec_df.at[0, 'Shift (Y)']
            z_dist = mc_trial_shift_vec_df.at[0, 'Shift (Z)']
            d_tot = math.sqrt(x_dist**2 + y_dist**2 + z_dist**2)

            if trial_annotation_style == 'arrow':
                annotation_text_for_trial = f"({x_dist:.1f},{y_dist:.1f},{z_dist:.1f}), d = {d_tot:.1f} mm"
                perform_and_plot_kernel_regression(
                    mc_trial['Z (Bx frame)'],
                    mc_trial[value_col_key],
                    x_range,
                    f"Trial: {trial}",
                    'black',
                    annotation_text=annotation_text_for_trial,
                    target_offset=annotation_offset_index,
                    linestyle='--',
                    linewidth=1
                )
                plt.scatter(mc_trial_voxelized['Z (Bx frame)'],
                            mc_trial_voxelized[value_col_key],
                            color='grey', alpha=0.1, s=10, zorder=1.1)
                annotation_offset_index += 1
            else:  # 'number' style
                line_str = f"{trial}: ({x_dist:.1f}, {y_dist:.1f}, {z_dist:.1f}), d = {d_tot:.1f} mm"
                annotation_lines.append(line_str)

                y_kr, _ = perform_and_plot_kernel_regression(
                    mc_trial['Z (Bx frame)'],
                    mc_trial[value_col_key],
                    x_range,
                    str(trial),
                    'black',
                    linestyle='--',
                    linewidth=1
                )
                plt.scatter(mc_trial_voxelized['Z (Bx frame)'],
                            mc_trial_voxelized[value_col_key],
                            color='grey', alpha=0.1, s=10, zorder=1.1)
                # Right-end line number
                plt.text(x_range[-1] + 0.1, y_kr[-1], str(trial),
                         fontsize=14, color='black', ha='left', va='center')

        ax = plt.gca()

        # --- Titles and labels (sizes & LaTeX units) ---
        if show_title and custom_fig_title not in (None, "", False):
            ax.set_title(f'{patientUID} - {bx_id} - {custom_fig_title}', fontsize=16)


        # Default label texts (LaTeX/mathtext friendly)
        if use_latex_labels:
            x_label_default = r'Biopsy Coordinate $z$ $(\mathrm{mm})$'
            if y_axis_label is not None and 'grad' in y_axis_label.lower():
                y_label_default = r'Dose-gradient Magnitude $(\mathrm{Gy}\ \mathrm{mm}^{-1})$'
            elif y_axis_label is not None and 'dose' in y_axis_label.lower():
                y_label_default = r'Dose $(\mathrm{Gy})$'
            else:
                # Infer from value_col_key
                if 'grad' in value_col_key.lower():
                    y_label_default = r'Dose-gradient Magnitude $(\mathrm{Gy}\ \mathrm{mm}^{-1})$'
                else:
                    y_label_default = r'Dose $(\mathrm{Gy})$'
        else:
            x_label_default = 'Biopsy Coordinate z (mm)'
            if y_axis_label is not None:
                y_label_default = y_axis_label
            else:
                y_label_default = 'Dose (Gy)' if 'grad' not in value_col_key.lower() else 'Dose-gradient Magnitude (Gy/mm)'

        ax.set_xlabel(x_label_default, fontsize=axis_label_fontsize)
        ax.set_ylabel(y_label_default, fontsize=axis_label_fontsize)

        # Legend (unchanged content)
        legend_handles = [fill_1, fill_2, fill_3] + [q50_line, nominal_line, max_density_line, mean_line]
        quantile_labels = ['5th–25th Q', '25th–75th Q', '75th–95th Q']
        main_labels = ['Median (Q50)', 'Nominal', 'Mode', 'Mean']
        leg = ax.legend(
            legend_handles,
            quantile_labels + main_labels,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            facecolor='white',
            framealpha=1.0,
            prop={'size': 12}
        )


        # Layout & the per-trial annotation block (unchanged)
        plt.tight_layout(rect=[0.01, 0.06, 1, 1])  # leave a bit more bottom room for the optional x-arrow

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_frame = leg.get_frame()
        bbox_disp = leg_frame.get_window_extent(renderer)
        inv_fig = fig.transFigure.inverted()
        frame_x0, frame_y0 = inv_fig.transform((bbox_disp.x0, bbox_disp.y0))

        text_x = frame_x0
        text_y = frame_y0 - 0.02
        if trial_annotation_style == 'number' and len(annotation_lines) > 0:
            annotation_text = "\n".join(annotation_lines)
            fig.text(
                text_x, text_y, annotation_text,
                transform=fig.transFigure,
                ha='left', va='top',
                multialignment='left',
                fontsize=14, color='black',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round')
            )

        # Grid & ticks (sizes customizable)
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullFormatter

        # draw grids under data
        ax.set_axisbelow(True)

        # --- ticks ---
        # sizes & direction
        ax.tick_params(axis='both', which='major', length=6, width=1.0, direction='out', labelsize=tick_labelsize)
        ax.tick_params(axis='both', which='minor', length=3, width=0.8, direction='out')

        # minor ticks on
        ax.minorticks_on()

        # If your x is in mm and you want integers as major ticks:
        ax.xaxis.set_major_locator(MultipleLocator(1))     # major every 1 mm (adjust as needed)
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))   # minor halfway between majors

        # Let y auto-place minors between majors (2 per major interval)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # (Optional) hide minor tick labels but keep ticks
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        # --- grids ---
        ax.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.grid(which='minor', linestyle=':',  linewidth=0.4, alpha=0.45)


        # Optional x-direction arrow + label just below x-axis label
        if show_x_direction_arrow:
            # Put a LaTeX right-arrow directly in the label text
            x_dir = x_direction_label
            if r"\rightarrow" not in x_dir:
                x_dir = x_dir.rstrip() + r" $\rightarrow$"

            # Draw only the text (no separate arrow annotation)
            y_off = -0.14  # keep your vertical placement
            ax.text(
                0.50, y_off - 0.02, x_dir,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=max(10, axis_label_fontsize - 1)
            )

        return fig

    # === Call the inner plotting function (unchanged data flow) ===
    fig = plot_quantile_regression_and_more_corrected(
        dose_output_nominal_and_all_MC_trials_pandas_data_frame,
        dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
        sp_patient_all_structure_shifts_pandas_data_frame,
        patientUID,
        bx_struct_roi,
        bx_struct_ind,
        bx_ref
    )

    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)



def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v2(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                     cumulative_dvh_pandas_dataframe,
                                                                     patient_sp_output_figures_dir,
                                                                    patientUID,
                                                                    bx_struct_roi,
                                                                    bx_struct_ind,
                                                                    bx_ref,
                                                                    general_plot_name_string,
                                                                    num_rand_trials_to_show,
                                                                    custom_fig_title,
                                                                    trial_annotation_style = 'number', # ← new: 'arrow' or 'number'
                                                                    dvh_option = {'dvh':'cumulative', 'x-col': 'Dose (Gy)', 'x-axis-label': 'Dose (Gy)', 'y-col': 'Percent volume', 'y-axis-label': 'Percent Volume (%)'},    # can be 'cumulative' or 'differential'
                                                                    random_trial_line_color = 'black'
                                                                    ):
    plt.ioff()

    def interpolate_quantile_line(x_vals, y_vals, x_dense):
        interpolator = PchipInterpolator(x_vals, y_vals)
        return interpolator(x_dense)


    def plot_kernel_regression(x, y, label, color, annotation_text = None, target_offset=0, linestyle='-', linewidth=2):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                x_range = np.linspace(x.min(), x.max(), 500)
                if dvh_option['dvh'] == 'differential':
                    y_kr = interpolate_quantile_line(x, y, x_range)
                else:
                    kr = KernelReg(endog=y, exog=x, var_type='c')
                    y_kr, _ = kr.fit(x_range)

                
                

                plotted_line, = plt.plot(x_range, y_kr, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                if w:
                    for warning in w:
                        print(f"Warning encountered for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}")
        except np.linalg.LinAlgError:
            print(f"SVD did not converge for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}")
            plotted_line, =  plt.plot(x, y, label=label, color=color, linestyle=linestyle, marker=None, linewidth=linewidth)

        # Add annotation if provided
        if annotation_text is not None:
            # Determine the total number of points
            total_points = len(x_range)
            
            # Calculate the target index based on the offset, with wrapping
            target_index = (total_points // 5 * target_offset) % total_points
            
            # Target point coordinates
            target_x = x_range[target_index]
            target_y = y_kr[target_index]
            
            # Add annotation with an arrow
            plt.annotate(
                annotation_text, 
                xy=(target_x, target_y),  # Point to annotate
                xytext=(target_x + 1, target_y + 1),  # Offset for text
                arrowprops=dict(
                    arrowstyle="->",  # Arrow style
                    color=color,      # Arrow color
                    lw=1.5            # Line width
                ),
                fontsize=10,        # Font size of annotation
                color=color,        # Color of annotation text
                bbox=dict(
                    boxstyle="round,pad=0.3",  # Text box style
                    edgecolor=color,          # Edge color of box
                    facecolor="white",        # Background color of box
                    alpha=0.8                 # Transparency of box
                )
            )

        return y_kr, x_range, plotted_line
            
    def build_plot(df, 
                   x_col, 
                   y_col, 
                   patientUID, 
                   bx_struct_roi,trial_annotation_style = trial_annotation_style, 
                   line_style_quantiles=':', 
                   line_width_quantiles=1, 
                   linestyle_regressions='-', 
                   line_width_regressions=2,
                   linestyle_regressions_trials='--', 
                   line_width_regressions_trials=1):
        
        fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

        # Calculate and plot kernel regressions for the desired quantiles
        quantiles = [0.05, 0.25, 0.75, 0.95]
        quantile_dfs = {}
        x_ranges = {}
        y_krs = {}
        
        for q in quantiles:
            q_df = df.groupby(x_col)[y_col].quantile(q).reset_index()
            quantile_dfs[q] = q_df
            x_range = np.linspace(df[x_col].min(), df[x_col].max(), 500)
            x_ranges[q] = x_range

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # if its differential then we interpolate
                    if dvh_option['dvh'] == 'differential':
                        y_kr = interpolate_quantile_line(q_df[x_col], q_df[y_col], x_range)
                    else: # if its cumulative then we do kernel regression
                        kr = KernelReg(endog=q_df[y_col], exog=q_df[x_col], var_type='c')
                        y_kr, _ = kr.fit(x_range)
                    
                    y_krs[q] = y_kr
                    if w:
                        for warning in w:
                            print(f"Warning encountered for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}")
            
            except np.linalg.LinAlgError:
                print(f"SVD did not converge for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}")
                # Perform linear interpolation
                if not q_df.empty:
                    y_kr = np.interp(x_range, q_df[x_col], q_df[y_col])
                else:
                    y_kr = np.zeros_like(x_range)
                y_krs[q] = y_kr  # Use interpolated values as a fallback or all zeros if have an empty dataframe, which we should never have
            

        # Filling the areas between quantile regressions
        fill_1 = plt.fill_between(x_ranges[0.05], y_krs[0.05], y_krs[0.25], color='springgreen', alpha=1)
        fill_2 = plt.fill_between(x_ranges[0.25], y_krs[0.25], y_krs[0.75], color='dodgerblue', alpha=1)
        fill_3 = plt.fill_between(x_ranges[0.75], y_krs[0.75], y_krs[0.95], color='springgreen', alpha=1)

        plt.plot(x_ranges[0.05], y_krs[0.05], linestyle=line_style_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_ranges[0.25], y_krs[0.25], linestyle=line_style_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_ranges[0.75], y_krs[0.75], linestyle=line_style_quantiles, linewidth=line_width_quantiles, color='black')
        plt.plot(x_ranges[0.95], y_krs[0.95], linestyle=line_style_quantiles, linewidth=line_width_quantiles, color='black')

        # 3. Kernel regression for 'MC trial' == 0
        df_trial_0 = df[df['MC trial'] == 0]
        _,_, nominal_line = plot_kernel_regression(df_trial_0[x_col], df_trial_0[y_col], 'Nominal', 'red', linestyle=linestyle_regressions, linewidth=line_width_regressions)

        # Scatter plot for the data points
        #plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.1, s=10)  # 's' controls size, 'alpha' controls transparency
        



        annotation_lines = []
        annotation_offset_index = 0

        for trial in range(1,num_rand_trials_to_show + 1): # +1 because we start at 1 in range()
            mc_trial_shift_vec_df = sp_patient_all_structure_shifts_pandas_data_frame[(sp_patient_all_structure_shifts_pandas_data_frame["Trial"] == trial) &
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) & 
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)].reset_index(drop=True)
            mc_trial = df[df['MC trial'] == trial]

            x_dist = mc_trial_shift_vec_df.at[0,'Shift (X)']
            y_dist = mc_trial_shift_vec_df.at[0,'Shift (Y)']
            z_dist = mc_trial_shift_vec_df.at[0,'Shift (Z)']
            d_tot = math.sqrt(x_dist**2 + y_dist**2 + z_dist**2)


            if trial_annotation_style == 'arrow':

                annotation_text_for_trial = f"({x_dist:.1f},{y_dist:.1f},{z_dist:.1f}), d = {d_tot:.1f} mm"
            
                plot_kernel_regression(mc_trial[dvh_option['x-col']], 
                                        mc_trial[dvh_option['y-col']], 
                                        f"Trial: {trial}", 
                                        random_trial_line_color, 
                                        annotation_text = annotation_text_for_trial, 
                                        target_offset=annotation_offset_index,
                                        linestyle=linestyle_regressions_trials,
                                        linewidth=line_width_regressions_trials)


                annotation_offset_index += 1

            
            else: # ‘number’ style

                # 1) Build the per‐trial annotation line
                line_str = f"{trial}: ({x_dist:.1f}, {y_dist:.1f}, {z_dist:.1f}), d = {d_tot:.1f} mm"
                annotation_lines.append(line_str)

                # 2) plot and grab the y-values at x_range
                y_kr, x_range, _ = plot_kernel_regression(
                            mc_trial[dvh_option['x-col']],
                            mc_trial[dvh_option['y-col']],
                            f"Trial: {trial}",      # legend entry is now just the number
                            random_trial_line_color,
                            linestyle=linestyle_regressions_trials,
                            linewidth=line_width_regressions_trials
                            )
            

                # —— 3) label each line end with its number
                if dvh_option['dvh'] == 'cumulative':
                    y_target = random.randrange(30, 60)
                    closest_idx = np.argmin(np.abs(y_kr - y_target))

                    plt.text(
                        x_range[closest_idx],    # 0.1 mm further from the right edge
                        y_target,           # the line’s final y-value
                        str(trial),
                        fontsize=14,
                        color='black',
                        ha='left', va='center'
                    )
                else:
                    # get the max y value
                    y_target = np.max(y_kr)
                    closest_idx = np.argmin(np.abs(y_kr - y_target))
                    plt.text(
                        x_range[closest_idx],    # 0.1 mm further from the right edge
                        y_target,           # the line’s final y-value
                        str(trial),
                        fontsize=14,
                        color='black',
                        ha='left', va='center'
                    )

        ax = plt.gca() 

        # Final plot adjustments
        ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}',
             fontsize=16,      # Increase the title font size
             #fontname='Arial' # Set the title font family
            )
        ax.set_xlabel(dvh_option['x-axis-label'],
              fontsize=16,    # sets the font size
              #fontname='Arial'
               )   # sets the font family

        ax.set_ylabel(dvh_option['y-axis-label'],
                    fontsize=16,
                    #fontname='Arial'
                    )


        legend_handles = [fill_1, fill_2, fill_3] + [nominal_line]

        quantile_labels = ['5th–25th Q','25th–75th Q','75th–95th Q']
        main_labels     = ['Nominal']


        leg = ax.legend(legend_handles,
            quantile_labels + main_labels ,
            loc='best', facecolor='white',
            prop={
                'size': 14,        # font size
                #'family': 'serif', # optional: font family
                #'weight': 'bold'   # optional: font weight
            }
        )



        plt.tight_layout(rect=[0.01, 0.01, 1, 1])  # Adjust layout to fit the legend


        fig.canvas.draw()  # needed to compute text/legend positions
        renderer = fig.canvas.get_renderer()

        # 3) Get the frame (background patch) bbox in display coords:
        leg_frame = leg.get_frame()
        bbox_disp = leg_frame.get_window_extent(renderer)
    
        # 4) Convert that to figure coords:
        inv_fig = fig.transFigure.inverted()
        # top-left corner of the frame:
        frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
    
        # 5) Compute just below the legend frame:
        text_x = frame_x1
        text_y = frame_y0 - 0.02  # 2% of figure height below the legend
    
        # 6) Draw in FIGURE coordinates so it lines up perfectly:
        annotation_text = "\n".join(annotation_lines)
        fig.text(
            text_x, text_y, annotation_text,
            transform=fig.transFigure,
            ha='right', va='top',
            multialignment='left',
            fontsize=14, color='black',
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.5,
                boxstyle='round'
            )
        )

        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes

        plt.grid(True, which='major', linestyle='--', linewidth=0.5)

        #plt.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal'], loc='best', facecolor = 'white')
        #plt.tight_layout()

        return fig


    df = cumulative_dvh_pandas_dataframe

    fig = build_plot(df, dvh_option['x-col'], dvh_option['y-col'], patientUID, bx_struct_roi, trial_annotation_style = trial_annotation_style)
    
    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

    fig.savefig(svg_dose_fig_file_path, format='svg')

    # clean up for memory
    plt.close(fig)



# Faster no kernel regression, not necessary
def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v3(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black'
):
    """
    v3: Linear interpolation only (fast), no kernel regression.
    Works for both grid-resampled or unique-knot DVH inputs.
    """
    plt.ioff()

    # ---------- helpers ----------
    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        """Linear interpolation; enforce monotone non-increasing for cumulative."""
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            # Ensure non-increasing with dose
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            # Safety clamp
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:  # differential
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _line_and_optional_annotation(x_grid, y_grid, label, color,
                                      annotation_text=None, target_offset=0,
                                      linestyle='-', linewidth=2):
        """Plot the line and optionally annotate one point."""
        plotted_line, = plt.plot(x_grid, y_grid, label=label,
                                 color=color, linestyle=linestyle, linewidth=linewidth)
        if annotation_text is not None and len(x_grid) > 0:
            total_points = len(x_grid)
            idx = (total_points // 5 * target_offset) % total_points
            tx, ty = x_grid[idx], y_grid[idx]
            plt.annotate(
                annotation_text,
                xy=(tx, ty), xytext=(tx + 1, ty + 1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                          facecolor="white", alpha=0.8)
            )
        return plotted_line

    # ---------- main plotting logic ----------
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    # Build a common dose grid for this plot (500 points across observed range)
    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min:
        x_grid = np.array([x_min], dtype=float)
    else:
        x_grid = np.linspace(x_min, x_max, 500, dtype=float)

    # Stack each trial on the same x_grid for quantile bands
    trial_ids = df['MC trial'].unique()
    trial_ids = np.sort(trial_ids)  # include trial 0 as part of distribution
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)

    # Pre-index shifts by trial for fast lookup
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ]
        .set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )

    # Pre-split dvh by trial for speed
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}

    for i, t in enumerate(trial_ids):
        g = trials_dict.get(t)
        if g is None or g.empty:
            Y[i] = np.zeros_like(x_grid)
            continue
        x = g[x_col].to_numpy()
        y = g[y_col].to_numpy()
        Y[i] = _interp_curve_linear(x, y, x_grid, dvh_kind=dvh_kind)

    # Quantile bands across trials
    q05 = np.percentile(Y, 5,  axis=0)
    q25 = np.percentile(Y, 25, axis=0)
    q75 = np.percentile(Y, 75, axis=0)
    q95 = np.percentile(Y, 95, axis=0)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    # Bands
    fill_1 = plt.fill_between(x_grid, q05, q25, color='springgreen', alpha=1)
    fill_2 = plt.fill_between(x_grid, q25, q75, color='dodgerblue',  alpha=1)
    fill_3 = plt.fill_between(x_grid, q75, q95, color='springgreen', alpha=1)
    # Quantile outlines
    plt.plot(x_grid, q05, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q25, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q75, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q95, linestyle=':', linewidth=1, color='black')

    # Nominal (MC trial == 0)
    g0 = trials_dict.get(0)
    if g0 is not None and not g0.empty:
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nominal_line = _line_and_optional_annotation(
            x_grid, y0, 'Nominal', 'red', linestyle='-', linewidth=2
        )
    else:
        nominal_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')

    # Random trials (exclude trial 0)
    nonzero_trials = trial_ids[trial_ids != 0]
    if len(nonzero_trials) > 0 and num_rand_trials_to_show > 0:
        k = min(num_rand_trials_to_show, len(nonzero_trials))
        # pick the first k or sample; keep deterministic order (fast & reproducible)
        chosen = nonzero_trials[:k]
        annotation_lines = []
        annotation_offset_index = 0

        for trial in chosen:
            gt = trials_dict.get(trial)
            if gt is None or gt.empty:
                continue
            yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                      x_grid, dvh_kind=dvh_kind)

            # Annotation text from shifts (if present)
            if trial in shifts_idx.index:
                sx, sy, sz = shifts_idx.loc[trial, ['Shift (X)','Shift (Y)','Shift (Z)']].astype(float)
                d_tot = math.sqrt(sx*sx + sy*sy + sz*sz)
                ann_text = f"({sx:.1f},{sy:.1f},{sz:.1f}), d = {d_tot:.1f} mm"
            else:
                ann_text = None

            if trial_annotation_style == 'arrow':
                _line_and_optional_annotation(
                    x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                    annotation_text=ann_text, target_offset=annotation_offset_index,
                    linestyle='--', linewidth=1
                )
                annotation_offset_index += 1
            else:
                # number style: draw the line, and place the trial number
                _line_and_optional_annotation(
                    x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                    linestyle='--', linewidth=1
                )
                if dvh_kind == 'cumulative':
                    # label around a mid band value
                    y_target = np.clip(0.5*(np.nanmin(yt)+np.nanmax(yt)), 0, 100)
                else:
                    y_target = np.nanmax(yt)
                if np.isfinite(y_target):
                    idx = int(np.nanargmin(np.abs(yt - y_target)))
                    plt.text(x_grid[idx], y_target, str(int(trial)), fontsize=14,
                             color='black', ha='left', va='center')

                # Collect side-box text
                if ann_text:
                    annotation_lines.append(f"{int(trial)}: {ann_text}")

        # Legend & optional side text
        ax = plt.gca()
        legend_handles = [fill_1, fill_2, fill_3, nominal_line]
        quantile_labels = ['5th–25th Q','25th–75th Q','75th–95th Q']
        main_labels     = ['Nominal']
        leg = ax.legend(legend_handles, quantile_labels + main_labels,
                        loc='best', facecolor='white',
                        prop={'size': 14})

        # Side annotation box (number style)
        if trial_annotation_style != 'arrow' and len(annotation_lines) > 0:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox_disp = leg.get_frame().get_window_extent(renderer)
            inv_fig = fig.transFigure.inverted()
            frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
            fig.text(frame_x1, frame_y0 - 0.02, "\n".join(annotation_lines),
                     transform=fig.transFigure, ha='right', va='top',
                     multialignment='left', fontsize=14, color='black',
                     bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round'))
    else:
        ax = plt.gca()
        legend_handles = [fill_1, fill_2, fill_3, nominal_line]
        leg = ax.legend(legend_handles, ['5th–25th Q','25th–75th Q','75th–95th Q','Nominal'],
                        loc='best', facecolor='white', prop={'size': 14})

    # Axes, labels, grids
    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    # Save + close
    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.savefig(svg_dose_fig_file_path, format='svg')
    plt.close(fig)






# includes mean and median lines as options 
# note that the mean line is likely useless, way too smooth 
def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    show_median_line=True,   # NEW: draw Q50 curve
    show_mean_line=False     # NEW: draw mean curve
):
    """
    v3: Linear interpolation only (fast), no kernel regression.
    - Works for both grid-resampled or unique-knot DVH inputs.
    - Adds optional median (Q50) and mean curves across trials.
    - Selects the first `k` nonzero trials (v2 behavior).
    """
    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            # enforce non-increasing vs dose
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        # tiny safety to remove any upward blips due to interpolation/precision
        return np.maximum.accumulate(y[::-1])[::-1]

    def _line_with_optional_annotation(x_grid, y_grid, label, color,
                                       annotation_text=None, target_offset=0,
                                       linestyle='-', linewidth=2):
        line, = plt.plot(x_grid, y_grid, label=label, color=color,
                         linestyle=linestyle, linewidth=linewidth)
        if annotation_text is not None and len(x_grid) > 0:
            n = len(x_grid)
            idx = (n // 5 * target_offset) % n
            tx, ty = x_grid[idx], y_grid[idx]
            plt.annotate(
                annotation_text, xy=(tx, ty), xytext=(tx + 1, ty + 1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                          facecolor="white", alpha=0.8)
            )
        return line

    # --- inputs & common grid ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 500, dtype=float)

    # Pre-index shifts and trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ]
        .set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))

    # Interpolate each trial to the grid
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Quantile bands across trials
    q05 = np.percentile(Y,  5, axis=0)
    q25 = np.percentile(Y, 25, axis=0)
    q50 = np.percentile(Y, 50, axis=0)  # median (for optional line)
    q75 = np.percentile(Y, 75, axis=0)
    q95 = np.percentile(Y, 95, axis=0)
    mean_curve = np.nanmean(Y, axis=0)  # optional mean line

    if dvh_kind == 'cumulative':
        # keep them strictly non-increasing
        q05 = _enforce_nonincreasing(q05)
        q25 = _enforce_nonincreasing(q25)
        q50 = _enforce_nonincreasing(q50)
        q75 = _enforce_nonincreasing(q75)
        q95 = _enforce_nonincreasing(q95)
        mean_curve = _enforce_nonincreasing(mean_curve)

    # --- figure & bands ---
    fig = plt.figure(figsize=(12, 8))
    fill_1 = plt.fill_between(x_grid, q05, q25, color='springgreen', alpha=1)
    fill_2 = plt.fill_between(x_grid, q25, q75, color='dodgerblue',  alpha=1)
    fill_3 = plt.fill_between(x_grid, q75, q95, color='springgreen', alpha=1)
    plt.plot(x_grid, q05, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q25, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q75, linestyle=':', linewidth=1, color='black')
    plt.plot(x_grid, q95, linestyle=':', linewidth=1, color='black')

    # Nominal line (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nominal_line = _line_with_optional_annotation(x_grid, y0, 'Nominal', 'red',
                                                      linestyle='-', linewidth=2)
    else:
        nominal_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')

    # Optional summary lines
    extra_handles = []
    extra_labels  = []
    if show_median_line:
        median_line, = plt.plot(x_grid, q50, color='black', linewidth=2, linestyle='-',
                                label='Q50 (median)')
        extra_handles.append(median_line); extra_labels.append('Q50 (median)')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_curve, color='orange', linewidth=2, linestyle='--',
                              label='Mean')
        extra_handles.append(mean_line); extra_labels.append('Mean')

    # First-k trials after 0 (v2 behavior)
    nonzero_trials = [t for t in trial_ids if t != 0]
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]

    annotation_lines = []
    annotation_offset_index = 0
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)

        # build annotation text from shifts if available
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial, ['Shift (X)','Shift (Y)','Shift (Z)']].astype(float)
            d_tot = math.sqrt(sx*sx + sy*sy + sz*sz)
            ann_text = f"({sx:.1f},{sy:.1f},{sz:.1f}), d = {d_tot:.1f} mm"
        else:
            ann_text = None

        if trial_annotation_style == 'arrow':
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                annotation_text=ann_text, target_offset=annotation_offset_index,
                linestyle='--', linewidth=1
            )
            annotation_offset_index += 1
        else:  # number style
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                linestyle='--', linewidth=1
            )
            # label near a mid value (deterministic)
            if dvh_kind == 'cumulative':
                y_target = float(np.clip(0.5*(np.nanmin(yt)+np.nanmax(yt)), 0, 100))
            else:
                y_target = float(np.nanmax(yt))
            if np.isfinite(y_target):
                idx = int(np.nanargmin(np.abs(yt - y_target)))
                plt.text(x_grid[idx], y_target, str(int(trial)), fontsize=14,
                         color='black', ha='left', va='center')
            if ann_text:
                annotation_lines.append(f"{int(trial)}: {ann_text}")

    # Legend & cosmetics
    ax = plt.gca()
    handles = [fill_1, fill_2, fill_3, nominal_line] + extra_handles
    labels  = ['5th–25th Q','25th–75th Q','75th–95th Q','Nominal'] + extra_labels
    leg = ax.legend(handles, labels, loc='best', facecolor='white',
                    prop={'size': 14})

    if trial_annotation_style != 'arrow' and len(annotation_lines) > 0:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_frame().get_window_extent(renderer)
        inv_fig = fig.transFigure.inverted()
        frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
        fig.text(frame_x1, frame_y0 - 0.02, "\n".join(annotation_lines),
                 transform=fig.transFigure, ha='right', va='top',
                 multialignment='left', fontsize=14, color='black',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round'))

    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.savefig(svg_dose_fig_file_path, format='svg')
    plt.close(fig)




def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_1(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    # NEW options
    bands_mode="horizontal",           # 'vertical' (as in v3), 'horizontal' (Dx-consistent), or 'both'
    show_median_line=True,
    show_mean_line=False,
    show_dx_vy_markers=True,
    dx_list=(2, 50, 98),
    vy_list=(100, 125, 150, 175, 200, 300),
    ref_dose_gy=13.5
):
    """
    v3.1: Linear interpolation only. Adds:
      - horizontal-quantile bands (Dx-consistent) via bands_mode
      - Dx/Vy overlay markers matching your tables
    """
    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        """Linear interp; enforce non-increasing for cumulative."""
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        return np.maximum.accumulate(y[::-1])[::-1]

    def _line_with_optional_annotation(x_grid, y_grid, label, color,
                                       annotation_text=None, target_offset=0,
                                       linestyle='-', linewidth=2):
        line, = plt.plot(x_grid, y_grid, label=label, color=color,
                         linestyle=linestyle, linewidth=linewidth)
        if annotation_text is not None and len(x_grid) > 0:
            n = len(x_grid)
            idx = (n // 5 * target_offset) % n
            tx, ty = x_grid[idx], y_grid[idx]
            plt.annotate(
                annotation_text, xy=(tx, ty), xytext=(tx + 1, ty + 1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                          facecolor="white", alpha=0.8)
            )
        return line

    # --- inputs & common x-grid ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 500, dtype=float)

    # Pre-index shifts and trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ]
        .set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))
    trial_to_row = {int(t): i for i, t in enumerate(trial_ids)}

    # Interpolate each trial to the grid
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Vertical quantiles (as in v3): distribution of %volume at fixed dose
    q05_v = np.percentile(Y,  5, axis=0)
    q25_v = np.percentile(Y, 25, axis=0)
    q50_v = np.percentile(Y, 50, axis=0)
    q75_v = np.percentile(Y, 75, axis=0)
    q95_v = np.percentile(Y, 95, axis=0)
    mean_v = np.nanmean(Y, axis=0)

    if dvh_kind == 'cumulative':
        q05_v = _enforce_nonincreasing(q05_v)
        q25_v = _enforce_nonincreasing(q25_v)
        q50_v = _enforce_nonincreasing(q50_v)
        q75_v = _enforce_nonincreasing(q75_v)
        q95_v = _enforce_nonincreasing(q95_v)
        mean_v = _enforce_nonincreasing(mean_v)

    # Horizontal quantiles (Dx-consistent): distribution of DOSE at fixed %volume
    # Build a y-grid (0..100). We’ll invert each trial: x_of_y = interp(y -> x).
    y_grid = np.linspace(100.0, 0.0, 500)  # descending for cumulative
    X_of_Y = None
    if dvh_kind == 'cumulative':
        X_of_Y = np.empty((len(trial_ids), y_grid.size), dtype=float)
        for i in range(len(trial_ids)):
            y = Y[i]              # shape: (len(x_grid),)
            x = x_grid
            # Ensure y is non-increasing (already enforced), then invert
            # interp expects increasing x; here we need x(y), so give y reversed and x reversed
            X_of_Y[i] = np.interp(y_grid, y[::-1], x[::-1],
                                  left=x.min(), right=x.max())
        # Percentiles of DOSE at fixed %volume:
        q05_h = np.percentile(X_of_Y,  5, axis=0)
        q25_h = np.percentile(X_of_Y, 25, axis=0)
        q50_h = np.percentile(X_of_Y, 50, axis=0)  # "median DVH" in Dx-sense
        q75_h = np.percentile(X_of_Y, 75, axis=0)
        q95_h = np.percentile(X_of_Y, 95, axis=0)
        mean_h = np.nanmean(X_of_Y, axis=0)

    # --- figure ---
    fig = plt.figure(figsize=(12, 8))
    handles, labels = [], []

    # Bands per mode
    if bands_mode in ("vertical", "both"):
        fill_1 = plt.fill_between(x_grid, q05_v, q25_v, color='springgreen', alpha=0.7)
        fill_2 = plt.fill_between(x_grid, q25_v, q75_v, color='dodgerblue',  alpha=0.7)
        fill_3 = plt.fill_between(x_grid, q75_v, q95_v, color='springgreen', alpha=0.7)
        plt.plot(x_grid, q05_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q25_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q75_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q95_v, linestyle=':', linewidth=1, color='black')
        handles += [fill_1, fill_2, fill_3]; labels += ['5th–25th Q','25th–75th Q','75th–95th Q']

    if dvh_kind == 'cumulative' and bands_mode in ("horizontal", "both"):
        # Plot horizontal-quantile curves as solid black outlines (Dx-consistent)
        plt.plot(q05_h, y_grid, linestyle='--', linewidth=1, color='gray', alpha=0.9)
        plt.plot(q25_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q75_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q95_h, y_grid, linestyle='--', linewidth=1, color='gray', alpha=0.9)
        # Add legend stubs
        h_stub, = plt.plot([], [], linestyle='-', color='black', linewidth=1.5)
        handles += [h_stub]; labels += ['Dx-consistent quantiles']

    # Nominal (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nominal_line = _line_with_optional_annotation(x_grid, y0, 'Nominal', 'red',
                                                      linestyle='-', linewidth=2)
    else:
        nominal_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')
    handles.append(nominal_line); labels.append('Nominal')

    # Optional summary lines (on vertical bands, for familiarity)
    if show_median_line:
        med_line, = plt.plot(x_grid, q50_v, color='black', linewidth=2, linestyle='-',
                             label='Q50 (median-%vol at fixed dose)')
        handles.append(med_line); labels.append('Q50 (vertical)')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_v, color='orange', linewidth=2, linestyle='--',
                              label='Mean (vertical)')
        handles.append(mean_line); labels.append('Mean')

    # First-k trials after 0 (v2 behavior)
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]
    annotation_lines = []
    annotation_offset_index = 0
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        # shifts text
        ann_text = None
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial, ['Shift (X)','Shift (Y)','Shift (Z)']].astype(float)
            d_tot = math.sqrt(sx*sx + sy*sy + sz*sz)
            ann_text = f"({sx:.1f},{sy:.1f},{sz:.1f}), d = {d_tot:.1f} mm"

        if trial_annotation_style == 'arrow':
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                annotation_text=ann_text, target_offset=annotation_offset_index,
                linestyle='--', linewidth=1
            )
            annotation_offset_index += 1
        else:
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                linestyle='--', linewidth=1
            )
            # deterministic label
            if dvh_kind == 'cumulative':
                y_target = float(np.clip(0.5*(np.nanmin(yt)+np.nanmax(yt)), 0, 100))
            else:
                y_target = float(np.nanmax(yt))
            if np.isfinite(y_target):
                idx = int(np.nanargmin(np.abs(yt - y_target)))
                plt.text(x_grid[idx], y_target, str(int(trial)), fontsize=14,
                         color='black', ha='left', va='center')
            if ann_text:
                annotation_lines.append(f"{int(trial)}: {ann_text}")

    ax = plt.gca()

    # Dx/Vy overlay markers (match your tables)
    if show_dx_vy_markers and dvh_kind == 'cumulative':
        # Dx markers: horizontal lines at y = x% and vertical ticks at quantiles of dose
        for X in dx_list:
            y_target = float(X)

            # Per-trial dose at this %volume (invert: y -> x)  [row-by-row!]
            x_at_y_trials = np.array([
                np.interp(y_target, Y[i, ::-1], x_grid[::-1],
                        left=x_grid[0], right=x_grid[-1])
                for i in range(Y.shape[0])
            ])

            # Stats across trials
            d_q50 = np.percentile(x_at_y_trials, 50)
            d_q25 = np.percentile(x_at_y_trials, 25)
            d_q75 = np.percentile(x_at_y_trials, 75)
            d_q05 = np.percentile(x_at_y_trials,  5)
            d_q95 = np.percentile(x_at_y_trials, 95)

            # Nominal, if trial 0 exists
            d_nom = np.nan
            if 0 in trial_to_row:
                i0 = trial_to_row[0]
                d_nom = np.interp(y_target, Y[i0, ::-1], x_grid[::-1],
                                left=x_grid[0], right=x_grid[-1])

            # Draw guideline + ticks (same as before)
            ax.axhline(y_target, color='gray', lw=0.7, ls=':')
            for x_val, c, lw in [(d_q50, 'k', 2.2), (d_q25, 'k', 1.2), (d_q75, 'k', 1.2),
                                (d_q05, 'k', 0.8), (d_q95, 'k', 0.8)]:
                ax.plot([x_val, x_val], [y_target-1.0, y_target+1.0], color=c, lw=lw)
            if np.isfinite(d_nom):
                ax.plot([d_nom, d_nom], [y_target-2.0, y_target+2.0], color='r', lw=2.2)


        # Vy markers: vertical lines at dose thresholds and horizontal ticks at quantiles of %volume
        for Yp in vy_list:
            thr = (Yp / 100.0) * float(ref_dose_gy)

            # Per-trial %volume at this dose (read forward: x -> y)  [row-by-row!]
            y_at_x_trials = np.array([
                np.interp(thr, x_grid, Y[i, :], left=100.0, right=0.0)
                for i in range(Y.shape[0])
            ])

            v_q50 = np.percentile(y_at_x_trials, 50)
            v_q25 = np.percentile(y_at_x_trials, 25)
            v_q75 = np.percentile(y_at_x_trials, 75)
            v_q05 = np.percentile(y_at_x_trials,  5)
            v_q95 = np.percentile(y_at_x_trials, 95)

            v_nom = np.nan
            if 0 in trial_to_row:
                i0 = trial_to_row[0]
                v_nom = np.interp(thr, x_grid, Y[i0, :], left=100.0, right=0.0)

            ax.axvline(thr, color='gray', lw=0.7, ls=':')
            for y_val, c, lw in [(v_q50, 'k', 2.2), (v_q25, 'k', 1.2), (v_q75, 'k', 1.2),
                                (v_q05, 'k', 0.8), (v_q95, 'k', 0.8)]:
                ax.plot([thr-0.25, thr+0.25], [y_val, y_val], color=c, lw=lw)
            if np.isfinite(v_nom):
                ax.plot([thr-0.5, thr+0.5], [v_nom, v_nom], color='r', lw=2.2)


    # Legend
    leg = ax.legend(handles, labels, loc='best', facecolor='white', prop={'size': 14})

    if trial_annotation_style != 'arrow' and len(annotation_lines) > 0:
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_frame().get_window_extent(renderer)
        inv_fig = fig.transFigure.inverted()
        frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
        fig.text(frame_x1, frame_y0 - 0.02, "\n".join(annotation_lines),
                 transform=fig.transFigure, ha='right', va='top',
                 multialignment='left', fontsize=14, color='black',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round'))

    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')
    plt.close()



def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_2(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    # bands/lines options
    bands_mode="horizontal",           # 'vertical', 'horizontal', or 'both'
    show_median_line=True,
    show_mean_line=False,
    show_dx_vy_markers=True,
    dx_list=(2, 50, 98),
    vy_list=(100, 125, 150, 175, 200, 300),
    ref_dose_gy=13.5,
    # NEW: overlay table metrics
    dvh_metrics_df=None,               # dataframe with columns shown in your message
    overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),  # which stats to plot
    overlay_metrics_alpha=0.95,
):
    """
    v4_2: Linear interpolation only (fast). Adds overlay of table-derived DVH metrics.
    - If dvh_metrics_df is provided, plots its Nominal/Qxx points on top of the DVH plot.
    - Dx points: (dose_from_table, y = x%)
    - Vy points: (x = (y% of ref dose), volume_from_table)
    """

    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        return np.maximum.accumulate(y[::-1])[::-1]

    def _line_with_optional_annotation(x_grid, y_grid, label, color,
                                       annotation_text=None, target_offset=0,
                                       linestyle='-', linewidth=2):
        line, = plt.plot(x_grid, y_grid, label=label, color=color,
                         linestyle=linestyle, linewidth=linewidth)
        if annotation_text is not None and len(x_grid) > 0:
            n = len(x_grid)
            idx = (n // 5 * target_offset) % n
            tx, ty = x_grid[idx], y_grid[idx]
            plt.annotate(
                annotation_text, xy=(tx, ty), xytext=(tx + 1, ty + 1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                          facecolor="white", alpha=0.8)
            )
        return line

    # --- inputs & common x-grid ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 2000, dtype=float)  # a bit denser for precise read-offs

    # Pre-index shifts and trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ]
        .set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))
    trial_to_row = {int(t): i for i, t in enumerate(trial_ids)}

    # Interpolate each trial to the grid
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Vertical quantiles: %volume at fixed dose
    q05_v = np.percentile(Y,  5, axis=0)
    q25_v = np.percentile(Y, 25, axis=0)
    q50_v = np.percentile(Y, 50, axis=0)
    q75_v = np.percentile(Y, 75, axis=0)
    q95_v = np.percentile(Y, 95, axis=0)
    mean_v = np.nanmean(Y, axis=0)

    if dvh_kind == 'cumulative':
        q05_v = _enforce_nonincreasing(q05_v)
        q25_v = _enforce_nonincreasing(q25_v)
        q50_v = _enforce_nonincreasing(q50_v)
        q75_v = _enforce_nonincreasing(q75_v)
        q95_v = _enforce_nonincreasing(q95_v)
        mean_v = _enforce_nonincreasing(mean_v)

    # Horizontal quantiles: DOSE at fixed %volume
    y_grid = np.linspace(100.0, 0.0, 1000)  # descending for cumulative
    X_of_Y = None
    if dvh_kind == 'cumulative':
        X_of_Y = np.empty((len(trial_ids), y_grid.size), dtype=float)
        for i, t in enumerate(trial_ids):
            g = trials_dict[t]
            x_t = g[x_col].to_numpy()
            y_t = g[y_col].to_numpy()
            o = np.argsort(x_t)
            x_t = x_t[o]
            # enforce monotone non-increasing for cumulative:
            y_t = np.minimum.accumulate(y_t[o])
            # invert: y -> x using the trial’s own knots
            X_of_Y[i] = np.interp(y_grid, y_t[::-1], x_t[::-1],
                                left=x_t.min(), right=x_t.max())
        q05_h = np.percentile(X_of_Y,  5, axis=0)
        q25_h = np.percentile(X_of_Y, 25, axis=0)
        q50_h = np.percentile(X_of_Y, 50, axis=0)
        q75_h = np.percentile(X_of_Y, 75, axis=0)
        q95_h = np.percentile(X_of_Y, 95, axis=0)
        mean_h = np.nanmean(X_of_Y, axis=0)

    # --- figure ---
    fig = plt.figure(figsize=(12, 8))
    handles, labels = [], []

    # Bands per mode
    if bands_mode in ("vertical", "both"):
        fill_1 = plt.fill_between(x_grid, q05_v, q25_v, color='springgreen', alpha=0.7)
        fill_2 = plt.fill_between(x_grid, q25_v, q75_v, color='dodgerblue',  alpha=0.7)
        fill_3 = plt.fill_between(x_grid, q75_v, q95_v, color='springgreen', alpha=0.7)
        plt.plot(x_grid, q05_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q25_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q75_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q95_v, linestyle=':', linewidth=1, color='black')
        handles += [fill_1, fill_2, fill_3]; labels += ['5th–25th Q','25th–75th Q','75th–95th Q']

    if dvh_kind == 'cumulative' and bands_mode in ("horizontal", "both"):
        # Thin outlines for Dx-consistent envelopes
        plt.plot(q25_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q75_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q05_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        plt.plot(q95_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        h_stub, = plt.plot([], [], linestyle='-', color='black', linewidth=1.5)
        handles += [h_stub]; labels += ['Dx-consistent quantiles']

    # Nominal line (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nominal_line = _line_with_optional_annotation(x_grid, y0, 'Nominal', 'red',
                                                      linestyle='-', linewidth=2)
    else:
        nominal_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')
    handles.append(nominal_line); labels.append('Nominal')

    # Optional summary lines (vertical sense)
    if show_median_line:
        med_line, = plt.plot(x_grid, q50_v, color='black', linewidth=2, linestyle='-',
                             label='Q50 (vertical)')
        handles.append(med_line); labels.append('Q50 (vertical)')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_v, color='orange', linewidth=2, linestyle='--',
                              label='Mean (vertical)')
        handles.append(mean_line); labels.append('Mean')

    # First-k trials after 0
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]
    annotation_lines = []
    annotation_offset_index = 0
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        # shifts
        ann_text = None
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial, ['Shift (X)','Shift (Y)','Shift (Z)']].astype(float)
            d_tot = math.sqrt(sx*sx + sy*sy + sz*sz)
            ann_text = f"({sx:.1f},{sy:.1f},{sz:.1f}), d = {d_tot:.1f} mm"

        if trial_annotation_style == 'arrow':
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                annotation_text=ann_text, target_offset=annotation_offset_index,
                linestyle='--', linewidth=1
            )
            annotation_offset_index += 1
        else:
            _line_with_optional_annotation(
                x_grid, yt, f"Trial: {trial}", random_trial_line_color,
                linestyle='--', linewidth=1
            )
            if dvh_kind == 'cumulative':
                y_target = float(np.clip(0.5*(np.nanmin(yt)+np.nanmax(yt)), 0, 100))
            else:
                y_target = float(np.nanmax(yt))
            if np.isfinite(y_target):
                idx = int(np.nanargmin(np.abs(yt - y_target)))
                plt.text(x_grid[idx], y_target, str(int(trial)), fontsize=14,
                         color='black', ha='left', va='center')
            if ann_text:
                annotation_lines.append(f"{int(trial)}: {ann_text}")

    ax = plt.gca()

    # Dx/Vy overlay markers computed from trials (optional)
    if show_dx_vy_markers and dvh_kind == 'cumulative':
        for X in dx_list:
            y_target = float(X)
            x_at_y_trials = np.array([
                np.interp(y_target, Y[i, ::-1], x_grid[::-1],
                          left=x_grid[0], right=x_grid[-1])
                for i in range(Y.shape[0])
            ])
            d_q50 = np.percentile(x_at_y_trials, 50)
            d_q25 = np.percentile(x_at_y_trials, 25)
            d_q75 = np.percentile(x_at_y_trials, 75)
            d_q05 = np.percentile(x_at_y_trials,  5)
            d_q95 = np.percentile(x_at_y_trials, 95)

            d_nom = np.nan
            if 0 in trial_to_row:
                i0 = trial_to_row[0]
                d_nom = np.interp(y_target, Y[i0, ::-1], x_grid[::-1],
                                  left=x_grid[0], right=x_grid[-1])

            ax.axhline(y_target, color='gray', lw=0.7, ls=':')
            for x_val, c, lw in [(d_q50, 'k', 2.2), (d_q25, 'k', 1.2), (d_q75, 'k', 1.2),
                                 (d_q05, 'k', 0.8), (d_q95, 'k', 0.8)]:
                ax.plot([x_val, x_val], [y_target-1.0, y_target+1.0], color=c, lw=lw)
            if np.isfinite(d_nom):
                ax.plot([d_nom, d_nom], [y_target-2.0, y_target+2.0], color='r', lw=2.2)

        for Yp in vy_list:
            thr = (Yp / 100.0) * float(ref_dose_gy)
            y_at_x_trials = np.array([
                np.interp(thr, x_grid, Y[i, :], left=100.0, right=0.0)
                for i in range(Y.shape[0])
            ])
            v_q50 = np.percentile(y_at_x_trials, 50)
            v_q25 = np.percentile(y_at_x_trials, 25)
            v_q75 = np.percentile(y_at_x_trials, 75)
            v_q05 = np.percentile(y_at_x_trials,  5)
            v_q95 = np.percentile(y_at_x_trials, 95)

            v_nom = np.nan
            if 0 in trial_to_row:
                i0 = trial_to_row[0]
                v_nom = np.interp(thr, x_grid, Y[i0, :], left=100.0, right=0.0)

            ax.axvline(thr, color='gray', lw=0.7, ls=':')
            for y_val, c, lw in [(v_q50, 'k', 2.2), (v_q25, 'k', 1.2), (v_q75, 'k', 1.2),
                                 (v_q05, 'k', 0.8), (v_q95, 'k', 0.8)]:
                ax.plot([thr-0.25, thr+0.25], [y_val, y_val], color=c, lw=lw)
            if np.isfinite(v_nom):
                ax.plot([thr-0.5, thr+0.5], [v_nom, v_nom], color='r', lw=2.2)

    # --- NEW: overlay of table metrics (if provided) ---
    if dvh_metrics_df is not None and dvh_kind == 'cumulative':
        sub = dvh_metrics_df[
            (dvh_metrics_df['Patient ID'] == patientUID) &
            (dvh_metrics_df['Struct index'] == bx_struct_ind)
        ]
        # if empty, try also matching on Bx ID
        if sub.empty and 'Bx ID' in dvh_metrics_df.columns:
            sub = dvh_metrics_df[
                (dvh_metrics_df['Patient ID'] == patientUID) &
                (dvh_metrics_df['Bx ID'] == bx_struct_roi)
            ]

        # parse rows like 'D_2', 'V_150'
        def _parse_metric_name(s):
            s = str(s).strip()
            s = s.replace('%', '')
            if s.upper().startswith('D_'):
                return ('D', int(round(float(s.split('_')[1]))))
            if s.upper().startswith('V_'):
                return ('V', int(round(float(s.split('_')[1]))))
            return (None, None)

        # marker style
        stat_style = {
            'Nominal': dict(marker='x',  c='red',   s=80,  lw=2),
            'Q50':    dict(marker='o',  c='black', s=50),
            'Q25':    dict(marker='s',  c='blue',  s=40),
            'Q75':    dict(marker='s',  c='blue',  s=40),
            'Q05':    dict(marker='^',  c='gray',  s=40),
            'Q95':    dict(marker='^',  c='gray',  s=40),
            # you can add 'Mean' etc. if you want
        }

        # Draw points
        for _, row in sub.iterrows():
            mtype, val = _parse_metric_name(row['Metric'])
            if mtype is None:
                continue

            # gather available stats for this row
            for stat_name in overlay_metrics_stats:
                if stat_name not in row or pd.isna(row[stat_name]):
                    continue
                style = stat_style.get(stat_name, dict(marker='o', c='black', s=40))
                if mtype == 'D':
                    # point at (dose=stat_value, y = x%)
                    x_pt = float(row[stat_name])
                    y_pt = float(val)
                    plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)
                else:
                    # Vy: x = (y% of ref dose), y = stat_value
                    x_pt = (float(val) / 100.0) * float(ref_dose_gy)
                    y_pt = float(row[stat_name])
                    plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)

        # Legend stubs for table metrics
        # (only once, create invisible points with the same style)
        stub_handles = []
        stub_labels  = []
        for name in ('Nominal','Q50','Q25/Q75','Q05/Q95'):
            if name == 'Nominal':
                h = plt.scatter([], [], **stat_style['Nominal'])
            elif name == 'Q50':
                h = plt.scatter([], [], **stat_style['Q50'])
            elif name == 'Q25/Q75':
                # reuse Q25 style
                h = plt.scatter([], [], **stat_style['Q25'])
            else:
                h = plt.scatter([], [], **stat_style['Q05'])
            stub_handles.append(h); stub_labels.append(f'Table {name}')
        handles += stub_handles; labels += stub_labels

    # Legend & cosmetics
    leg = ax.legend(handles, labels, loc='best', facecolor='white', prop={'size': 14})

    if trial_annotation_style != 'arrow' and len(annotation_lines) > 0:
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_frame().get_window_extent(renderer)
        inv_fig = fig.transFigure.inverted()
        frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
        fig.text(frame_x1, frame_y0 - 0.02, "\n".join(annotation_lines),
                 transform=fig.transFigure, ha='right', va='top',
                 multialignment='left', fontsize=14, color='black',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round'))

    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    svg_dose_fig_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')
    plt.close()








def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_3(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    # band options
    bands_mode="horizontal",           # 'vertical', 'horizontal', or 'both'
    show_median_line=True,
    show_mean_line=False,
    # tick/marker options
    show_computed_ticks=True,          # draw ticks from trials (what you compare against)
    show_table_markers=True,           # overlay markers from dvh_metrics_df
    dx_list=(2, 50, 98),
    vy_list=(100, 125, 150, 175, 200, 300),
    ref_dose_gy=13.5,
    # table DF (optional)
    dvh_metrics_df=None,
    overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),
    overlay_metrics_alpha=0.95,
    # styles
    dx_tick_color='black',
    vy_tick_color='tab:blue',
):
    """
    v4_3: Like v4_2 but:
      - Dx/Vy *ticks* are computed from original per-trial curves (less drift).
      - Overlays *markers* pulled from dvh_metrics_df so you can check alignment.
      - Dx visuals = black; Vy visuals = blue; Nominal = red 'x'.

    Assumes cumulative DVH if dvh_option['dvh'] == 'cumulative'.
    """

    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        return np.maximum.accumulate(y[::-1])[::-1]

    # --- inputs & common x-grid for display/bands ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 1500, dtype=float)  # dense grid for smooth lines

    # Index shifts and gather trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ]
        .set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))
    trial_to_row = {int(t): i for i, t in enumerate(trial_ids)}

    # Interpolate each trial onto x_grid ONLY for plotting bands/nominal lines
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Vertical quantiles (Vy-consistent)
    q05_v = np.percentile(Y,  5, axis=0)
    q25_v = np.percentile(Y, 25, axis=0)
    q50_v = np.percentile(Y, 50, axis=0)
    q75_v = np.percentile(Y, 75, axis=0)
    q95_v = np.percentile(Y, 95, axis=0)
    mean_v = np.nanmean(Y, axis=0)

    if dvh_kind == 'cumulative':
        q05_v = _enforce_nonincreasing(q05_v)
        q25_v = _enforce_nonincreasing(q25_v)
        q50_v = _enforce_nonincreasing(q50_v)
        q75_v = _enforce_nonincreasing(q75_v)
        q95_v = _enforce_nonincreasing(q95_v)
        mean_v = _enforce_nonincreasing(mean_v)

    # Horizontal quantiles (Dx-consistent) — invert each ORIGINAL trial (reduces drift)
    y_grid = np.linspace(100.0, 0.0, 1500)
    if dvh_kind == 'cumulative':
        X_of_Y = np.empty((len(trial_ids), y_grid.size), dtype=float)
        for i, t in enumerate(trial_ids):
            g = trials_dict[t]
            x_t = g[x_col].to_numpy()
            y_t = g[y_col].to_numpy()
            o = np.argsort(x_t)
            x_t = x_t[o]
            y_t = np.minimum.accumulate(y_t[o])
            X_of_Y[i] = np.interp(y_grid, y_t[::-1], x_t[::-1],
                                  left=x_t.min(), right=x_t.max())
        q05_h = np.percentile(X_of_Y,  5, axis=0)
        q25_h = np.percentile(X_of_Y, 25, axis=0)
        q50_h = np.percentile(X_of_Y, 50, axis=0)
        q75_h = np.percentile(X_of_Y, 75, axis=0)
        q95_h = np.percentile(X_of_Y, 95, axis=0)
        mean_h = np.nanmean(X_of_Y, axis=0)

    # --- figure ---
    fig = plt.figure(figsize=(12, 8))
    handles, labels = [], []

    # Bands per mode
    if bands_mode in ("vertical", "both"):
        f1 = plt.fill_between(x_grid, q05_v, q25_v, color='springgreen', alpha=0.7)
        f2 = plt.fill_between(x_grid, q25_v, q75_v, color='dodgerblue',  alpha=0.7)
        f3 = plt.fill_between(x_grid, q75_v, q95_v, color='springgreen', alpha=0.7)
        plt.plot(x_grid, q05_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q25_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q75_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q95_v, linestyle=':', linewidth=1, color='black')
        handles += [f1, f2, f3]; labels += ['5th–25th Q (Vy)', '25th–75th Q (Vy)', '75th–95th Q (Vy)']

    if dvh_kind == 'cumulative' and bands_mode in ("horizontal", "both"):
        plt.plot(q25_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q75_h, y_grid, linestyle='-',  linewidth=1.5, color='black', alpha=0.9)
        plt.plot(q05_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        plt.plot(q95_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        h_stub, = plt.plot([], [], linestyle='-', color='black', linewidth=1.5)
        handles += [h_stub]; labels += ['Dx-consistent quantiles']

    # Nominal line (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nom_line, = plt.plot(x_grid, y0, color='red', linewidth=2, label='Nominal')
    else:
        nom_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')
    handles.append(nom_line); labels.append('Nominal')

    # Optional vertical-sense summary lines
    if show_median_line:
        med_line, = plt.plot(x_grid, q50_v, color='black', linewidth=2, linestyle='-', label='Q50 (Vy)')
        handles.append(med_line); labels.append('Q50 (Vy)')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_v, color='orange', linewidth=2, linestyle='--', label='Mean (Vy)')
        handles.append(mean_line); labels.append('Mean (Vy)')

    # First-k trials after 0 (reference lines)
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        plt.plot(x_grid, yt, color=random_trial_line_color, linestyle='--', linewidth=1)

    ax = plt.gca()

    # -------- Computed ticks from trials (what we compare against) --------
    if show_computed_ticks and dvh_kind == 'cumulative':
        # Per-trial ORIGINAL curves for accurate read-offs:
        per_trial_xy = []
        for t in trial_ids:
            g = trials_dict[t]
            x_t = g[x_col].to_numpy()
            y_t = g[y_col].to_numpy()
            o = np.argsort(x_t)
            x_t = x_t[o]
            y_t = np.minimum.accumulate(y_t[o])
            per_trial_xy.append((x_t, y_t))

        # Dx: vertical ticks at x = quantiles of dose @ fixed y=x%
        for X in dx_list:
            y_target = float(X)
            x_samples = np.array([np.interp(y_target, y[::-1], x[::-1],
                                            left=x.min(), right=x.max())
                                  for (x, y) in per_trial_xy])
            d_q50 = np.percentile(x_samples, 50)
            d_q25 = np.percentile(x_samples, 25)
            d_q75 = np.percentile(x_samples, 75)
            d_q05 = np.percentile(x_samples,  5)
            d_q95 = np.percentile(x_samples, 95)

            # guide line
            ax.axhline(y_target, color='lightgray', lw=0.7, ls=':')
            # vertical ticks (black)
            for x_val, lw in [(d_q50, 2.2), (d_q25, 1.3), (d_q75, 1.3), (d_q05, 0.9), (d_q95, 0.9)]:
                ax.plot([x_val, x_val], [y_target-1.0, y_target+1.0],
                        color=dx_tick_color, lw=lw)

        # Vy: horizontal ticks at y = quantiles of %volume @ fixed x = thr
        for Yp in vy_list:
            thr = (Yp / 100.0) * float(ref_dose_gy)
            y_samples = np.array([np.interp(thr, x, y, left=100.0, right=0.0)
                                  for (x, y) in per_trial_xy])
            v_q50 = np.percentile(y_samples, 50)
            v_q25 = np.percentile(y_samples, 25)
            v_q75 = np.percentile(y_samples, 75)
            v_q05 = np.percentile(y_samples,  5)
            v_q95 = np.percentile(y_samples, 95)

            ax.axvline(thr, color='lightgray', lw=0.7, ls=':')
            # horizontal ticks (blue)
            for y_val, lw in [(v_q50, 2.2), (v_q25, 1.3), (v_q75, 1.3), (v_q05, 0.9), (v_q95, 0.9)]:
                ax.plot([thr-0.25, thr+0.25], [y_val, y_val],
                        color=vy_tick_color, lw=lw)

    # -------- Overlay markers from the table DF (what you’re checking) --------
    if show_table_markers and dvh_metrics_df is not None and dvh_kind == 'cumulative':
        sub = dvh_metrics_df[
            (dvh_metrics_df['Patient ID'] == patientUID) &
            (dvh_metrics_df['Struct index'] == bx_struct_ind)
        ]
        if sub.empty and 'Bx ID' in dvh_metrics_df.columns:
            sub = dvh_metrics_df[
                (dvh_metrics_df['Patient ID'] == patientUID) &
                (dvh_metrics_df['Bx ID'] == bx_struct_roi)
            ]

        def _parse_metric_name(s):
            s = str(s).strip().replace('%','')
            if s.upper().startswith('D_'):
                return ('D', int(round(float(s.split('_')[1]))))
            if s.upper().startswith('V_'):
                return ('V', int(round(float(s.split('_')[1]))))
            return (None, None)

        # marker styles (Dx black, Vy blue; Nominal red 'x')
        dx_style = {'Q50':dict(marker='o', c='black', s=45),
                    'Q25':dict(marker='s', c='black', s=40),
                    'Q75':dict(marker='s', c='black', s=40),
                    'Q05':dict(marker='^', c='black', s=40),
                    'Q95':dict(marker='^', c='black', s=40)}
        vy_style = {'Q50':dict(marker='o', c='tab:blue', s=45),
                    'Q25':dict(marker='s', c='tab:blue', s=40),
                    'Q75':dict(marker='s', c='tab:blue', s=40),
                    'Q05':dict(marker='^', c='tab:blue', s=40),
                    'Q95':dict(marker='^', c='tab:blue', s=40)}
        nominal_style = dict(marker='x', c='red', s=80, lw=2)

        # build legend stubs once
        dx_tick_stub, = plt.plot([], [], color=dx_tick_color, lw=2.2)
        vy_tick_stub, = plt.plot([], [], color=vy_tick_color, lw=2.2)
        dx_pt_stub = plt.scatter([], [], **dx_style['Q50'])
        vy_pt_stub = plt.scatter([], [], **vy_style['Q50'])
        nom_stub   = plt.scatter([], [], **nominal_style)

        for _, row in sub.iterrows():
            mtype, val = _parse_metric_name(row['Metric'])
            if mtype is None:
                continue
            for stat_name in overlay_metrics_stats:
                if stat_name not in row or pd.isna(row[stat_name]):
                    continue
                if mtype == 'D':
                    x_pt = float(row[stat_name]); y_pt = float(val)
                    style = nominal_style if stat_name=='Nominal' else dx_style.get(stat_name, dx_style['Q50'])
                    plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)
                else:
                    x_pt = (float(val)/100.0) * float(ref_dose_gy); y_pt = float(row[stat_name])
                    style = nominal_style if stat_name=='Nominal' else vy_style.get(stat_name, vy_style['Q50'])
                    plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)

        handles += [dx_tick_stub, vy_tick_stub, dx_pt_stub, vy_pt_stub, nom_stub]
        labels  += ['Dx ticks (computed)', 'Vy ticks (computed)',
                    'Dx markers (table)', 'Vy markers (table)', 'Nominal marker (table)']

    # Labels, legend, save
    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)

    leg = ax.legend(handles, labels, loc='best', facecolor='white', prop={'size': 13})
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    svg_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_path = patient_sp_output_figures_dir.joinpath(svg_name)
    plt.savefig(svg_path, format='svg')
    plt.close()










def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_4(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    # Bands
    bands_mode="horizontal",           # 'vertical', 'horizontal', or 'both'
    show_median_line=True,
    show_mean_line=False,
    # Ticks & markers (master toggles)
    show_ticks=True,                   # turn ALL computed ticks on/off
    show_markers=True,                 # turn ALL table markers on/off
    # (Optional) finer toggles if you need them later
    show_dx_ticks=True,
    show_vy_ticks=True,
    show_dx_markers=True,
    show_vy_markers=True,
    # Which metrics to annotate
    dx_list=(2, 50, 98),
    vy_list=(100, 125, 150, 175, 200, 300),
    ref_dose_gy=13.5,
    # Table DF overlay
    dvh_metrics_df=None,
    overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),
    overlay_metrics_alpha=0.95,
    # Styling for ticks/markers (paper-friendly)
    marker_color='black',
    dx_marker_shape='o',   # circle for Dx
    vy_marker_shape='s',   # square for Vy
    nominal_marker_style=dict(marker='x', c='red', s=80, lw=2),
    tick_color='black',
    dx_tick_len_y=1.0,     # vertical tick half-height (in %volume units)
    vy_tick_len_x=0.4,     # horizontal tick half-width (in Gy)
    # Horizontal envelope extrapolation policy
    limit_horizontal_to_common=False,  # if True: restrict y-grid to common range across trials (no extrapolation)
):
    """
    v4_4: Paper-friendly, minimal legend, robust inversion.
      - Horizontal (Dx-consistent) envelopes: invert ORIGINAL per-trial curves with correct extrapolation.
      - Computed Dx/Vy ticks from ORIGINAL curves (tight alignment).
      - Table markers overlaid in two black shapes: Dx=○, Vy=■ ; Nominal=red ×.
      - Master toggles: show_ticks / show_markers.
    """

    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        """Interpolate a single trial to a common x-grid (for drawing lines/bands)."""
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            y = np.minimum.accumulate(y)                 # enforce monotone ↓
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        return np.maximum.accumulate(y[::-1])[::-1]

    # --- inputs & common x-grid for drawing ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 1500, dtype=float)

    # Index shifts and gather trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ].set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))

    # Interpolate each trial onto x_grid ONLY for drawing bands/nominal/random-trial lines
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Vertical quantiles (Vy-consistent)
    q05_v = np.percentile(Y,  5, axis=0)
    q25_v = np.percentile(Y, 25, axis=0)
    q50_v = np.percentile(Y, 50, axis=0)
    q75_v = np.percentile(Y, 75, axis=0)
    q95_v = np.percentile(Y, 95, axis=0)
    mean_v = np.nanmean(Y, axis=0)

    if dvh_kind == 'cumulative':
        q05_v = _enforce_nonincreasing(q05_v)
        q25_v = _enforce_nonincreasing(q25_v)
        q50_v = _enforce_nonincreasing(q50_v)
        q75_v = _enforce_nonincreasing(q75_v)
        q95_v = _enforce_nonincreasing(q95_v)
        mean_v = _enforce_nonincreasing(mean_v)

    # Horizontal (Dx-consistent) envelopes: invert ORIGINAL per-trial curves (correct extrapolation).
    y_grid = np.linspace(100.0, 0.0, 1500)
    per_trial_xy = []  # cache original curves for ticks/markers too
    if dvh_kind == 'cumulative':
        X_of_Y = np.empty((len(trial_ids), y_grid.size), dtype=float)

        # Optionally limit y_grid to common attainable range across trials (zero extrapolation).
        if limit_horizontal_to_common:
            mins = []
            for t in trial_ids:
                g = trials_dict[t]
                xt = g[x_col].to_numpy()
                yt = g[y_col].to_numpy()
                o = np.argsort(xt)
                yt = np.minimum.accumulate(yt[o])
                mins.append(np.nanmin(yt))
            y_lo = float(np.max(mins))  # highest min
            y_grid = np.linspace(100.0, y_lo, 1500)

        for i, t in enumerate(trial_ids):
            g = trials_dict[t]
            xt = g[x_col].to_numpy()
            yt = g[y_col].to_numpy()
            o = np.argsort(xt)
            xt = xt[o]
            yt = np.minimum.accumulate(yt[o])
            per_trial_xy.append((xt, yt))

            # Correct extrapolation: for very small y, dose should clamp to xt.max()
            # (left=high dose), for very large y, clamp to xt.min() (right=low dose).
            X_of_Y[i] = np.interp(
                y_grid, yt[::-1], xt[::-1],
                left=xt.max(),   # was xt.min(): fixed
                right=xt.min()   # was xt.max(): fixed
            )

        q05_h = np.percentile(X_of_Y,  5, axis=0)
        q25_h = np.percentile(X_of_Y, 25, axis=0)
        q50_h = np.percentile(X_of_Y, 50, axis=0)
        q75_h = np.percentile(X_of_Y, 75, axis=0)
        q95_h = np.percentile(X_of_Y, 95, axis=0)
        mean_h = np.nanmean(X_of_Y, axis=0)

    # --- figure ---
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    handles, labels = [], []

    # Bands per mode (minimal legend wording)
    if bands_mode in ("vertical", "both"):
        f1 = plt.fill_between(x_grid, q05_v, q25_v, color='springgreen', alpha=0.7)
        f2 = plt.fill_between(x_grid, q25_v, q75_v, color='dodgerblue',  alpha=0.7)
        f3 = plt.fill_between(x_grid, q75_v, q95_v, color='springgreen', alpha=0.7)
        plt.plot(x_grid, q05_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q25_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q75_v, linestyle=':', linewidth=1, color='black')
        plt.plot(x_grid, q95_v, linestyle=':', linewidth=1, color='black')
        handles += [f1, f2, f3]; labels += ['5–25% (Vy)', '25–75% (Vy)', '75–95% (Vy)']

    if dvh_kind == 'cumulative' and bands_mode in ("horizontal", "both"):
        plt.plot(q25_h, y_grid, linestyle='-',  linewidth=1.6, color='black', alpha=0.95)
        plt.plot(q75_h, y_grid, linestyle='-',  linewidth=1.6, color='black', alpha=0.95)
        plt.plot(q05_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        plt.plot(q95_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        h_stub, = plt.plot([], [], linestyle='-', color='black', linewidth=1.6)
        handles += [h_stub]; labels += ['Dx-consistent envelopes']

    # Nominal line (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nom_line, = plt.plot(x_grid, y0, color='red', linewidth=2, label='Nominal')
    else:
        nom_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')
    handles.append(nom_line); labels.append('Nominal')

    # Optional vertical-sense median/mean lines
    if show_median_line:
        med_line, = plt.plot(x_grid, q50_v, color='black', linewidth=2, linestyle='-',
                             label='Median (Vy)')
        handles.append(med_line); labels.append('Median (Vy)')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_v, color='orange', linewidth=2, linestyle='--',
                              label='Mean (Vy)')
        handles.append(mean_line); labels.append('Mean (Vy)')

    # First-k trials after 0 (thin dashed refs)
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        plt.plot(x_grid, yt, color=random_trial_line_color, linestyle='--', linewidth=1)

    # -------- Computed ticks from ORIGINAL per-trial curves --------
    if show_ticks and dvh_kind == 'cumulative' and len(per_trial_xy) == len(trial_ids):
        # Dx ticks: vertical black ticks on y = x%
        if show_dx_ticks and len(dx_list) > 0:
            for X in dx_list:
                y_target = float(X)
                x_samples = np.array([
                    np.interp(
                        y_target, yt[::-1], xt[::-1],
                        left=xt.max(), right=xt.min()  # correct extrapolation
                    )
                    for (xt, yt) in per_trial_xy
                ])
                q05, q25, q50, q75, q95 = np.percentile(x_samples, [5,25,50,75,95])
                ax.axhline(y_target, color='lightgray', lw=0.7, ls=':')
                for xv, lw in [(q50, 2.2), (q25, 1.4), (q75, 1.4), (q05, 1.0), (q95, 1.0)]:
                    ax.plot([xv, xv], [y_target-dx_tick_len_y, y_target+dx_tick_len_y],
                            color=tick_color, lw=lw)

        # Vy ticks: horizontal black ticks on x = y% * ref
        if show_vy_ticks and len(vy_list) > 0:
            for Yp in vy_list:
                thr = (Yp / 100.0) * float(ref_dose_gy)
                y_samples = np.array([
                    np.interp(thr, xt, yt, left=100.0, right=0.0)
                    for (xt, yt) in per_trial_xy
                ])
                q05, q25, q50, q75, q95 = np.percentile(y_samples, [5,25,50,75,95])
                ax.axvline(thr, color='lightgray', lw=0.7, ls=':')
                for yv, lw in [(q50, 2.2), (q25, 1.4), (q75, 1.4), (q05, 1.0), (q95, 1.0)]:
                    ax.plot([thr-vy_tick_len_x, thr+vy_tick_len_x], [yv, yv],
                            color=tick_color, lw=lw)

    # -------- Overlay markers from your table DF (two shapes, all black) --------
    if show_markers and dvh_metrics_df is not None and dvh_kind == 'cumulative':
        sub = dvh_metrics_df[
            (dvh_metrics_df['Patient ID'] == patientUID) &
            (dvh_metrics_df['Struct index'] == bx_struct_ind)
        ]
        if sub.empty and 'Bx ID' in dvh_metrics_df.columns:
            sub = dvh_metrics_df[
                (dvh_metrics_df['Patient ID'] == patientUID) &
                (dvh_metrics_df['Bx ID'] == bx_struct_roi)
            ]

        def _parse_metric_name(s):
            s = str(s).strip().replace('%','')
            # accept forms like 'D_2', 'D_2% (Gy)' etc.
            s = s.replace('(Gy)','').strip()
            if s.upper().startswith('D_'):
                try:
                    return ('D', int(round(float(s.split('_')[1]))))
                except Exception:
                    return (None, None)
            if s.upper().startswith('V_'):
                try:
                    return ('V', int(round(float(s.split('_')[1]))))
                except Exception:
                    return (None, None)
            return (None, None)

        # Legend stubs for markers (minimal)
        dx_marker_stub = plt.scatter([], [], marker=dx_marker_shape, c=marker_color, s=45)
        vy_marker_stub = plt.scatter([], [], marker=vy_marker_shape, c=marker_color, s=45)
        nom_marker_stub = plt.scatter([], [], **nominal_marker_style)

        # Plot markers
        for _, row in sub.iterrows():
            mtype, val = _parse_metric_name(row['Metric'])
            if mtype is None:
                continue

            # Only plot for metrics requested in dx_list/vy_list (avoid clutter)
            if mtype == 'D' and (int(val) not in set(int(v) for v in dx_list)):
                continue
            if mtype == 'V' and (int(val) not in set(int(v) for v in vy_list)):
                continue

            for stat_name in overlay_metrics_stats:
                if stat_name not in row or pd.isna(row[stat_name]):
                    continue
                if stat_name == 'Nominal':
                    style = nominal_marker_style.copy()
                else:
                    style = dict(marker=(dx_marker_shape if mtype=='D' else vy_marker_shape),
                                 c=marker_color, s=45)

                if mtype == 'D':   # Dx: (dose, y=x%)
                    x_pt = float(row[stat_name])
                    y_pt = float(val)
                else:              # Vy: (x= y% * ref, %volume)
                    x_pt = (float(val)/100.0) * float(ref_dose_gy)
                    y_pt = float(row[stat_name])

                plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)

        handles += [dx_marker_stub, vy_marker_stub, nom_marker_stub]
        labels  += ['Dx markers (table)', 'Vy markers (table)', 'Nominal marker (table)']

    # ---- Labels, legend, save ----
    ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}', fontsize=16)
    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=16)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)

    leg = ax.legend(handles, labels, loc='best', facecolor='white', prop={'size': 13})
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    svg_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_path = patient_sp_output_figures_dir.joinpath(svg_name)
    plt.savefig(svg_path, format='svg')
    plt.close()



### helpers for 4_5 and potentially higher versions

EPS = 1e-12

def dx_from_step(xt, yt, X):
    """
    Step-consistent inversion y(x) -> x for cumulative DVH.
    xt: ascending doses
    yt: non-increasing %volume (right-continuous step at xt)
    Returns the smallest dose x such that y(x) <= X  (matches 'higher' quantile).
    """
    # -yt is non-decreasing; find first index where yt <= X
    idx = np.searchsorted(-yt, -(X + EPS), side='left')
    if idx >= len(yt):
        idx = len(yt) - 1    # y never drops below X → clamp to max dose
    return float(xt[idx])

def x_of_y_step_vectorized(xt, yt, y_vec):
    """
    Vectorized step-consistent inversion for many y values.
    Returns x(y) for each y in y_vec, using same convention as dx_from_step.
    """
    idxs = np.searchsorted(-yt, -(y_vec + EPS), side='left')
    idxs = np.clip(idxs, 0, len(yt) - 1)
    return xt[idxs]



def production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_5(
    sp_patient_all_structure_shifts_pandas_data_frame,
    cumulative_dvh_pandas_dataframe,
    patient_sp_output_figures_dir,
    patientUID,
    bx_struct_roi,
    bx_struct_ind,
    bx_ref,
    general_plot_name_string,
    num_rand_trials_to_show,
    custom_fig_title,
    trial_annotation_style='number',  # 'arrow' or 'number'
    dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
    random_trial_line_color='black',
    # Bands
    bands_mode="horizontal",           # 'vertical', 'horizontal', or 'both'
    quantile_line_style='smooth',  # 'smooth' or 'step'
    show_median_line=True,
    show_mean_line=False,
    # Ticks & markers (master toggles)
    show_ticks=True,                   # turn ALL computed ticks on/off
    show_markers=True,                 # turn ALL table markers on/off
    # (Optional) finer toggles
    show_dx_ticks=True,
    show_vy_ticks=True,
    show_dx_markers=True,
    show_vy_markers=True,
    # Which metrics to annotate
    dx_list=(2, 50, 98),
    vy_list=(100, 125, 150, 175, 200, 300),
    ref_dose_gy=13.5,
    # Table DF overlay
    dvh_metrics_df=None,
    overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),
    overlay_metrics_alpha=0.95,
    # Styling for ticks/markers (paper-friendly)
    marker_color='black',
    dx_marker_shape='o',   # circle for Dx
    vy_marker_shape='s',   # square for Vy
    nominal_marker_style=dict(marker='x', c='red', s=80, lw=2),
    tick_color='black',
    dx_tick_len_y=1.0,     # vertical tick half-height (%volume)
    vy_tick_len_x=0.4,     # horizontal tick half-width (Gy)
    # Horizontal envelope extrapolation policy
    limit_horizontal_to_common=False,  # restrict y-grid to common range (no extrapolation),
    # NEW typography controls
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 14,
    show_title: bool = False,
    title_fontsize: int = 16,
):
    """
    v4_5: v4_4 + restored trial annotations:
      - 'arrow' style: arrow callouts on the dashed random trials.
      - 'number' style: numeric labels near lines + gray annotation box listing shifts & d.
      (All the previous fixes/toggles remain.)
    """
    plt.ioff()

    def _interp_curve_linear(x, y, x_grid, dvh_kind='cumulative'):
        if len(x) == 0:
            return np.zeros_like(x_grid, dtype=float)
        o = np.argsort(x)
        x = np.asarray(x, float)[o]
        y = np.asarray(y, float)[o]
        if dvh_kind == 'cumulative':
            y = np.minimum.accumulate(y)
            y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
            y_grid = np.clip(y_grid, 0.0, 100.0)
        else:
            y_grid = np.interp(x_grid, x, y, left=0.0, right=0.0)
        return y_grid

    def _enforce_nonincreasing(y):
        return np.maximum.accumulate(y[::-1])[::-1]

    def _annotate_arrow(x_grid, y_vals, color, text, offset_index=0):
        n = len(x_grid)
        if n == 0: return
        idx = (n // 5 * offset_index) % n
        tx, ty = x_grid[idx], y_vals[idx]
        plt.annotate(
            text,
            xy=(tx, ty),
            xytext=(tx + 1.0, ty + 1.0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            fontsize=10, color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                      facecolor="white", alpha=0.8)
        )

    _is_step = (quantile_line_style.lower() == 'step')
    def _plot_qline_x(x, y, **kw):      # vertical-sense quantiles: x along dose
        if _is_step: return plt.step(x, y, where='post', **kw)
        return plt.plot(x, y, **kw)

    def _plot_qline_y(x, y, **kw):      # horizontal-sense quantiles: x is dose-of-quantile, y is %vol
        if _is_step: return plt.step(x, y, where='post', **kw)
        return plt.plot(x, y, **kw)


    # --- inputs & common x-grid for drawing ---
    df = cumulative_dvh_pandas_dataframe.copy()
    x_col = dvh_option['x-col']
    y_col = dvh_option['y-col']
    dvh_kind = dvh_option.get('dvh', 'cumulative')

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.array([x_min], dtype=float) if (not np.isfinite(x_min) or not np.isfinite(x_max) or x_max == x_min) \
             else np.linspace(x_min, x_max, 1500, dtype=float)

    # Index shifts and gather trials
    shifts_idx = (
        sp_patient_all_structure_shifts_pandas_data_frame[
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) &
            (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)
        ].set_index('Trial')[['Shift (X)','Shift (Y)','Shift (Z)']]
    )
    trials_dict = {t: g for t, g in df.groupby('MC trial', sort=False)}
    trial_ids = np.array(sorted(trials_dict.keys()))

    # Interpolate each trial onto x_grid for lines/bands
    Y = np.empty((len(trial_ids), x_grid.size), dtype=float)
    for i, t in enumerate(trial_ids):
        g = trials_dict[t]
        Y[i] = _interp_curve_linear(g[x_col].to_numpy(), g[y_col].to_numpy(),
                                    x_grid, dvh_kind=dvh_kind)

    # Vertical percentiles (Vy-consistent)
    q05_v = np.percentile(Y,  5, axis=0)
    q25_v = np.percentile(Y, 25, axis=0)
    q50_v = np.percentile(Y, 50, axis=0)
    q75_v = np.percentile(Y, 75, axis=0)
    q95_v = np.percentile(Y, 95, axis=0)
    mean_v = np.nanmean(Y, axis=0)

    if dvh_kind == 'cumulative':
        q05_v = _enforce_nonincreasing(q05_v)
        q25_v = _enforce_nonincreasing(q25_v)
        q50_v = _enforce_nonincreasing(q50_v)
        q75_v = _enforce_nonincreasing(q75_v)
        q95_v = _enforce_nonincreasing(q95_v)
        mean_v = _enforce_nonincreasing(mean_v)

    # Horizontal envelopes (Dx-consistent): invert ORIGINAL curves using step mapping
    y_grid = np.linspace(100.0, 0.0, 1500)
    per_trial_xy = []
    if dvh_kind == 'cumulative':
        if limit_horizontal_to_common:
            mins = []
            for t in trial_ids:
                xt = trials_dict[t][x_col].to_numpy()
                yt = trials_dict[t][y_col].to_numpy()
                o = np.argsort(xt); xt = xt[o]; yt = np.minimum.accumulate(yt[o])
                mins.append(np.nanmin(yt))
            y_lo = float(np.max(mins))
            y_grid = np.linspace(100.0, y_lo, 1500)

        X_of_Y = np.empty((len(trial_ids), y_grid.size), dtype=float)
        for i, t in enumerate(trial_ids):
            g = trials_dict[t]
            xt = g[x_col].to_numpy()
            yt = g[y_col].to_numpy()
            o = np.argsort(xt); xt = xt[o]; yt = np.minimum.accumulate(yt[o])
            per_trial_xy.append((xt, yt))
            X_of_Y[i, :] = x_of_y_step_vectorized(xt, yt, y_grid)

        q05_h = np.percentile(X_of_Y,  5, axis=0)
        q25_h = np.percentile(X_of_Y, 25, axis=0)
        q50_h = np.percentile(X_of_Y, 50, axis=0)
        q75_h = np.percentile(X_of_Y, 75, axis=0)
        q95_h = np.percentile(X_of_Y, 95, axis=0)


    # --- figure ---
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    handles, labels = [], []

    # Bands
    if bands_mode in ("vertical", "both"):
        f1 = plt.fill_between(x_grid, q05_v, q25_v, color='springgreen', alpha=0.5)
        f2 = plt.fill_between(x_grid, q25_v, q75_v, color='dodgerblue',  alpha=0.5)
        f3 = plt.fill_between(x_grid, q75_v, q95_v, color='springgreen', alpha=0.5)
        _plot_qline_x(x_grid, q05_v, linestyle=':', linewidth=1, color='black')
        _plot_qline_x(x_grid, q25_v, linestyle=':', linewidth=1, color='black')
        _plot_qline_x(x_grid, q75_v, linestyle=':', linewidth=1, color='black')
        _plot_qline_x(x_grid, q95_v, linestyle=':', linewidth=1, color='black')

        handles += [f1, f2, f3]; labels += ['5–25%', '25–75%', '75–95%']

    if dvh_kind == 'cumulative' and bands_mode in ("horizontal", "both"):
        _plot_qline_y(q25_h, y_grid, linestyle='-',  linewidth=1.6, color='black', alpha=0.95)
        _plot_qline_y(q75_h, y_grid, linestyle='-',  linewidth=1.6, color='black', alpha=0.95)
        _plot_qline_y(q05_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)
        _plot_qline_y(q95_h, y_grid, linestyle='--', linewidth=1.0, color='gray',  alpha=0.9)

        h_stub, = plt.plot([], [], linestyle='-', color='black', linewidth=1.6)
        handles += [h_stub]; labels += ['Dx-consistent envelopes']

    # Nominal line (trial 0)
    if 0 in trials_dict:
        g0 = trials_dict[0]
        y0 = _interp_curve_linear(g0[x_col].to_numpy(), g0[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        nom_line, = plt.plot(x_grid, y0, color='red', linewidth=2, label='Nominal')
    else:
        nom_line, = plt.plot([], [], color='red', linewidth=2, label='Nominal')
    handles.append(nom_line); labels.append('Nominal')

    # Optional vertical-sense median/mean
    if show_median_line:
        med_line, = plt.plot(x_grid, q50_v, color='black', linewidth=2, linestyle='-',
                             label='Median (Vy)')
        handles.append(med_line); labels.append('Median')
    if show_mean_line:
        mean_line, = plt.plot(x_grid, mean_v, color='orange', linewidth=2, linestyle='--',
                              label='Mean (Vy)')
        handles.append(mean_line); labels.append('Mean')

    # --- Random trials (dashed) + annotations (arrow/number) ---
    chosen = [t for t in range(1, num_rand_trials_to_show + 1) if t in trials_dict]
    annotation_lines = []
    annotation_offset_index = 0
    for trial in chosen:
        gt = trials_dict.get(trial)
        if gt is None or gt.empty:
            continue
        yt = _interp_curve_linear(gt[x_col].to_numpy(), gt[y_col].to_numpy(),
                                  x_grid, dvh_kind=dvh_kind)
        plt.plot(x_grid, yt, color=random_trial_line_color, linestyle='--', linewidth=1)

        # Pull shifts + distance
        ann_text = None
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial, ['Shift (X)','Shift (Y)','Shift (Z)']].astype(float)
            d_tot = math.sqrt(sx*sx + sy*sy + sz*sz)
            ann_text = f"({sx:.1f}, {sy:.1f}, {sz:.1f}), d = {d_tot:.1f} mm"

        if trial_annotation_style == 'arrow' and ann_text:
            _annotate_arrow(x_grid, yt, random_trial_line_color, ann_text, offset_index=annotation_offset_index)
            annotation_offset_index += 1
        elif trial_annotation_style == 'number':
            # place the trial number on the curve
            y_target = float(np.clip(0.5*(np.nanmin(yt)+np.nanmax(yt)), 0, 100)) if dvh_kind=='cumulative' else float(np.nanmax(yt))
            idx = int(np.nanargmin(np.abs(yt - y_target))) if np.isfinite(y_target) else len(x_grid)//2
            plt.text(x_grid[idx], y_target, str(int(trial)), fontsize=14,
                     color='black', ha='left', va='center')
            if ann_text:
                annotation_lines.append(f"{int(trial)}: {ann_text}")

    # -------- Computed ticks from ORIGINAL per-trial curves --------
    if show_ticks and dvh_kind == 'cumulative':
        # cache per_trial_xy if not already built
        if not per_trial_xy:
            for t in trial_ids:
                g = trials_dict[t]
                xt = g[x_col].to_numpy()
                yt = g[y_col].to_numpy()
                o = np.argsort(xt)
                xt = xt[o]
                yt = np.minimum.accumulate(yt[o])
                per_trial_xy.append((xt, yt))

        # Dx ticks: vertical black ticks on y = X%
        if show_dx_ticks and len(dx_list) > 0:
            for X in dx_list:
                y_target = float(X)
                x_samples = np.array([dx_from_step(xt, yt, y_target) for (xt, yt) in per_trial_xy])
                q05, q25, q50, q75, q95 = np.percentile(x_samples, [5,25,50,75,95])
                ax.axhline(y_target, color='lightgray', lw=0.7, ls=':')
                for xv, lw in [(q50, 2.2), (q25, 1.4), (q75, 1.4), (q05, 1.0), (q95, 1.0)]:
                    ax.plot([xv, xv], [y_target-dx_tick_len_y, y_target+dx_tick_len_y],
                            color=tick_color, lw=lw)


        # Vy ticks: horizontal black ticks on x = y% * ref
        if show_vy_ticks and len(vy_list) > 0:
            for Yp in vy_list:
                thr = (Yp / 100.0) * float(ref_dose_gy)
                def step_readout_percent_at_dose(xt, yt, thr):
                    # xt ascending doses, yt the cumulative %-volume (non-increasing)
                    # We want y(thr) = value at smallest x >= thr
                    idx = np.searchsorted(xt, thr, side='left')
                    if idx >= len(xt):      # thr above max dose → 0%
                        return 0.0
                    return float(yt[idx])   # step value at the “≥ thr” boundary

                y_samples = np.array([step_readout_percent_at_dose(xt, yt, thr)
                                    for (xt, yt) in per_trial_xy])
                q05, q25, q50, q75, q95 = np.percentile(y_samples, [5,25,50,75,95])
                ax.axvline(thr, color='lightgray', lw=0.7, ls=':')
                for yv, lw in [(q50, 2.2), (q25, 1.4), (q75, 1.4), (q05, 1.0), (q95, 1.0)]:
                    ax.plot([thr-vy_tick_len_x, thr+vy_tick_len_x], [yv, yv],
                            color=tick_color, lw=lw)

    # -------- Overlay markers from your table DF (two shapes, all black) --------
    if show_markers and dvh_metrics_df is not None and dvh_kind == 'cumulative':
        df_plot = dvh_metrics_df.copy()
        df_plot['Struct index'] = pd.to_numeric(df_plot['Struct index'], errors='coerce').astype('Int64')
        sub = df_plot[
            (df_plot['Patient ID'] == patientUID) &
            (df_plot['Struct index'] == int(bx_struct_ind)) &
            (df_plot['Bx ID'] == bx_struct_roi)
        ]


        def _parse_metric_name(s):
            s = str(s).strip().replace('%','')
            s = s.replace('(Gy)','').strip()
            if s.upper().startswith('D_'):
                try: return ('D', int(round(float(s.split('_')[1]))))
                except Exception: return (None, None)
            if s.upper().startswith('V_'):
                try: return ('V', int(round(float(s.split('_')[1]))))
                except Exception: return (None, None)
            return (None, None)

        dx_marker_stub = plt.scatter([], [], marker=dx_marker_shape, c=marker_color, s=45)
        vy_marker_stub = plt.scatter([], [], marker=vy_marker_shape, c=marker_color, s=45)
        nom_marker_stub = plt.scatter([], [], **nominal_marker_style)

        for _, row in sub.iterrows():
            mtype, val = _parse_metric_name(row['Metric'])
            if mtype is None:
                continue
            if mtype == 'D' and (int(val) not in set(int(v) for v in dx_list)): continue
            if mtype == 'V' and (int(val) not in set(int(v) for v in vy_list)): continue

            for stat_name in overlay_metrics_stats:
                if stat_name not in row or pd.isna(row[stat_name]): continue
                if stat_name == 'Nominal':
                    style = nominal_marker_style.copy()
                else:
                    style = dict(marker=(dx_marker_shape if mtype=='D' else vy_marker_shape),
                                 c=marker_color, s=45)
                if mtype == 'D':
                    x_pt = float(row[stat_name]); y_pt = float(val)
                else:
                    x_pt = (float(val)/100.0) * float(ref_dose_gy); y_pt = float(row[stat_name])
                plt.scatter([x_pt], [y_pt], alpha=overlay_metrics_alpha, **style)

        handles += [dx_marker_stub, vy_marker_stub, nom_marker_stub]
        labels  += ['Dx markers (table)', 'Vy markers (table)', 'Nominal marker (table)']

    # ---- Labels, legend ----
    if show_title and custom_fig_title not in (None, "", False):
        ax.set_title(f'{patientUID} - {bx_struct_roi} - {custom_fig_title}',
                    fontsize=title_fontsize)

    ax.set_xlabel(dvh_option['x-axis-label'], fontsize=axis_label_fontsize)
    ax.set_ylabel(dvh_option['y-axis-label'], fontsize=axis_label_fontsize)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    leg = ax.legend(handles, labels, loc='best', facecolor='white', prop={'size': 13})
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])

    # ---- Restore the gray annotation box (for 'number' style) ----
    if (trial_annotation_style == 'number') and (len(annotation_lines) > 0):
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_frame().get_window_extent(renderer)
        inv_fig = fig.transFigure.inverted()
        frame_x1, frame_y0 = inv_fig.transform((bbox_disp.x1, bbox_disp.y0))
        fig.text(
            frame_x1, frame_y0 - 0.02,
            "\n".join(annotation_lines),
            transform=fig.transFigure, ha='right', va='top',
            multialignment='left', fontsize=14, color='black',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round')
        )

    # ---- Save ----
    svg_name = f"{patientUID}-{bx_struct_roi}-{general_plot_name_string}.svg"
    svg_path = patient_sp_output_figures_dir.joinpath(svg_name)
    plt.savefig(svg_path, format='svg')
    plt.close()

















def histogram_and_fit(df, dists_to_try=None, bin_size=1, dose_col='Dose (Gy)', save_path=None, custom_name = None, xrange=None, vertical_gridlines=True, horizontal_gridlines=True):
    """
    Histograms the values of the specified dose column, fits multiple distributions,
    and annotates the plot with best fit statistics and parameters including mean,
    standard deviation, mode, and quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame with dose data.
        dose_col (str): Column name for dose values.
        save_path (str, optional): Base path to save the plot as PNG and SVG. If None, displays the plot.

    Returns:
        None
    """
    # Extract dose values
    doses = df[dose_col].dropna()
    num_trials_per_voxel_plus_nom = df['MC trial num'].max() + 1
    num_nominal_voxels_in_cohort = len(df[(df['MC trial num'] == 0)])
    num_rows_ie_num_total_points = len(doses)

    # Define distributions to fit
    distributions = {
        'truncnorm': truncnorm,
        'lognorm': lognorm,
        'gamma': gamma,
        'weibull_min': weibull_min,
        'expon': expon,
        'pareto': pareto,
        'rice': rice,
        'gengamma': gengamma,
        'gennorm': gennorm,
        'skewnorm': skewnorm,
    }
    if dists_to_try is not None:
        distributions = {k: distributions[k] for k in dists_to_try if k in distributions}



    fit_param_labels = {
        'truncnorm': ['a', 'b', 'mean', 'std'],
        'lognorm': ['shape (s)', 'loc', 'scale'],
        'gamma': ['shape (a)', 'loc', 'scale'],
        'weibull_min': ['shape (c)', 'loc', 'scale'],
        'expon': ['loc', 'scale'],
        'pareto': ['b', 'loc', 'scale'],
        'rice': ['b', 'loc', 'scale'],
        'gengamma': ['a', 'c', 'loc', 'scale'],
        'gennorm': ['beta', 'loc', 'scale'],
        'skewnorm': ['a (skew)', 'loc', 'scale'],
    }


    def perform_fits(data, distributions):
        """Fit data to multiple distributions and select the best one."""
        best_fit = None
        best_stat = float('inf')
        best_p = 0
        best_dist_name = None

        fit_results = {}
        for dist_name, dist_func in distributions.items():
            print(f'Fitting: {dist_name}')
            if dist_name == 'truncnorm':
                mean, std = np.mean(data), np.std(data)
                a, b = (0 - mean) / std, (np.inf - mean) / std
                fit = (a, b, mean, std)
                cdf_func = lambda x: truncnorm.cdf(x, a, b, loc=mean, scale=std)
            else:
                fit = dist_func.fit(data)
                cdf_func = lambda x: dist_func.cdf(x, *fit)

            # Perform KS test
            stat, p_value = kstest(data, cdf_func)
            fit_results[dist_name] = (stat, p_value, fit)

            # Update the best fit
            if stat < best_stat:
                best_stat, best_p, best_fit = stat, p_value, fit
                best_dist_name = dist_name

        return best_dist_name, best_stat, best_p, best_fit, fit_results

    # Perform fits
    best_dist, best_stat, best_p, best_fit, fit_results = perform_fits(doses, distributions)

    # Format best fit parameters with labels
    param_labels = fit_param_labels.get(best_dist, [f'param{i}' for i in range(len(best_fit))])
    fit_param_str = ', '.join(f'{label}={val:.3g}' for label, val in zip(param_labels, best_fit))


    # Generate data for the best fit distribution
    #x = np.linspace(doses.min(), doses.max(), 1000)
    x_min = xrange[0] if xrange else doses.min()
    x_max = xrange[1] if xrange else doses.max()
    x = np.linspace(x_min, x_max, 1000)
    if best_dist == 'truncnorm':
        pdf = truncnorm.pdf(x, *best_fit)
    else:
        pdf = distributions[best_dist].pdf(x, *best_fit)

    # Define bin edges based on bin size
    #bin_edges = np.arange(start=doses.min(), stop=doses.max() + bin_size, step=bin_size)
    bin_edges = np.arange(start=x_min, stop=x_max + bin_size, step=bin_size)


    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(doses, bins=bin_edges, density=True, alpha=0.6, color='blue', label='Histogram of Doses')

    # Plot the best fit distribution
    #plt.plot(x, pdf, 'r-', label=f'Best Fit: {best_dist}\nKS stat: {best_stat:.2f}, P-value: {best_p:.2e}')
    #plt.plot(x, pdf, 'r-', label=f'Best Fit: {best_dist}\nKS stat: {best_stat:.2f}')
    plt.plot(x, pdf, 'r-', label=(
        f'Best Fit: {best_dist}\n'
        f'KS stat: {best_stat:.2f}\n'
        f'{fit_param_str}'
    ))

    # Annotate fit parameters, mean, and standard deviation
    mean = np.mean(doses)
    std = np.std(doses)
    min_dose = np.min(doses)
    max_dose = np.max(doses)
    # Calculate argmax from the density plot
    argmax_x = x[np.argmax(pdf)]

    # Calculate quantiles before annotations
    quantiles = np.percentile(doses, [5, 25, 50, 75, 95])
    quantile_labels = ['Q5', 'Q25', 'Median', 'Q75', 'Q95']
    quantiles_text = ', '.join([f'{label}: {q:.2f}' for label, q in zip(quantile_labels, quantiles)])

    # Generate annotation text
    stats_text = f'Mean: {mean:.2f}, Std: {std:.2f}, Mode: {argmax_x:.2f}, Min: {min_dose:.2f}, Max: {max_dose:.2f}\n{quantiles_text}\n$N_{{trials/voxel}}$: {num_trials_per_voxel_plus_nom}, $N_{{voxels}}$: {num_nominal_voxels_in_cohort}, $N_{{data points}}$: {num_rows_ie_num_total_points}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))



    # Add quantiles
    quantiles = np.percentile(doses, [5, 25, 50, 75, 95])
    quantile_colors = ['red', 'blue', 'black', 'blue', 'red']
    quantile_labels = ['Q5', 'Q25', 'Q50', 'Q75', 'Q95']
    for q, color, label in zip(quantiles, quantile_colors, quantile_labels):
        plt.axvline(q, color=color, linestyle='--', label=f'{label}', linewidth=0.75)



    # Add labels and legend
    plt.xlabel(dose_col)
    plt.ylabel('Density')
    plt.title('Histogram of Dose with Best Fit Distribution')

    if xrange:
        plt.xlim(xrange)

    # add vertical grid lines
    if vertical_gridlines:
        plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    # add horizontal grid lines
    if horizontal_gridlines:
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)


    plt.legend()

    # Save or display the plot
    if save_path:
        if custom_name == None:
            png_path = save_path.joinpath("all_voxels_dose_histogram_fit.png")
            svg_path = save_path.joinpath("all_voxels_dose_histogram_fit.svg")
        else:
            png_path = save_path.joinpath(f"{custom_name}.png")
            svg_path = save_path.joinpath(f"{custom_name}.svg")
        plt.savefig(png_path, format='png', bbox_inches='tight')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Plot saved as PNG: {png_path}")
        print(f"Plot saved as SVG: {svg_path}")
        plt.close()
    else:
        plt.show()


def histogram_and_fit_v2(
    df,
    dists_to_try=None,
    bin_size=1,
    dose_col='Dose (Gy)',
    save_path=None,
    custom_name=None,
    xrange=None,
    vertical_gridlines=True,
    horizontal_gridlines=True,
    # --- NEW options below ---
    quantity_tex=None,                 # e.g. r"D_{b,v}^{(t)}" or r"G_{b,v}^{(t)}"
    quantity_unit_tex=None,            # e.g. "Gy" or r"Gy mm$^{-1}$"
    ylabel="Density",
    title=None,
    show_stats_box=True,
    show_minor_ticks=False,
    vertical_minor_gridlines=False,
    horizontal_minor_gridlines=False,
    legend_fontsize=11,
    stats_box_fontsize=11,
    param_precision=3,
    stat_precision=1,
    # NEW:
    axis_label_fontsize=16,
    tick_label_fontsize=14,
):
    """
    Histograms the values of the specified column, fits multiple distributions,
    and annotates the plot with fit statistics and parameters.

    Selection rule:
        - Among the candidate distributions, the "best" is chosen by
          minimum AIC = 2*k - 2*logL (log-likelihood-based).
        - KS statistic and p-value for the best family are still computed
          and reported for descriptive purposes.

    New arguments:
        quantity_tex: LaTeX symbol for the x-variable (no $...$), e.g. r"D_{b,v}^{(t)}".
        quantity_unit_tex: Unit text appended in parentheses, e.g. "Gy" or r"Gy mm$^{-1}$".
        ylabel: y-axis label (default "Density").
        title: optional custom title (None -> generic).
        show_stats_box: toggle the statistics annotation box.
        show_minor_ticks: turn minor ticks on for both axes.
        vertical_minor_gridlines / horizontal_minor_gridlines:
            draw minor grids along x / y.
        legend_fontsize: fontsize for legend.
        stats_box_fontsize: fontsize for stats box text.
        param_precision: significant digits for fitted distribution parameters.
        stat_precision: decimal places for mean/std/quantiles, etc.
        axis_label_fontsize: fontsize for x/y axis labels.
        tick_label_fontsize: fontsize for tick labels on both axes.

    """
    # Extract quantity values
    doses = df[dose_col].dropna().values
    num_trials_per_voxel_plus_nom = df['MC trial num'].max() + 1
    num_nominal_voxels_in_cohort = len(df[df['MC trial num'] == 0])
    num_rows_ie_num_total_points = len(doses)

    # NEW: number of biopsies (unique Patient ID, Bx index combinations), if available
    if {"Patient ID", "Bx index"}.issubset(df.columns):
        num_biopsies = (
            df.dropna(subset=["Patient ID", "Bx index"])
              [["Patient ID", "Bx index"]]
              .drop_duplicates()
              .shape[0]
        )
    else:
        num_biopsies = None

    # --- distributions to try ---
    distributions = {
        'truncnorm': truncnorm,
        'lognorm': lognorm,
        'gamma': gamma,
        'weibull_min': weibull_min,
        'expon': expon,
        'pareto': pareto,
        'rice': rice,
        'gengamma': gengamma,
        'gennorm': gennorm,
        'skewnorm': skewnorm,
    }
    if dists_to_try is not None:
        distributions = {k: distributions[k] for k in dists_to_try if k in distributions}

    fit_param_labels = {
        'truncnorm': ['a', 'b', 'mean', 'std'],
        #'lognorm':   ['shape (s)', 'loc', 'scale'],
        "lognorm": [r"$\kappa$", r"$\xi$", r"$\beta$"],
        'gamma':     ['shape (a)', 'loc', 'scale'],
        'weibull_min': ['shape (c)', 'loc', 'scale'],
        'expon':     ['loc', 'scale'],
        'pareto':    ['b', 'loc', 'scale'],
        'rice':      ['b', 'loc', 'scale'],
        'gengamma':  ['a', 'c', 'loc', 'scale'],
        'gennorm':   ['beta', 'loc', 'scale'],
        'skewnorm':  ['a (skew)', 'loc', 'scale'],
    }

    def perform_fits(data, distributions):
        """
        Fit data to multiple distributions, compute log-likelihood and AIC,
        as well as KS stat and p-value, and select the best by minimum AIC.
        """
        best_fit = None
        best_AIC = float('inf')
        best_dist_name = None
        best_ks_stat = None
        best_ks_p = None

        fit_results = {}

        for dist_name, dist_func in distributions.items():
            print(f'Fitting: {dist_name}')
            try:
                if dist_name == 'truncnorm':
                    # Use a data-driven truncnorm around the observed support
                    mean_, std_ = np.mean(data), np.std(data)
                    # Lower bound at 0, upper at +inf (or could clip by data range)
                    a, b = (0 - mean_) / std_, (np.inf - mean_) / std_
                    fit = (a, b, mean_, std_)
                    frozen = truncnorm(*fit)
                    # CDF and logpdf from the frozen distribution
                    cdf_func = lambda x, frozen=frozen: frozen.cdf(x)
                    logpdf_vals = frozen.logpdf(data)
                else:
                    # Generic SciPy fit
                    fit = dist_func.fit(data)
                    frozen = dist_func(*fit)
                    cdf_func = lambda x, frozen=frozen: frozen.cdf(x)
                    logpdf_vals = frozen.logpdf(data)

                # Log-likelihood, ignoring any -inf / NaN contributions
                logpdf_vals = np.asarray(logpdf_vals)
                finite_mask = np.isfinite(logpdf_vals)
                if not np.any(finite_mask):
                    print(f"  Skipping {dist_name}: no finite logpdf values.")
                    continue
                logL = np.sum(logpdf_vals[finite_mask])

                # Number of parameters
                k_params = len(fit)
                AIC = 2 * k_params - 2 * logL

                # KS test (descriptive; uses full sample size)
                ks_stat, ks_p = kstest(data, cdf_func)

                fit_results[dist_name] = {
                    "fit": fit,
                    "logL": logL,
                    "AIC": AIC,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                }

                # Select best by minimum AIC
                if AIC < best_AIC:
                    best_AIC = AIC
                    best_fit = fit
                    best_dist_name = dist_name
                    best_ks_stat = ks_stat
                    best_ks_p = ks_p

            except Exception as e:
                print(f"  Error fitting {dist_name}: {e}")
                continue

        return best_dist_name, best_fit, best_AIC, best_ks_stat, best_ks_p, fit_results

    # --- Perform fits ---
    best_dist, best_fit, best_AIC, best_stat, best_p, fit_results = perform_fits(doses, distributions)

    if best_dist is None or best_fit is None:
        raise RuntimeError("No distribution could be successfully fit to the data.")

    # Format best fit parameters with labels
    param_labels = fit_param_labels.get(best_dist, [f'param{i}' for i in range(len(best_fit))])
    fit_param_str = ', '.join(
        f'{label}={val:.{param_precision}g}'
        for label, val in zip(param_labels, best_fit)
    )

    # --- Generate data for the best fit distribution ---
    x_min = xrange[0] if xrange is not None else doses.min()
    x_max = xrange[1] if xrange is not None else doses.max()
    x = np.linspace(x_min, x_max, 1000)

    if best_dist == 'truncnorm':
        pdf = truncnorm.pdf(x, *best_fit)
    else:
        pdf = distributions[best_dist].pdf(x, *best_fit)

    # --- Bin edges based on bin size ---
    bin_edges = np.arange(start=x_min, stop=x_max + bin_size, step=bin_size)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(
        doses,
        bins=bin_edges,
        density=True,
        alpha=0.7,
        color='C0',
        edgecolor='white',
        label='Histogram of values',
    )

    # Best-fit PDF (label includes AIC + KS)
    # In this version the AIC is huge because N_samples is huge ~4 million for 27 biopsies so we dont include it in the annotation and dont report it, just use it to find the best fit
    """
    ax.plot(
        x, pdf,
        'r-',
        linewidth=2.0,
        label=(
            f'Best fit (min AIC): {best_dist}\n'
            f'AIC: {best_AIC:.1f}, KS: {best_stat:.2f}, p={best_p:.1e}\n'
            f'{fit_param_str}'
        ),
    )
    """
    
    ax.plot(
        x, pdf,
        'r-',
        linewidth=2.0,
        label=(
            f'Best fit (AIC): {best_dist}\n'
            f'KS: {best_stat:.2f}\n'
            f'{fit_param_str}'
        ),
    )

    # --- KDE mode computation ---
    kde = gaussian_kde(doses)
    xs = np.linspace(doses.min(), doses.max(), 1000)  # or pass kde_grid_size in
    kde_pdf = kde(xs)
    kde_mode = xs[np.argmax(kde_pdf)]



    # --- Basic stats and quantiles (single computation used everywhere) ---
    mean = np.mean(doses)
    std = np.std(doses)
    min_val = np.min(doses)
    max_val = np.max(doses)

    # argmax from PDF
    argmax_x = x[np.argmax(pdf)]

    quantile_probs = [5, 25, 50, 75, 95]
    quantiles = np.percentile(doses, quantile_probs)

    # For vertical lines & legend
    quantile_labels_tex = [r"$Q_{5}$", r"$Q_{25}$", r"$Q_{50}$", r"$Q_{75}$", r"$Q_{95}$"]
    quantile_colors = ['red', 'blue', 'black', 'blue', 'red']

    # For the annotation box: one unit tag per group of stats
    if quantity_unit_tex:
        # e.g. " (Gy)" or " (Gy mm$^{-1}$)"
        unit_group_suffix = f" ({quantity_unit_tex})"
    else:
        unit_group_suffix = ""



    # --- Optional stats annotation box ---
    if show_stats_box:
        # Line 1: central tendency + range, with one unit tag at the end
        line1 = (
            r'$\mu, \sigma, \mathrm{mode}, \mathrm{min}, \mathrm{max}$'
            f'{unit_group_suffix}: '
            f'{mean:.{stat_precision}f}, '
            f'{std:.{stat_precision}f}, '
            #f'{argmax_x:.{stat_precision}f}, ' # swapped out for kde based mode to match csv file 
            f'{kde_mode:.{stat_precision}f}, '
            f'{min_val:.{stat_precision}f}, '
            f'{max_val:.{stat_precision}f}'
        )

        # Line 2: quantiles, again with a single unit tag
        line2 = (
            r'$Q_{5}, Q_{25}, Q_{50}, Q_{75}, Q_{95}$'
            f'{unit_group_suffix}: '
            + ', '.join(f'{q:.{stat_precision}f}' for q in quantiles)
        )

        # Line 3: counts (dimensionless)
        counts_parts = []
        if num_biopsies is not None:
            counts_parts.append(r'$N_{\mathrm{biopsies}}$: ' + f'{num_biopsies}')
        counts_parts.append(r'$N_{\mathrm{trials/voxel}}$: ' + f'{num_trials_per_voxel_plus_nom}')
        counts_parts.append(r'$N_{\mathrm{voxels}}$: ' + f'{num_nominal_voxels_in_cohort}')
        counts_parts.append(r'$N_{\mathrm{data\ points}}$: ' + f'{num_rows_ie_num_total_points}')
        line3 = ', '.join(counts_parts)

        stats_text = line1 + "\n" + line2 + "\n" + line3

        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=stats_box_fontsize,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3',
                      edgecolor='black',
                      facecolor='white',
                      alpha=0.9),
        )


    # --- Vertical quantile lines (colors + labels consistent with legend) ---
    for q, color, label_tex in zip(quantiles, quantile_colors, quantile_labels_tex):
        ax.axvline(
            q, color=color, linestyle='--', linewidth=0.9,
            label=label_tex
        )

    # --- Labels and title ---
    if quantity_tex is not None:
        # LaTeX-style axis label, e.g. "$D_{b,v}^{(t)}$ (Gy)"
        xlabel = rf"${quantity_tex}$"
        if quantity_unit_tex:
            xlabel += f" ({quantity_unit_tex})"
    else:
        xlabel = dose_col  # fallback

    # Axis labels with configurable fontsize
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # Tick label font size
    ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)

    # Title (leave fontsize tied to axis_label_fontsize, or change if you prefer)
    if title is None:
        pass
    else:
        ax.set_title(title, fontsize=axis_label_fontsize)


    # x-range
    if xrange is not None:
        ax.set_xlim(xrange)

    # Ticks & grids
    if show_minor_ticks or vertical_minor_gridlines or horizontal_minor_gridlines:
        ax.minorticks_on()

    if vertical_gridlines:
        ax.grid(axis='x', which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    if horizontal_gridlines:
        ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    if vertical_minor_gridlines:
        ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    if horizontal_minor_gridlines:
        ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

    # Slightly heavier axes spines for print clarity
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Legend
    ax.legend(fontsize=legend_fontsize)

    # Major ticks: outward, visible on all sides
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",   # <-- outward toward the labels
        length=6,
        width=1.0,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )

    # Minor ticks: also outward if enabled
    if show_minor_ticks:
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3,
            width=0.8,
            bottom=True,
            top=True,
            left=True,
            right=True,
        )



    fig.tight_layout()

    # Save or show
    if save_path:
        if custom_name is None:
            base = "all_voxels_dose_histogram_fit"
        else:
            base = custom_name

        png_path = save_path.joinpath(f"{base}.png")
        svg_path = save_path.joinpath(f"{base}.svg")
        pdf_path = save_path.joinpath(f"{base}.pdf")   # <--- NEW

        # high-quality exports
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
        fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=600)  # <--- NEW

        print(f"Plot saved as PNG: {png_path}")
        print(f"Plot saved as SVG: {svg_path}")
        print(f"Plot saved as PDF: {pdf_path}")         # <--- NEW
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)





def plot_eff_size_heatmaps(eff_size_df, 
                           patient_id_col, 
                           bx_index_col, 
                           bx_id_col, 
                           eff_size_col, 
                           eff_size_type, 
                           save_dir=None,
                           save_name_base=None,
                            annotation_info=None,
                            vmin=None,
                            vmax=None): 
    """
    Create and optionally save heatmaps for effect size values for all unique combinations 
    of Patient ID, Bx index, and Bx ID.
    
    Args:
        eff_size_df (pd.DataFrame): DataFrame containing effect size values with columns for:
                                    Patient ID, Bx index, Bx ID, Voxel 1, Voxel 2, Effect Size.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID.
        eff_size_col (str): Column name for the effect size values.
        save_dir (str, optional): Directory to save heatmaps. If None, only displays the heatmaps.
    
    Returns:
        None
    """

    # Define default color scale ranges for different effect size types
    default_vmin_vmax = {
        "cohen": (-2, 2),
        "hedges": (-2, 2),
        "glass": (-2, 2),
        "mean_diff": (None, None),       # will compute max from data
        "auc": (0, 1),
        "cles": (0, 1),
    }



    # Ensure the save directory exists if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Group by Patient ID, Bx index, and Bx ID
    grouped = eff_size_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    
    for (patient_id, bx_index, bx_id), group in grouped:

        # Calculate default annotation info if not provided
        if annotation_info is None:
            annotation_info = {}

        # Add biopsy and patient info 
        annotation_info["Patient ID"] = patient_id
        annotation_info["Biopsy ID"] = bx_id
        annotation_info["Num Observations"] = group["Num Observations Voxel 1"].iloc[0]  # Assuming all rows in group have the same value
        annotation_info["Effect Size Type"] = group["Effect Size Type"].iloc[0]

        # Pivot the DataFrame to create a matrix for heatmap plotting
        heatmap_data = group.pivot(index="Voxel 1", columns="Voxel 2", values=eff_size_col)

        # Get non-NaN values for robust min/max if needed
        non_nan_values = heatmap_data.values[~np.isnan(heatmap_data.values)]

        # Apply defaults if vmin or vmax not provided
        if vmin is None or vmax is None:
            effect_type_key = eff_size_type.lower().replace(" ", "_")
            vmin_default, vmax_default = default_vmin_vmax.get(effect_type_key, (None, None))
            
            if vmin is None:
                if vmin_default is not None:
                    vmin = vmin_default
                elif len(non_nan_values) > 0:
                    vmin = float(np.min(non_nan_values))
                else:
                    vmin = 0

            if vmax is None:
                if vmax_default is not None:
                    vmax = vmax_default
                elif len(non_nan_values) > 0:
                    vmax = float(np.max(non_nan_values))
                else:
                    vmax = 1

        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,  # Display annotations in cells
            fmt=".2f",   # Use decimal format for effect size
            cmap="coolwarm",
            cbar_kws={'label': "Effect Size"},
            annot_kws={'size': 8},  # Font size for annotations
            vmin=vmin,  # Adjust range for effect size
            vmax=vmax
        )
        title = (f"Effect Size Heatmap for {patient_id_col}: {patient_id}, "
                 f"{bx_index_col}: {bx_index}, {bx_id_col}: {bx_id}")
        plt.title(title)
        plt.xlabel("Voxel 2")
        plt.ylabel("Voxel 1")
        plt.tight_layout()

        # Create annotation text block
        annotation_text = '\n'.join(f"{key}: {val}" for key, val in annotation_info.items())
        plt.gca().text(
            0.02, 0.1, annotation_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.9)
        )
        
        if save_dir is not None:
            # Save the plot as PNG and SVG
            base_file_name = (f"EffSize_{eff_size_type}_Heatmap_{patient_id_col}_{patient_id}_{save_name_base}"
                              f"{bx_index_col}_{bx_index}_{bx_id_col}_{bx_id}")
            png_path = os.path.join(save_dir, f"{base_file_name}.png")
            svg_path = os.path.join(save_dir, f"{base_file_name}.svg")
            
            plt.savefig(png_path)
            plt.savefig(svg_path)
            print(f"Saved heatmap as PNG: {png_path}")
            print(f"Saved heatmap as SVG: {svg_path}")
        
        # Close the plot to avoid displaying in non-interactive environments
        plt.close()


def plot_diff_stats_heatmaps_with_std(
    diff_stats_df,
    patient_id_col,
    bx_index_col,
    bx_id_col,
    mean_col: str = "mean_diff",
    std_col: str = None,
    save_dir: str = None,
    save_name_base: str = "dose",
    annotation_info: dict = None,
    vmin: float = None,
    vmax: float = None
):
    """
    One heatmap per biopsy, showing mean_diff (± std_diff if provided).

    Args:
        diff_stats_df (pd.DataFrame):
            Must contain [patient_id_col, bx_index_col, bx_id_col,
            'voxel1','voxel2', mean_col, optional std_col].
        patient_id_col, bx_index_col, bx_id_col : str
            Grouping columns in diff_stats_df.
        mean_col : str
            Column name for the mean differences.
        std_col : str, optional
            Column name for standard deviations. If given, cells show "mean±std".
        save_dir : str, optional
            Directory to save PNG/SVG. If None, plots are shown.
        annotation_info : dict, optional
            Extra key→value lines drawn on each plot.
        vmin, vmax : float, optional
            Global color‐scale limits for mean_diff.
    """
    
    plt.ioff()  # turn interactive mode off, stops figures from displaying immediately


    # defaults for different metrics (only mean_diff here)
    default_vmin_vmax = {
        "mean_diff": (None, None),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    grouped = diff_stats_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    for (pid, bxi, bxid), grp in grouped:
        # build per-plot annotations
        ann = {} if annotation_info is None else dict(annotation_info)
        ann["Patient ID"] = pid
        ann["Biopsy ID"]  = bxid
        ann["Voxel Pairs"] = grp.shape[0]

        # pivot into square matrix
        voxels = sorted(set(grp["voxel1"]) | set(grp["voxel2"]))
        mean_mat = (
            grp.pivot(index="voxel1", columns="voxel2", values=mean_col)
               .reindex(index=voxels, columns=voxels)
        )

        # optional std mat
        if std_col:
            std_mat = (
                grp.pivot(index="voxel1", columns="voxel2", values=std_col)
                   .reindex(index=voxels, columns=voxels)
            )

        # determine local vmin/vmax
        arr = mean_mat.values
        nonnan = arr[~np.isnan(arr)]
        key = mean_col.lower()
        dvmin, dvmax = default_vmin_vmax.get(key, (None, None))
        lvmin = vmin if vmin is not None else (dvmin if dvmin is not None else (np.min(nonnan) if nonnan.size else 0))
        lvmax = vmax if vmax is not None else (dvmax if dvmax is not None else (np.max(nonnan) if nonnan.size else 1))

        # build annotation matrix or fallback
        if std_col:
            annot = np.full(mean_mat.shape, "", dtype=object)
            for i in range(len(voxels)):
                for j in range(len(voxels)):
                    m = mean_mat.iat[i, j]
                    s = std_mat.iat[i, j]
                    if np.isnan(m):
                        continue
                    if np.isnan(s):
                        annot[i, j] = f"{m:.2f}"
                    else:
                        annot[i, j] = f"{m:.2f}\n±\n{s:.2f}"
            fmt = ""
        else:
            annot = True
            fmt = ".2f"

        # plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mean_mat,
            annot=annot,
            fmt=fmt,
            cmap="coolwarm",
            vmin=lvmin,
            vmax=lvmax,
            cbar_kws={'label': mean_col},
            annot_kws={
            "ha": "center",
            "va": "center",
            "multialignment": "center",
            "size": 9
            },
            linewidths=0.3,
            linecolor="white",
            square=True,
        )
        plt.title(f"Heatmap of {mean_col}" + (f" ±{std_col}" if std_col else ""))
        plt.xlabel("voxel2")
        plt.ylabel("voxel1")

        # draw annotations block
        text = "\n".join(f"{k}: {v}" for k, v in ann.items())
        plt.gca().text(
            0.02, 0.02, text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )
        plt.tight_layout()

        if save_dir:
            fname = f"MeanDiff_HM_P{pid}_Bx{bxi}_{bxid}" + (f"_pm_{std_col}" if std_col else "") + f"_{save_name_base}"
            for ext in (".png", ".svg"):
                plt.savefig(os.path.join(save_dir, fname + ext), bbox_inches="tight")
            plt.close()
            print(f"Saved heatmap for PID={pid}, BxID={bxid}: {fname}.png/.svg")
        else:
            plt.show()










def get_contrasting_color(val, vmin, vmax, cmap):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(val))
    # perceived luminance (simple formula)
    r, g, b = rgba[:3]
    luminance = 0.299*r + 0.587*g + 0.114*b
    return "black" if luminance > 0.5 else "white"



def plot_diff_stats_heatmap_upper_lower(
        upper_df,                      # e.g., dose
        lower_df,                      # e.g., dose gradient
        patient_id_col,
        bx_index_col,
        bx_id_col,
        upper_mean_col: str = "mean_diff",
        upper_std_col: str | None = "std_diff",
        lower_mean_col: str = "mean_diff",
        lower_std_col: str | None = "std_diff",
        save_dir: str | None = None,
        save_name_base: str = "dose_upper__grad_lower",
        annotation_info: dict | None = None,
        # global fallback limits (used only if per-triangle limits not provided)
        vmin: float | None = None,
        vmax: float | None = None,
        # OPTIONAL: per-triangle limits (take precedence if provided)
        vmin_upper: float | None = None,
        vmax_upper: float | None = None,
        vmin_lower: float | None = None,
        vmax_lower: float | None = None,
        # typography
        tick_label_fontsize: int = 9,
        axis_label_fontsize: int = 11,
        cbar_tick_fontsize: int = 9,
        cbar_label_fontsize: int = 11,
        cbar_label_upper: str = "Mean (Upper)",
        cbar_label_lower: str = "Mean (Lower)",
        # title & corner annotation
        show_title: bool = True,
        show_annotation_box: bool = True,
        # per cell annotation fontsize
        cell_annot_fontsize: int = 8,
        cell_value_decimals: int = 1,          # <--- NEW
        y_axis_upper_tri_label = r"Voxel $i$",
        x_axis_upper_tri_label = r"Voxel $j$",
        y_axis_lower_tri_label = r"Voxel $j$",
        x_axis_lower_tri_label = r"Voxel $i$",
        color_bar_positions: str = "top_bottom",  # "left_right" or "top_bottom"
        # NEW: colormap and padding
        cmap: str = "coolwarm",
        cbar_pad: float = 0.6,
        cbar_label_pad: float = 8.0,
    ):

    """
    One heatmap per biopsy: upper triangle from `upper_df`, lower triangle from `lower_df` (mirrored).
    Cells display mean (± std if provided). **Independent** colorbars for upper/lower.

    Each DataFrame must contain:
      [patient_id_col, bx_index_col, bx_id_col, 'voxel1','voxel2', mean_col, optional std_col]
    """

    plt.ioff()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # group by biopsy keys
    group_keys = [patient_id_col, bx_index_col, bx_id_col]
    upper_groups = upper_df.groupby(group_keys)
    lower_groups = lower_df.groupby(group_keys)

    # iterate over union of keys present in either df
    all_keys = set(upper_groups.groups.keys()) | set(lower_groups.groups.keys())

    for key in sorted(all_keys):
        pid, bxi, bxid = key

        grp_upper = upper_groups.get_group(key) if key in upper_groups.groups else None
        grp_lower = lower_groups.get_group(key) if key in lower_groups.groups else None
        if grp_upper is None and grp_lower is None:
            continue

        # corner annotation text
        ann = {} if annotation_info is None else dict(annotation_info)
        ann["Patient ID"] = pid
        ann["Biopsy index"] = bxi
        ann["Biopsy ID"]  = bxid
        ann["Upper"] = f"{upper_mean_col}" + (f" ± {upper_std_col}" if upper_std_col else "")
        ann["Lower"] = f"{lower_mean_col}" + (f" ± {lower_std_col}" if lower_std_col else "")

        # unified voxel list across both groups
        voxels = set()
        if grp_upper is not None:
            voxels |= (set(grp_upper["voxel1"]) | set(grp_upper["voxel2"]))
        if grp_lower is not None:
            voxels |= (set(grp_lower["voxel1"]) | set(grp_lower["voxel2"]))
        voxels = sorted(voxels)
        n = len(voxels)

        # pivot helpers
        def pivot_val(grp, col):
            if grp is None or col is None:
                return None
            return (grp.pivot(index="voxel1", columns="voxel2", values=col)
                        .reindex(index=voxels, columns=voxels))

        UM_df = pivot_val(grp_upper, upper_mean_col)
        if UM_df is None:
            UM_df = pd.DataFrame(index=voxels, columns=voxels, dtype=float)

        LM_df = pivot_val(grp_lower, lower_mean_col)
        if LM_df is None:
            LM_df = pd.DataFrame(index=voxels, columns=voxels, dtype=float)

        US_df = pivot_val(grp_upper, upper_std_col) if upper_std_col else None
        LS_df = pivot_val(grp_lower, lower_std_col) if lower_std_col else None

        UM = UM_df.values
        LM = LM_df.values

        # build display matrices for **independent** colorbars
        upper_display = np.full((n, n), np.nan, dtype=float)
        lower_display = np.full((n, n), np.nan, dtype=float)

        for i in range(n):
            for j in range(n):
                if i < j:
                    upper_display[i, j] = UM[i, j]
                elif i > j:
                    lower_display[i, j] = LM[j, i]  # mirror into lower
                else:
                    # diagonal: prefer upper; else lower
                    if np.isfinite(UM[i, j]):
                        upper_display[i, j] = UM[i, j]
                    elif np.isfinite(LM[i, j]):
                        lower_display[i, j] = LM[i, j]

        # per-triangle vmin/vmax with optional global fallback
        def finite_vals(arr):
            v = arr[np.isfinite(arr)]
            return v if v.size else np.array([])

        if vmin_upper is None or vmax_upper is None:
            up_vals = finite_vals(upper_display)
        if vmin_lower is None or vmax_lower is None:
            lo_vals = finite_vals(lower_display)

        if vmin_upper is None:
            vmin_upper = float(up_vals.min()) if up_vals.size else (vmin if vmin is not None else 0.0)
        if vmax_upper is None:
            vmax_upper = float(up_vals.max()) if up_vals.size else (vmax if vmax is not None else 1.0)
        if vmin_lower is None:
            vmin_lower = float(lo_vals.min()) if lo_vals.size else (vmin if vmin is not None else 0.0)
        if vmax_lower is None:
            vmax_lower = float(lo_vals.max()) if lo_vals.size else (vmax if vmax is not None else 1.0)

        # annotation strings (mean ± std) placed manually with contrasting colors
        annot = np.full((n, n), "", dtype=object)
        US = US_df.values if US_df is not None else None
        LS = LS_df.values if LS_df is not None else None
        fmt = f"{{:.{int(cell_value_decimals)}f}}"

        for i in range(n):
            for j in range(n):
                if i < j:
                    m = UM[i, j]
                    if not np.isfinite(m): 
                        continue
                    s = US[i, j] if US is not None else np.nan
                    annot[i, j] = (
                        fmt.format(m)
                        if not np.isfinite(s)
                        else f"{fmt.format(m)}\n±\n{fmt.format(s)}"
                    )
                elif i > j:
                    m = LM[j, i]
                    if not np.isfinite(m):
                        continue
                    s = LS[j, i] if LS is not None else np.nan
                    annot[i, j] = (
                        fmt.format(m)
                        if not np.isfinite(s)
                        else f"{fmt.format(m)}\n±\n{fmt.format(s)}"
                    )
                else:
                    if np.isfinite(UM[i, j]):
                        s = US[i, j] if US is not None else np.nan
                        m = UM[i, j]
                    elif np.isfinite(LM[i, j]):
                        s = LS[i, j] if LS is not None else np.nan
                        m = LM[i, j]
                    else:
                        continue

                    annot[i, j] = (
                        fmt.format(m)
                        if not np.isfinite(s)
                        else f"{fmt.format(m)}\n±\n{fmt.format(s)}"
                    )


        # ===== Plot: two overlaid heatmaps + two independent colorbars =====
        fig, ax = plt.subplots(figsize=(10, 8))

        # lower first
        hm_lower = sns.heatmap(
            lower_display,
            cmap=cmap,
            vmin=vmin_lower, vmax=vmax_lower,
            mask=~np.isfinite(lower_display),
            cbar=False,
            linewidths=0.3, linecolor="white",
            square=True, ax=ax,
        )
        # upper second
        hm_upper = sns.heatmap(
            upper_display,
            cmap=cmap,
            vmin=vmin_upper, vmax=vmax_upper,
            mask=~np.isfinite(upper_display),
            cbar=False,
            linewidths=0.3, linecolor="white",
            square=True, ax=ax,
        )

        # manual annotations with contrasting color per triangle
        cmap_up = plt.get_cmap(cmap)
        cmap_lo = plt.get_cmap(cmap)
        for i in range(n):
            for j in range(n):
                txt = annot[i, j]
                if not txt:
                    continue
                if i < j or (i == j and np.isfinite(upper_display[i, j])):
                    color = get_contrasting_color(upper_display[i, j], vmin_upper, vmax_upper, cmap_up)
                else:
                    color = get_contrasting_color(lower_display[i, j], vmin_lower, vmax_lower, cmap_lo)
                ax.text(
                    j + 0.5, i + 0.5, txt,
                    ha="center", va="center",
                    fontsize=cell_annot_fontsize,   # 🔹 now configurable
                    color=color
                )


        # Optional title
        if show_title:
            ax.set_title(
                f"Upper: {upper_mean_col}" + (f" ± {upper_std_col}" if upper_std_col else "") +
                "   |   " +
                f"Lower: {lower_mean_col}" + (f" ± {lower_std_col}" if lower_std_col else "")
            )

        # Bottom/Left = LOWER triangle semantics
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
        ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)
        ax.set_xlabel(x_axis_lower_tri_label, fontsize=axis_label_fontsize)
        ax.set_ylabel(y_axis_lower_tri_label, fontsize=axis_label_fontsize)
        ax.tick_params(axis="x", bottom=True, top=False, direction="out", length=4, width=1, color="black")
        ax.tick_params(axis="y", left=True, right=False, direction="out", length=4, width=1, color="black")

        # Top/Right = UPPER triangle semantics
        top_ax = ax.secondary_xaxis('top')
        top_ax.set_xticks(ax.get_xticks())
        top_ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
        top_ax.set_xlabel(x_axis_upper_tri_label, fontsize=axis_label_fontsize)
        top_ax.tick_params(axis="x", top=True, direction="out", length=4, width=1, color="black")

        right_ax = ax.secondary_yaxis('right')
        right_ax.set_yticks(ax.get_yticks())
        right_ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)
        right_ax.set_ylabel(y_axis_upper_tri_label, fontsize=axis_label_fontsize)
        right_ax.tick_params(axis="y", right=True, direction="out", length=4, width=1, color="black")

        # independent colorbars (LEFT=lower, RIGHT=upper)
        if color_bar_positions == "left_right":    
            divider = make_axes_locatable(ax)
            cax_left  = divider.append_axes("left",  size="4%", pad=cbar_pad)
            cax_right = divider.append_axes("right", size="4%", pad=cbar_pad)

            sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap=cmap)
            sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap=cmap)
            sm_lower.set_array([])
            sm_upper.set_array([])

            cbar_lower = plt.colorbar(sm_lower, cax=cax_left)
            cbar_upper = plt.colorbar(sm_upper, cax=cax_right)
            cbar_lower.ax.tick_params(labelsize=cbar_tick_fontsize)
            cbar_upper.ax.tick_params(labelsize=cbar_tick_fontsize)
            cbar_lower.set_label(
                cbar_label_lower,
                fontsize=cbar_label_fontsize,
                labelpad=cbar_label_pad,
            )
            cbar_upper.set_label(
                cbar_label_upper,
                fontsize=cbar_label_fontsize,
                labelpad=cbar_label_pad,
            )

            cax_left.yaxis.set_ticks_position('left')
            cax_left.yaxis.set_label_position('left')
            cax_right.yaxis.set_ticks_position('right')
            cax_right.yaxis.set_label_position('right')
        

        # independent colorbars (BOTTOM=lower, TOP=upper)
        if color_bar_positions == "top_bottom":
            divider = make_axes_locatable(ax)
            cax_bottom = divider.append_axes("bottom", size="4%", pad=cbar_pad)
            cax_top    = divider.append_axes("top",    size="4%", pad=cbar_pad)

            sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap=cmap)
            sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap=cmap)
            sm_lower.set_array([])
            sm_upper.set_array([])

            # lower triangle colorbar - below the matrix
            cbar_lower = plt.colorbar(sm_lower, cax=cax_bottom, orientation="horizontal")
            # upper triangle colorbar - above the matrix
            cbar_upper = plt.colorbar(sm_upper, cax=cax_top,    orientation="horizontal")

            # tick sizes
            cbar_lower.ax.tick_params(axis="x", labelsize=cbar_tick_fontsize)
            cbar_upper.ax.tick_params(axis="x", labelsize=cbar_tick_fontsize)

            # labels
            cbar_lower.set_label(
                cbar_label_lower,
                fontsize=cbar_label_fontsize,
                labelpad=cbar_label_pad,
            )
            cbar_upper.set_label(
                cbar_label_upper,
                fontsize=cbar_label_fontsize,
                labelpad=cbar_label_pad,
            )


            # put ticks and label on the top edge for the upper bar
            cax_top.xaxis.set_ticks_position("top")
            cax_top.xaxis.set_label_position("top")
            # bottom bar keeps default bottom ticks and label

        # corner annotation
        if show_annotation_box and ann:
            ax.text(
                0.02, 0.02,
                "\n".join(f"{k}: {v}" for k, v in ann.items()),
                transform=ax.transAxes,
                fontsize=10,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
            )

        plt.tight_layout()

        # save/show
        if save_dir:
            fname = f"DualTri_HM_P{pid}_Bx{bxi}_{bxid}_{save_name_base}"
            plt.savefig(os.path.join(save_dir, fname + ".png"), bbox_inches="tight", dpi=300)
            plt.savefig(os.path.join(save_dir, fname + ".svg"), bbox_inches="tight")
            plt.close(fig)
            print(f"Saved dual-triangle heatmap for PID={pid}, BxIndex={bxi}, BxID={bxid}: {fname}.png/.svg")
        else:
            plt.show()
            plt.close(fig)




































def plot_cohort_eff_size_heatmap(
    eff_size_df,
    eff_size_col,
    eff_size_type,
    save_path_base=None,
    annotation_info=None,
    aggregate_abs=False,
    vmin=None,
    vmax=None
):
    """
    Plot a single heatmap of the mean effect size values for voxel pairs,
    averaged across all biopsies and patients.

    Args:
        eff_size_df (pd.DataFrame): DataFrame with 'Voxel 1', 'Voxel 2', effect size columns,
                                    and optionally biopsy/patient identifiers.
        eff_size_col (str): Column name for effect size values.
        eff_size_type (str): Type of effect size for labeling.
        save_path_base (str, optional): If provided, saves PNG and SVG files to this base path.
        annotation_info (dict, optional): Dictionary of annotation key-value pairs to display.

    Returns:
        None
    """

    default_vmin_vmax_signed = {
    "cohen": (-2, 2),
    "hedges": (-2, 2),
    "glass": (-2, 2),
    "mean_diff": (None, None),  # let data decide
    }

    default_vmin_vmax_abs = {
        "cohen": (0, 2),
        "hedges": (0, 2),
        "glass": (0, 2),
        "mean_diff": (0, None),  # auto upper limit
    }

    default_vmin_vmax_unsigned = {
        "auc": (0, 1),
        "cles": (0, 1),
    }

    # Determine effect type key
    effect_type_key = eff_size_type.lower().replace(" ", "_")

    # Choose dictionary based on aggregate_abs
    if effect_type_key in default_vmin_vmax_unsigned:
        vmin_default, vmax_default = default_vmin_vmax_unsigned[effect_type_key]
    elif aggregate_abs:
        vmin_default, vmax_default = default_vmin_vmax_abs.get(effect_type_key, (None, None))
    else:
        vmin_default, vmax_default = default_vmin_vmax_signed.get(effect_type_key, (None, None))





    # Calculate default annotation info if not provided
    if annotation_info is None:
        annotation_info = {}

    # Add biopsy and patient count if available
    if any("Bx" in col for col in eff_size_df.columns):
        bx_cols = [col for col in eff_size_df.columns if "Bx" in col]
        annotation_info["# Biopsies"] = eff_size_df[bx_cols].drop_duplicates().shape[0]
    if "Patient ID" in eff_size_df.columns:
        annotation_info["# Patients"] = eff_size_df["Patient ID"].nunique()

    # If aggregate_abs is True, take absolute values of effect sizes
    if aggregate_abs:
        eff_size_df = eff_size_df.copy()
        eff_size_df[eff_size_col] = eff_size_df[eff_size_col].abs()


    # Compute mean effect size across all biopsies for each voxel pair
    grouped = eff_size_df.groupby(["Voxel 1", "Voxel 2"])[eff_size_col].mean().reset_index()

    # Pivot to matrix form for heatmap
    heatmap_data = grouped.pivot(index="Voxel 1", columns="Voxel 2", values=eff_size_col)

    # Ensure matrix is square
    all_voxels = sorted(set(heatmap_data.index).union(set(heatmap_data.columns)))
    heatmap_data = heatmap_data.reindex(index=all_voxels, columns=all_voxels)


    # Get non-NaN values for fallback
    non_nan_values = heatmap_data.values[~np.isnan(heatmap_data.values)]

    if vmin is None:
        vmin = vmin_default if vmin_default is not None else float(np.min(non_nan_values)) if len(non_nan_values) > 0 else 0

    if vmax is None:
        vmax = vmax_default if vmax_default is not None else float(np.max(non_nan_values)) if len(non_nan_values) > 0 else 1



    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': "Mean Effect Size"},
        annot_kws={'size': 7},
        vmin=vmin,
        vmax=vmax
    )
    plt.title(f"Mean Effect Size Heatmap ({eff_size_type})")
    plt.xlabel("Voxel 2")
    plt.ylabel("Voxel 1")
    plt.tight_layout()

    # Create annotation text block
    annotation_text = '\n'.join(f"{key}: {val}" for key, val in annotation_info.items())
    plt.gca().text(
        0.02, 0.05, annotation_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.9)
    )

    # Save or display
    if save_path_base is not None:
        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
        save_path = save_path_base.joinpath(f"{eff_size_type}_heatmap_abs_is-{str(aggregate_abs).lower()}")
        png_path = f"{save_path}.png"
        svg_path = f"{save_path}.svg"
        plt.savefig(png_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"Saved heatmap: {png_path} and {svg_path}")
    else:
        plt.show()

    plt.close()


def plot_cohort_eff_size_heatmap_separate_triangles(
    eff_size_df: pd.DataFrame,
    eff_size_col: str,
    eff_size_type: str,
    save_path_base=None,
    annotation_info: dict = None,
    aggregate_abs: bool = False,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot two separate heatmaps for cohort‐level voxel pair effect sizes:
      1) Upper‐triangle mean effect sizes.
      2) Lower‐triangle sample counts (mirrored from the upper).
    """

    # 1) Build annotation text
    if annotation_info is None:
        annotation_info = {}
    if any("Bx" in c for c in eff_size_df.columns):
        bx_cols = [c for c in eff_size_df.columns if "Bx" in c]
        annotation_info["# Biopsies"] = eff_size_df[bx_cols].drop_duplicates().shape[0]
    if "Patient ID" in eff_size_df.columns:
        annotation_info["# Patients"] = eff_size_df["Patient ID"].nunique()
    annotation_text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())

    # 2) Absolute effect sizes?
    if aggregate_abs:
        eff_size_df = eff_size_df.copy()
        eff_size_df[eff_size_col] = eff_size_df[eff_size_col].abs()

    # 3) Compute stats
    stats = (
        eff_size_df
        .groupby(["Voxel 1","Voxel 2"])[eff_size_col]
        .agg(mean="mean", count="count")
        .reset_index()
    )

    # 4) Pivot
    mean_mat  = stats.pivot(index="Voxel 1",  columns="Voxel 2", values="mean")
    count_mat = stats.pivot(index="Voxel 1",  columns="Voxel 2", values="count")
    voxels = sorted(set(mean_mat.index) | set(mean_mat.columns))
    mean_mat  = mean_mat.reindex(index=voxels, columns=voxels)
    count_mat = count_mat.reindex(index=voxels, columns=voxels)

    # 5) Mirror counts to lower triangle
    #    combine_first fills NaNs in count_mat with the transposed values
    count_mat = count_mat.combine_first(count_mat.T)
    #    fill diagonal with zero (or whatever you prefer)
    np.fill_diagonal(count_mat.values, 0)

    # 6) Determine vmin/vmax for mean heatmap
    arr = mean_mat.values
    nonan = arr[~np.isnan(arr)]
    if vmin is None:
        vmin = float(np.min(nonan)) if nonan.size else 0
    if vmax is None:
        vmax = float(np.max(nonan)) if nonan.size else 1

    # make save dir
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)

    # --- Plot 1: Upper triangle of means ---
    mask_lower = np.tril(np.ones_like(mean_mat, dtype=bool))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mean_mat,
        mask=mask_lower,
        annot=True, fmt=".2f",
        cmap="coolwarm",
        vmin=vmin, vmax=vmax,
        cbar_kws={"label": "Mean Effect Size"},
        annot_kws={"size": 6}
    )
    plt.title(f"Upper Triangle: Mean Effect Size ({eff_size_type})")
    plt.xlabel("Voxel 2"); plt.ylabel("Voxel 1")
    plt.text(
        0.02, 0.02, annotation_text,
        transform=plt.gca().transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8)
    )
    plt.tight_layout()
    if save_path_base:
        p_up = save_path_base.joinpath(f"{eff_size_type}_mean_upper_abs-{aggregate_abs}")
        plt.savefig(p_up.with_suffix(".png"), bbox_inches="tight")
        plt.savefig(p_up.with_suffix(".svg"), bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # --- Plot 2: Lower triangle of counts ---
    mask_upper = np.triu(np.ones_like(count_mat, dtype=bool))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        count_mat,
        mask=mask_upper,
        annot=True, fmt=".0f",
        cmap="Greens",
        cbar_kws={"label": "Sample Count"},
        annot_kws={"size": 6, "color": "black"}
    )
    plt.title(f"Lower Triangle: Sample Count ({eff_size_type})")
    plt.xlabel("Voxel 2"); plt.ylabel("Voxel 1")
    plt.text(
        0.02, 0.02, annotation_text,
        transform=plt.gca().transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8)
    )
    plt.tight_layout()
    if save_path_base:
        p_lo = save_path_base.joinpath(f"{eff_size_type}_count_lower_abs-{aggregate_abs}")
        plt.savefig(p_lo.with_suffix(".png"), bbox_inches="tight")
        plt.savefig(p_lo.with_suffix(".svg"), bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_cohort_eff_size_heatmap_combined(
    eff_size_df: pd.DataFrame,
    eff_size_col: str,
    eff_size_type: str,
    save_path_base=None,
    annotation_info: dict = None,
    aggregate_abs: bool = False,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot a single heatmap with:
      - Upper triangle: mean effect sizes
      - Lower triangle: sample counts

    All other behavior (data prep, abs‐aggregation, vmin/vmax defaults, annotations)
    is identical to your separate‐triangles version.
    """

    # 1) Build annotation text
    if annotation_info is None:
        annotation_info = {}
    if any("Bx" in c for c in eff_size_df.columns):
        bx_cols = [c for c in eff_size_df.columns if "Bx" in c]
        annotation_info["# Biopsies"] = eff_size_df[bx_cols].drop_duplicates().shape[0]
    if "Patient ID" in eff_size_df.columns:
        annotation_info["# Patients"] = eff_size_df["Patient ID"].nunique()
    annotation_text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())

    # 2) Optionally take absolute values
    if aggregate_abs:
        eff_size_df = eff_size_df.copy()
        eff_size_df[eff_size_col] = eff_size_df[eff_size_col].abs()

    # 3) Compute mean & count
    stats = (
        eff_size_df
        .groupby(["Voxel 1","Voxel 2"])[eff_size_col]
        .agg(mean="mean", count="count")
        .reset_index()
    )

    # 4) Pivot to square matrices
    mean_mat  = stats.pivot(index="Voxel 1",  columns="Voxel 2", values="mean")
    count_mat = stats.pivot(index="Voxel 1",  columns="Voxel 2", values="count")
    voxels = sorted(set(mean_mat.index) | set(mean_mat.columns))
    mean_mat  = mean_mat.reindex(index=voxels, columns=voxels)
    count_mat = count_mat.reindex(index=voxels, columns=voxels)

    # 5) Mirror counts into lower triangle
    count_mat = count_mat.combine_first(count_mat.T)
    np.fill_diagonal(count_mat.values, 0)

    # 6) Determine vmin/vmax for mean
    arr = mean_mat.values
    nonan = arr[~np.isnan(arr)]
    if vmin is None:
        vmin = float(np.min(nonan)) if nonan.size else 0
    if vmax is None:
        vmax = float(np.max(nonan)) if nonan.size else 1

    # 7) Make save dir if needed
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)

    # 8) Plot combined heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # mask lower to show only upper triangle of means
    mask_lower = np.tril(np.ones_like(mean_mat, dtype=bool))
    sns.heatmap(
        mean_mat,
        mask=mask_lower,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Mean Effect Size"},
        annot_kws={"size": 7},
        ax=ax
    )

    # annotate lower triangle with counts
    n = len(voxels)
    for i in range(n):
        for j in range(n):
            if i > j:
                c = count_mat.iat[i, j]
                if not pd.isna(c):
                    ax.text(
                        j + 0.5, i + 0.5,
                        f"n={int(c)}",
                        ha="center", va="center",
                        fontsize=7, color="black"
                    )

    ax.set_title(f"Effect Size & Sample Count ({eff_size_type})")
    ax.set_xlabel("Voxel 2")
    ax.set_ylabel("Voxel 1")
    ax.text(
        0.02, 0.02, annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    if save_path_base:
        p = save_path_base.joinpath(f"{eff_size_type}_combined_heatmap_abs-{aggregate_abs}")
        plt.savefig(p.with_suffix(".png"), bbox_inches="tight")
        plt.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        print(f"Saved combined heatmap to {p}.png/.svg")
    else:
        plt.show()
    plt.close()



def plot_cohort_eff_size_heatmap_boxed_counts(
    eff_size_df: pd.DataFrame,
    eff_size_col: str,
    eff_size_type: str,
    save_path_base=None,
    save_name_base = None,
    annotation_info: dict = None,
    aggregate_abs: bool = False,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot a single heatmap with:
      - Upper triangle: mean effect sizes
      - Lower triangle: sample counts, with separation lines at count-change boundaries

    Args:
      eff_size_df: DataFrame with ['Voxel 1','Voxel 2', effect size, optional Bx/Patient].
      eff_size_col: Effect size column name.
      eff_size_type: Label descriptor (e.g. 'cohen', 'mean_diff').
      save_path_base: pathlib.Path or str directory to save files (PNG/SVG). If None, show plot.
      annotation_info: Dict of extra annotations.
      aggregate_abs: Bool, use abs(effect size).
      vmin, vmax: Optional color limits for mean heatmap.
    """
    plt.ioff()

    # 1) Build annotation text
    if annotation_info is None:
        annotation_info = {}
    if any("Bx" in c for c in eff_size_df.columns):
        bx_cols = [c for c in eff_size_df.columns if "Bx" in c]
        annotation_info["# Biopsies"] = eff_size_df[bx_cols].drop_duplicates().shape[0]
    if "Patient ID" in eff_size_df.columns:
        annotation_info["# Patients"] = eff_size_df["Patient ID"].nunique()
    annotation_text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())

    # 2) Optionally take absolute effect sizes
    if aggregate_abs:
        eff_size_df = eff_size_df.copy()
        eff_size_df[eff_size_col] = eff_size_df[eff_size_col].abs()

    # 3) Compute mean & count
    stats = (
        eff_size_df
        .groupby(["Voxel 1", "Voxel 2"])[eff_size_col]
        .agg(mean="mean", count="count")
        .reset_index()
    )

    # 4) Pivot to square matrices
    mean_mat = stats.pivot(index="Voxel 1", columns="Voxel 2", values="mean")
    count_mat = stats.pivot(index="Voxel 1", columns="Voxel 2", values="count")
    voxels = sorted(set(mean_mat.index) | set(mean_mat.columns))
    mean_mat = mean_mat.reindex(index=voxels, columns=voxels)
    count_mat = count_mat.reindex(index=voxels, columns=voxels)

    # 5) Mirror counts and zero diagonal
    count_mat = count_mat.combine_first(count_mat.T)
    np.fill_diagonal(count_mat.values, 0)

    # 6) Determine vmin/vmax for mean
    arr = mean_mat.values
    nonan = arr[~np.isnan(arr)]
    if vmin is None:
        vmin = float(np.min(nonan)) if nonan.size else 0
    if vmax is None:
        vmax = float(np.max(nonan)) if nonan.size else 1

    # 7) Ensure output directory
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)

    # 8) Plot combined heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_lower = np.tril(np.ones_like(mean_mat, dtype=bool))
    sns.heatmap(
        mean_mat,
        mask=mask_lower,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Mean Effect Size"},
        annot_kws={"size": 7},
        ax=ax
    )

    # 9) Draw separation lines for each unique count region
    n = len(voxels)
    mask_lt = np.tril(np.ones((n, n), dtype=bool), k=-1)
    unique_counts = np.unique(count_mat.values[mask_lt])
    unique_counts = unique_counts[~np.isnan(unique_counts)]

    for c in unique_counts:
        region = (count_mat.values == c) & mask_lt
        coords = np.argwhere(region)
        if coords.size == 0:
            continue
        i_min, j_min = coords.min(axis=0)
        i_max, j_max = coords.max(axis=0)
        # lower separation: horizontal line at bottom of region
        ax.hlines(
            y=i_max + 1,
            xmin=j_min,
            xmax=j_max + 2,
            colors='black', linewidth=1.5
        )
        # upper separation: vertical line at right of region
        ax.vlines(
            x=j_max + 2,
            ymin=0,
            ymax=i_max + 1,
            colors='black', linewidth=1.5
        )
        # label count once at lower region center
        cx = j_min + (j_max - j_min + 1) / 2
        cy = i_min + (i_max - i_min + 1) / 2
        ax.text(
            cx, cy, f"n={int(c)}",
            ha='center', va='center', fontsize=8, color='black'
        )

    # Final touches
    ax.set_title(f"Effect Size & Sample Count ({eff_size_type})")
    ax.set_xlabel("Voxel 2")
    ax.set_ylabel("Voxel 1")
    ax.text(
        0.02, 0.02, annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
    )
    plt.tight_layout()

    # 10) Save or show
    if save_path_base:
        p = save_path_base.joinpath(f"{eff_size_type}_boxed_counts_abs-{aggregate_abs}_{save_name_base}")
        plt.savefig(p.with_suffix(".png"), bbox_inches="tight")
        plt.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        print(f"Saved boxed-counts heatmap to {p}.png/.svg")
    else:
        plt.show()
    plt.close()


def plot_cohort_eff_size_heatmap_boxed_counts_and_std(
    diff_stats_df: pd.DataFrame,
    eff_size_col: str,
    eff_size_type: str,
    save_path_base=None,
    save_name_base= None,
    annotation_info: dict = None,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot a single cohort‐level heatmap from your per‐biopsy mean-difference table:
      - Upper triangle: mean(eff_size_col) ± std(eff_size_col) across biopsies
      - Lower triangle: number of biopsies (count), boxed by count‐change regions

    Args:
      diff_stats_df : DataFrame with columns ['Patient ID','Bx index','Bx ID',
                      'voxel1','voxel2', eff_size_col, …].
      eff_size_col  : e.g. "mean_diff" or "mean_diff_abs"
      eff_size_type : label for title/filenames (e.g. "Mean Difference")
      save_path_base: dir (or Path) to save PNG/SVG; if None, show the plot.
      annotation_info: extra key→value lines to draw on the plot.
      aggregate_abs : if True, take abs(eff_size_col) before aggregating.
      vmin, vmax    : optional color scale limits for the means.
    """
    plt.ioff()  # turn interactive mode off, stops figures from displaying immediately

    # 1) Build annotation text
    annotation_info = annotation_info or {}
    if "Patient ID" in diff_stats_df.columns:
        annotation_info["# Patients"] = diff_stats_df["Patient ID"].nunique()
    if "Bx ID" in diff_stats_df.columns:
        annotation_info["# Biopsies"] = diff_stats_df[["Patient ID","Bx ID"]].drop_duplicates().shape[0]
    annotation_text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())

    # 2) Optionally take absolute effect sizes
    df = diff_stats_df

    # 3) Compute cohort‐level mean, std, and count per voxel‐pair
    stats = (
        df
        .groupby(["voxel1", "voxel2"])[eff_size_col]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    # 4) Pivot to square matrices
    voxels   = sorted(set(stats["voxel1"]) | set(stats["voxel2"]))
    mean_mat = (
        stats
        .pivot(index="voxel1", columns="voxel2", values="mean")
        .reindex(index=voxels, columns=voxels)
    )
    std_mat  = (
        stats
        .pivot(index="voxel1", columns="voxel2", values="std")
        .reindex(index=voxels, columns=voxels)
    )
    count_mat= (
        stats
        .pivot(index="voxel1", columns="voxel2", values="count")
        .reindex(index=voxels, columns=voxels)
    )

    # 5) Mirror counts & zero diagonal
    count_mat = count_mat.combine_first(count_mat.T)
    np.fill_diagonal(count_mat.values, 0)

    # 6) Determine vmin/vmax for mean
    arr = mean_mat.values
    nonan = arr[~np.isnan(arr)]
    if vmin is None:
        vmin = float(np.min(nonan)) if nonan.size else 0.0
    if vmax is None:
        vmax = float(np.max(nonan)) if nonan.size else 1.0

    # 7) Ensure output directory
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)

    # 8) Build annotation matrix for upper triangle: "mean±std"
    n = len(voxels)
    mask_lower = np.tril(np.ones((n, n), dtype=bool))
    annot = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(n):
            if mask_lower[i, j]:
                continue
            m = mean_mat.iat[i, j]
            s = std_mat.iat[i, j]
            if np.isnan(m):
                continue
            annot[i, j] = (
                f"{m:.2f}"
                if np.isnan(s)
                else f"{m:.2f}\n±\n{s:.2f}"
            )
    fmt = ""  # pre-built strings

    # 9) Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mean_mat,
        mask=mask_lower,
        annot=annot,
        fmt=fmt,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": eff_size_col},
        annot_kws={
            "ha": "center",
            "va": "center",
            "multialignment": "center",
            "size": 6
        },
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        square=True,
    )

    # 10) Draw boxed‐count regions on lower triangle
    mask_lt = np.tril(np.ones((n, n), dtype=bool), k=-1)
    unique_counts = np.unique(count_mat.values[mask_lt])
    unique_counts = unique_counts[~np.isnan(unique_counts)]
    for c in unique_counts:
        region = (count_mat.values == c) & mask_lt
        coords = np.argwhere(region)
        if coords.size == 0:
            continue
        i_min, j_min = coords.min(axis=0)
        i_max, j_max = coords.max(axis=0)
        # horizontal line below the block
        ax.hlines(y=i_max + 1, xmin=j_min, xmax=j_max + 2,
                  colors="black", linewidth=1.5)
        # vertical line right of the block
        ax.vlines(x=j_max + 2, ymin=0, ymax=i_max + 1,
                  colors="black", linewidth=1.5)
        # label count in center
        cx = j_min + (j_max - j_min + 1) / 2
        cy = i_min + (i_max - i_min + 1) / 2
        ax.text(cx, cy, f"n={int(c)}",
                ha="center", va="center",
                fontsize=8, color="black")

    # Final touches
    ax.set_title(f"{eff_size_type} & Sample Count")
    ax.set_xlabel("voxel2")
    ax.set_ylabel("voxel1")
    if annotation_text:
        ax.text(
            0.02, 0.02, annotation_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3",
                      edgecolor="gray", facecolor="white", alpha=0.9)
        )
    plt.tight_layout()

    # 11) Save or show
    if save_path_base:
        base = os.path.join(
            save_path_base,
            f"{eff_size_type}_boxed_counts_abs-{save_name_base}"
        )
        plt.savefig(base + ".png", bbox_inches="tight",dpi=300)
        plt.savefig(base + ".svg", bbox_inches="tight")
        print(f"Saved boxed-counts heatmap to {base}.png/.svg")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)








def plot_cohort_eff_size_dualtri_mean_std(
    upper_df: pd.DataFrame,             # e.g., dose
    lower_df: pd.DataFrame,
    eff_size_col: str,                  # e.g., "mean_diff"
    eff_size_type_upper: str,           # e.g., "Dose mean±std"
    eff_size_type_lower: str,           # e.g., "Dose-Gradient mean±std"
    save_path_base=None,
    save_name_base=None,
    annotation_info: dict | None = None,
    aggregate_abs: bool = False,
    # global fallback limits (used only if per-triangle limits not provided)
    vmin: float | None = None,
    vmax: float | None = None,
    # OPTIONAL: per-triangle limits (take precedence if provided)
    vmin_upper: float | None = None,
    vmax_upper: float | None = None,
    vmin_lower: float | None = None,
    vmax_lower: float | None = None,
    # counts overlay
    show_counts_boxes: bool = True,
    counts_source: str = "lower",       # "lower" or "upper"
    # typography controls
    tick_label_fontsize: int = 9,
    axis_label_fontsize: int = 11,
    cbar_tick_fontsize: int = 9,
    cbar_label_fontsize: int = 11,
    cbar_label_upper: str = "Mean Difference (Gy, Upper)",
    cbar_label_lower: str = "Mean Difference (Gy, Lower)",
    # title
    show_title: bool = True,
    # n= annotation fontsize
    n_label_fontsize: int = 8,
    # annotation box
    show_annotation_box: bool = True,

):
    """
    Cohort-level single heatmap:
      - Upper triangle : cohort mean±std from `upper_df` (not mirrored)
      - Lower triangle : cohort mean±std from `lower_df` (MIRRORED into lower half)
      - Two colorbars with independent normalization:
            LEFT  -> lower triangle values (gradient)
            RIGHT -> upper triangle values (dose)
      - Optional count boxes on lower triangle, with 'n=' text centered inside a small white inset
        in the diagonal cell at the lower-right of each box.

    Axes/ticks (hugging triangles; labels switched per your request):
      Bottom X : Voxel 1 (lower triangle)
      Left   Y : Voxel 2 (lower triangle)
      Top    X : Voxel 1 (upper triangle)
      Right  Y : Voxel 2 (upper triangle)
    """


    plt.ioff()

    # ---- copies / abs
    up = upper_df.copy()
    lo = lower_df.copy()
    if aggregate_abs:
        up[eff_size_col] = up[eff_size_col].abs()
        lo[eff_size_col] = lo[eff_size_col].abs()

    # ---- annotation block
    annotation_text = ""
    if show_annotation_box:
        ann = {} if annotation_info is None else dict(annotation_info)
        for df, tag in ((up, "Upper"), (lo, "Lower")):
            if "Patient ID" in df.columns:
                ann[f"# Patients ({tag})"] = df["Patient ID"].nunique()
            if set(["Patient ID","Bx ID"]).issubset(df.columns):
                ann[f"# Biopsies ({tag})"] = df[["Patient ID","Bx ID"]].drop_duplicates().shape[0]
        annotation_text = "\n".join(f"{k}: {v}" for k, v in ann.items())


    # ---- aggregate mean/std per voxel pair
    def agg_mean_std(df):
        return (
            df.groupby(["voxel1","voxel2"])[eff_size_col]
              .agg(mean="mean", std="std")
              .reset_index()
        )
    up_stats = agg_mean_std(up)
    lo_stats = agg_mean_std(lo)

    # ---- optional counts
    def agg_count(df):
        return (
            df.groupby(["voxel1","voxel2"])[eff_size_col]
              .agg(count="count")
              .reset_index()
        )
    count_mat = None
    if show_counts_boxes:
        src = up if counts_source == "upper" else lo
        cnt_stats = agg_count(src)
    else:
        cnt_stats = None

    # ---- unified voxel set
    voxels = sorted(
        set(up_stats["voxel1"]).union(up_stats["voxel2"])
        .union(lo_stats["voxel1"]).union(lo_stats["voxel2"])
    )

    # ---- pivots
    def pivots(stats_df):
        mean_mat = (stats_df.pivot(index="voxel1", columns="voxel2", values="mean")
                    .reindex(index=voxels, columns=voxels))
        std_mat  = (stats_df.pivot(index="voxel1", columns="voxel2", values="std")
                    .reindex(index=voxels, columns=voxels))
        return mean_mat, std_mat

    up_mean, up_std = pivots(up_stats)
    lo_mean, lo_std = pivots(lo_stats)

    if cnt_stats is not None:
        count_mat = (cnt_stats.pivot(index="voxel1", columns="voxel2", values="count")
                     .reindex(index=voxels, columns=voxels))
        count_mat = count_mat.combine_first(count_mat.T)
        np.fill_diagonal(count_mat.values, 0)

    n = len(voxels)

    # ---- composite matrix just for annotations (upper direct, lower mirrored)
    composite = pd.DataFrame(index=voxels, columns=voxels, dtype=float)
    C  = composite.values
    UM = up_mean.values
    LM = lo_mean.values
    for i in range(n):
        for j in range(n):
            if i < j:
                C[i, j] = UM[i, j]
            elif i > j:
                C[i, j] = LM[j, i]  # mirror lower into lower triangle
            else:
                C[i, j] = UM[i, j] if not np.isnan(UM[i, j]) else LM[i, j]

    # ---- build display matrices for independent colorbars
    upper_display = np.full((n, n), np.nan, dtype=float)
    lower_display = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i < j:
                upper_display[i, j] = UM[i, j]
            elif i > j:
                lower_display[i, j] = LM[j, i]
            else:
                if not np.isnan(UM[i, j]):
                    upper_display[i, j] = UM[i, j]
                elif not np.isnan(LM[i, j]):
                    lower_display[i, j] = LM[i, j]

    # ---- per-triangle vmin/vmax
    def finite_vals(arr):
        v = arr[np.isfinite(arr)]
        return v if v.size else np.array([0.0])

    if vmin_upper is None or vmax_upper is None:
        up_vals = finite_vals(upper_display)
    if vmin_lower is None or vmax_lower is None:
        lo_vals = finite_vals(lower_display)

    if vmin_upper is None:
        vmin_upper = (float(np.min(up_vals)) if up_vals.size else (vmin if vmin is not None else 0.0))
    if vmax_upper is None:
        vmax_upper = (float(np.max(up_vals)) if up_vals.size else (vmax if vmax is not None else 1.0))

    if vmin_lower is None:
        vmin_lower = (float(np.min(lo_vals)) if lo_vals.size else (vmin if vmin is not None else 0.0))
    if vmax_lower is None:
        vmax_lower = (float(np.max(lo_vals)) if lo_vals.size else (vmax if vmax is not None else 1.0))

    # ---- annotations (mean ± std) based on composite C
    annot = np.full((n, n), "", dtype=object)
    UPS = up_std.values
    LOS = lo_std.values
    for i in range(n):
        for j in range(n):
            m = C[i, j]
            if np.isnan(m):
                continue
            if i < j:
                s = UPS[i, j] if up_std is not None else np.nan
            elif i > j:
                s = LOS[j, i] if lo_std is not None else np.nan
            else:
                if not np.isnan(UM[i, j]):
                    s = UPS[i, j] if up_std is not None else np.nan
                else:
                    s = LOS[i, j] if lo_std is not None else np.nan
            annot[i, j] = f"{m:.2f}" if np.isnan(s) else f"{m:.2f}\n±\n{s:.2f}"

    # ---- plot: layer two heatmaps with independent colorbars
    fig, ax = plt.subplots(figsize=(10, 8))

    # lower first
    hm_lower = sns.heatmap(
        lower_display,
        cmap="coolwarm",
        vmin=vmin_lower, vmax=vmax_lower,
        mask=~np.isfinite(lower_display),
        cbar=False,
        linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )
    # upper second
    hm_upper = sns.heatmap(
        upper_display,
        cmap="coolwarm",
        vmin=vmin_upper, vmax=vmax_upper,
        mask=~np.isfinite(upper_display),
        cbar=False,
        linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )
    # manual annotation pass (on top)
    cmap_up = plt.get_cmap("coolwarm")
    cmap_lo = plt.get_cmap("coolwarm")

    for i in range(n):
        for j in range(n):
            if annot[i, j] != "":
                if i < j:   # upper triangle
                    color = get_contrasting_color(upper_display[i, j], vmin_upper, vmax_upper, cmap_up)
                else:       # lower triangle (or diagonal)
                    color = get_contrasting_color(lower_display[i, j], vmin_lower, vmax_lower, cmap_lo)
                ax.text(j + 0.5, i + 0.5, annot[i, j],
                        ha="center", va="center", fontsize=6, color=color)


    # Optional title
    if show_title:
        ax.set_title(f"Upper: {eff_size_type_upper}   |   Lower: {eff_size_type_lower}")

        # ---- Four-sided ticks & labels (switched as requested)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
    ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)

    # bottom/left now describe LOWER triangle semantics
    ax.set_xlabel("Voxel 1 (Lower triangle)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Voxel 2 (Lower triangle)", fontsize=axis_label_fontsize)

    # ensure bottom/left ticks are visible with markers
    ax.tick_params(axis="x", bottom=True, top=False, direction="out", length=4, width=1, color="black")
    ax.tick_params(axis="y", left=True, right=False, direction="out", length=4, width=1, color="black")

    # top/right describe UPPER triangle semantics
    top_ax = ax.secondary_xaxis('top')
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
    top_ax.set_xlabel("Voxel 2 (Upper triangle)", fontsize=axis_label_fontsize)
    top_ax.tick_params(axis="x", top=True, direction="out", length=4, width=1, color="black")

    right_ax = ax.secondary_yaxis('right')
    right_ax.set_yticks(ax.get_yticks())
    right_ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)
    right_ax.set_ylabel("Voxel 1 (Upper triangle)", fontsize=axis_label_fontsize)
    right_ax.tick_params(axis="y", right=True, direction="out", length=4, width=1, color="black")


    # ---- colorbars: LEFT for lower (independent), RIGHT for upper (independent)
    divider = make_axes_locatable(ax)
    cax_left  = divider.append_axes("left",  size="4%", pad=0.8)
    cax_right = divider.append_axes("right", size="4%", pad=0.8)

    sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap="coolwarm")
    sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap="coolwarm")
    sm_lower.set_array([])
    sm_upper.set_array([])

    cbar_lower = plt.colorbar(sm_lower, cax=cax_left)
    cbar_upper = plt.colorbar(sm_upper, cax=cax_right)

    cbar_lower.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_upper.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_lower.set_label(cbar_label_lower, fontsize=cbar_label_fontsize)
    cbar_upper.set_label(cbar_label_upper, fontsize=cbar_label_fontsize)

    cax_left.yaxis.set_ticks_position('left')
    cax_left.yaxis.set_label_position('left')
    cax_right.yaxis.set_ticks_position('right')
    cax_right.yaxis.set_label_position('right')

    # ---- count boxes with 'n=' shifted one cell down-right and centered
    if show_counts_boxes and count_mat is not None:
        mask_lt = np.tril(np.ones((n, n), dtype=bool), k=-1)
        vals_lt = count_mat.values.copy()
        vals_lt[~mask_lt] = np.nan
        unique_counts = np.unique(vals_lt[~np.isnan(vals_lt)])

        for c in unique_counts:
            region = (count_mat.values == c) & mask_lt
            coords = np.argwhere(region)
            if coords.size == 0:
                continue

            i_min, j_min = coords.min(axis=0)
            i_max, j_max = coords.max(axis=0)

            # box lines on top
            ax.hlines(y=i_max + 1, xmin=j_min, xmax=j_max + 2,
                      colors="black", linewidth=1.5, zorder=6)
            ax.vlines(x=j_max + 2, ymin=0, ymax=i_max + 1,
                      colors="black", linewidth=1.5, zorder=6)

            # pick diagonal cell inside box and shift one cell down-right
            k_in = min(max(min(i_max, n - 1), j_min), min(j_max, n - 1))
            k = min(k_in + 1, n - 1)

            # white inset above heatmap but below lines
            inset = 0.06
            rect = Rectangle(
                (k + inset, k + inset),
                1 - 2*inset, 1 - 2*inset,
                facecolor="white",
                edgecolor="white",
                zorder=4,
            )
            ax.add_patch(rect)

            # centered 'n=' text
            cx = k + 0.5
            cy = k + 0.5
            ax.text(
                cx, cy, f"n={int(c)}",
                ha="center", va="center",
                fontsize=n_label_fontsize, color="black", zorder=7,
                clip_on=False, transform=ax.transData
            )

    # corner annotation
    if show_annotation_box and annotation_text:
        ax.text(
            0.02, 0.02, annotation_text,
            transform=ax.transAxes,
            fontsize=10,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
        )

    plt.tight_layout()

    # ---- save/show
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)
        base = os.path.join(
            save_path_base,
            f"cohort_dualtri_abs-{aggregate_abs}_{save_name_base or 'dose_upper__dosegrad_lower'}"
            + ("" if not show_counts_boxes else "_with_counts")
            + (f"_{counts_source}_counts" if show_counts_boxes else "")
        )
        plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".svg", bbox_inches="tight")
        print(f"Saved dual-triangle cohort heatmap to {base}.png/.svg")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)








def plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs(
    upper_df: pd.DataFrame,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)
    lower_df: pd.DataFrame,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)

    # tell the function which columns to render in cells:
    upper_mean_col: str = "mean_diff",
    upper_std_col:  str = "std_diff",
    lower_mean_col: str = "mean_diff",
    lower_std_col:  str = "std_diff",

    # which column to use for the "n=" boxes (per voxel pair)
    n_col: str = "n_biopsies",

    eff_size_type_upper: str = "Dose mean±std",
    eff_size_type_lower: str = "Dose-Gradient mean±std",

    save_path_base=None,
    save_name_base=None,
    annotation_info: dict | None = None,

    # color range controls
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_upper: float | None = None,
    vmax_upper: float | None = None,
    vmin_lower: float | None = None,
    vmax_lower: float | None = None,

    # counts overlay
    show_counts_boxes: bool = True,
    counts_source: str = "lower",       # "lower" or "upper"

    # typography
    tick_label_fontsize: int = 9,
    axis_label_fontsize: int = 11,
    cbar_tick_fontsize: int = 9,
    cbar_label_fontsize: int = 11,
    cbar_label_upper: str = "Mean Difference (Gy, Upper)",
    cbar_label_lower: str = "Mean Difference (Gy, Lower)",

    # title
    show_title: bool = True,

    # n= caption fontsize inside boxes
    n_label_fontsize: int = 8,

    # corner annotation box
    show_annotation_box: bool = True,
):
    """
    Plot a dual-triangle heatmap:
      - Upper triangle: means/std from `upper_df` (direct)
      - Lower triangle: means/std from `lower_df` (MIRRORED into lower half)
      - Independent colorbars for upper/lower
      - Optional per-pair "n=" overlay, taken from `n_col` (default 'n_biopsies').

    Required columns in both dataframes:
      - 'voxel1', 'voxel2'
      - mean/std columns as specified by upper_mean_col/upper_std_col, lower_mean_col/lower_std_col
      - Optional: `n_col` for counts overlay (per (voxel1, voxel2)).
    """


    plt.ioff()

    # --- helpers ---
    def get_contrasting_color(val, vmin, vmax, cmap):
        if val is None or not np.isfinite(val):
            return "black"
        norm = (val - vmin) / (vmax - vmin + 1e-12)
        r, g, b, _ = cmap(norm)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "white"

    def _ensure_cols(df, need, tag):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{tag}: missing required columns {missing}")

    # --- validate inputs & extract minimal frames ---
    _ensure_cols(upper_df, ["voxel1", "voxel2", upper_mean_col, upper_std_col], "Upper")
    _ensure_cols(lower_df, ["voxel1", "voxel2", lower_mean_col, lower_std_col], "Lower")

    up_stats = upper_df[["voxel1", "voxel2", upper_mean_col, upper_std_col] + ([n_col] if n_col in upper_df.columns else [])] \
        .rename(columns={upper_mean_col: "mean", upper_std_col: "std"})
    lo_stats = lower_df[["voxel1", "voxel2", lower_mean_col, lower_std_col] + ([n_col] if n_col in lower_df.columns else [])] \
        .rename(columns={lower_mean_col: "mean", lower_std_col: "std"})

    # --- unified voxel set ---
    voxels = sorted(
        set(up_stats["voxel1"]).union(up_stats["voxel2"])
        .union(lo_stats["voxel1"]).union(lo_stats["voxel2"])
    )
    n = len(voxels)

    # --- pivot to matrices ---
    def pivots(stats_df):
        mean_mat = (stats_df.pivot(index="voxel1", columns="voxel2", values="mean")
                    .reindex(index=voxels, columns=voxels))
        std_mat  = (stats_df.pivot(index="voxel1", columns="voxel2", values="std")
                    .reindex(index=voxels, columns=voxels))
        return mean_mat, std_mat

    up_mean, up_std = pivots(up_stats)
    lo_mean, lo_std = pivots(lo_stats)

    # counts (optional overlay) from n_col
    count_mat = None
    if show_counts_boxes:
        src = up_stats if counts_source == "upper" else lo_stats
        if n_col in src.columns:
            count_mat = (src.pivot(index="voxel1", columns="voxel2", values=n_col)
                         .reindex(index=voxels, columns=voxels))
            # mirror-symmetrize & clear diagonal
            count_mat = count_mat.combine_first(count_mat.T)
            if count_mat.values.size:  # avoid empty diagonal fill error
                np.fill_diagonal(count_mat.values, 0)

    # --- composite & display matrices ---
    composite = pd.DataFrame(index=voxels, columns=voxels, dtype=float)
    C  = composite.values
    UM = up_mean.values
    LM = lo_mean.values

    for i in range(n):
        for j in range(n):
            if i < j:
                C[i, j] = UM[i, j]
            elif i > j:
                C[i, j] = LM[j, i]  # mirror lower into lower
            else:
                C[i, j] = UM[i, j] if not np.isnan(UM[i, j]) else LM[i, j]

    upper_display = np.full((n, n), np.nan, dtype=float)
    lower_display = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i < j:
                upper_display[i, j] = UM[i, j]
            elif i > j:
                lower_display[i, j] = LM[j, i]
            else:
                if not np.isnan(UM[i, j]):
                    upper_display[i, j] = UM[i, j]
                elif not np.isnan(LM[i, j]):
                    lower_display[i, j] = LM[i, j]

    # --- vmin/vmax resolve per triangle (fall back to global vmin/vmax if provided) ---
    def finite_vals(arr):
        v = arr[np.isfinite(arr)]
        return v if v.size else np.array([0.0])

    if vmin_upper is None or vmax_upper is None:
        up_vals = finite_vals(upper_display)
    if vmin_lower is None or vmax_lower is None:
        lo_vals = finite_vals(lower_display)

    if vmin_upper is None:
        vmin_upper = (float(np.min(up_vals)) if up_vals.size else (vmin if vmin is not None else 0.0))
    if vmax_upper is None:
        vmax_upper = (float(np.max(up_vals)) if up_vals.size else (vmax if vmax is not None else 1.0))
    if vmin_lower is None:
        vmin_lower = (float(np.min(lo_vals)) if lo_vals.size else (vmin if vmin is not None else 0.0))
    if vmax_lower is None:
        vmax_lower = (float(np.max(lo_vals)) if lo_vals.size else (vmax if vmax is not None else 1.0))

    # --- annotations (mean ± std) ---
    annot = np.full((n, n), "", dtype=object)
    UPS = up_std.values
    LOS = lo_std.values
    for i in range(n):
        for j in range(n):
            m = C[i, j]
            if np.isnan(m):
                continue
            if i < j:
                s = UPS[i, j]
            elif i > j:
                s = LOS[j, i]
            else:
                s = UPS[i, j] if np.isfinite(UM[i, j]) else LOS[i, j]
            annot[i, j] = f"{m:.2f}" if not np.isfinite(s) else f"{m:.2f}\n±\n{s:.2f}"

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    hm_lower = sns.heatmap(
        lower_display, cmap="coolwarm",
        vmin=vmin_lower, vmax=vmax_lower,
        mask=~np.isfinite(lower_display),
        cbar=False, linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )
    hm_upper = sns.heatmap(
        upper_display, cmap="coolwarm",
        vmin=vmin_upper, vmax=vmax_upper,
        mask=~np.isfinite(upper_display),
        cbar=False, linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )

    cmap_up = plt.get_cmap("coolwarm")
    cmap_lo = plt.get_cmap("coolwarm")
    for i in range(n):
        for j in range(n):
            if annot[i, j] != "":
                color = (
                    get_contrasting_color(upper_display[i, j], vmin_upper, vmax_upper, cmap_up)
                    if i < j else
                    get_contrasting_color(lower_display[i, j], vmin_lower, vmax_lower, cmap_lo)
                )
                ax.text(j + 0.5, i + 0.5, annot[i, j],
                        ha="center", va="center", fontsize=6, color=color)

    if show_title:
        ax.set_title(f"Upper: {eff_size_type_upper}   |   Lower: {eff_size_type_lower}")

    # ticks/labels
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels([int(v) for v in voxels], fontsize=tick_label_fontsize)
    ax.set_yticklabels([int(v) for v in voxels], fontsize=tick_label_fontsize)
    ax.set_xlabel("Voxel 1 (Lower triangle)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Voxel 2 (Lower triangle)", fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", bottom=True, top=False, direction="out", length=4, width=1, color="black")
    ax.tick_params(axis="y", left=True, right=False, direction="out", length=4, width=1, color="black")

    top_ax = ax.secondary_xaxis('top')
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels([int(v) for v in voxels], fontsize=tick_label_fontsize)
    top_ax.set_xlabel("Voxel 2 (Upper triangle)", fontsize=axis_label_fontsize)
    top_ax.tick_params(axis="x", top=True, direction="out", length=4, width=1, color="black")

    right_ax = ax.secondary_yaxis('right')
    right_ax.set_yticks(ax.get_yticks())
    right_ax.set_yticklabels([int(v) for v in voxels], fontsize=tick_label_fontsize)
    right_ax.set_ylabel("Voxel 1 (Upper triangle)", fontsize=axis_label_fontsize)
    right_ax.tick_params(axis="y", right=True, direction="out", length=4, width=1, color="black")

    # colorbars (independent)
    divider = make_axes_locatable(ax)
    cax_left  = divider.append_axes("left",  size="4%", pad=0.8)
    cax_right = divider.append_axes("right", size="4%", pad=0.8)

    sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap="coolwarm")
    sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap="coolwarm")
    sm_lower.set_array([])
    sm_upper.set_array([])

    cbar_lower = plt.colorbar(sm_lower, cax=cax_left)
    cbar_upper = plt.colorbar(sm_upper, cax=cax_right)
    cbar_lower.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_upper.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_lower.set_label(cbar_label_lower, fontsize=cbar_label_fontsize)
    cbar_upper.set_label(cbar_label_upper, fontsize=cbar_label_fontsize)
    cax_left.yaxis.set_ticks_position('left')
    cax_left.yaxis.set_label_position('left')
    cax_right.yaxis.set_ticks_position('right')
    cax_right.yaxis.set_label_position('right')

    # counts overlay using n_col
    if show_counts_boxes and count_mat is not None:
        mask_lt = np.tril(np.ones((n, n), dtype=bool), k=-1)
        vals_lt = count_mat.values.copy()
        vals_lt[~mask_lt] = np.nan
        unique_counts = np.unique(vals_lt[~np.isnan(vals_lt)])

        for c in unique_counts:
            region = (count_mat.values == c) & mask_lt
            coords = np.argwhere(region)
            if coords.size == 0:
                continue

            i_min, j_min = coords.min(axis=0)
            i_max, j_max = coords.max(axis=0)

            # box lines
            ax.hlines(y=i_max + 1, xmin=j_min, xmax=j_max + 2,
                      colors="black", linewidth=1.5, zorder=6)
            ax.vlines(x=j_max + 2, ymin=0, ymax=i_max + 1,
                      colors="black", linewidth=1.5, zorder=6)

            # pick diagonal-ish cell, shift one down-right
            k_in = min(max(min(i_max, n - 1), j_min), min(j_max, n - 1))
            k = min(k_in + 1, n - 1)

            inset = 0.06
            rect = Rectangle(
                (k + inset, k + inset),
                1 - 2*inset, 1 - 2*inset,
                facecolor="white",
                edgecolor="white",
                zorder=4,
            )
            ax.add_patch(rect)

            cx = k + 0.5
            cy = k + 0.5
            ax.text(
                cx, cy, f"n={int(c)}",
                ha="center", va="center",
                fontsize=n_label_fontsize, color="black", zorder=7,
                clip_on=False, transform=ax.transData
            )

    # corner annotation (optional, user-supplied dict)
    if show_annotation_box and annotation_info:
        text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
        )

    plt.tight_layout()

    # save/show
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)
        base = os.path.join(
            save_path_base,
            f"cohort_dualtri_{save_name_base or 'dose_upper__dosegrad_lower'}"
        )
        plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".svg", bbox_inches="tight")
        print(f"Saved dual-triangle cohort heatmap to {base}.png/.svg")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)






def plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
    upper_df: pd.DataFrame,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)
    lower_df: pd.DataFrame,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)

    # tell the function which columns to render in cells:
    upper_mean_col: str = "mean_diff",
    upper_std_col:  str | None = "std_diff",
    lower_mean_col: str = "mean_diff",
    lower_std_col:  str | None = "std_diff",

    # which column to use for the "n=" boxes (per voxel pair)
    n_col: str = "n_biopsies",

    eff_size_type_upper: str = "Dose mean±std",
    eff_size_type_lower: str = "Dose-Gradient mean±std",

    save_path_base=None,
    save_name_base=None,
    annotation_info: dict | None = None,

    # color range controls
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_upper: float | None = None,
    vmax_upper: float | None = None,
    vmin_lower: float | None = None,
    vmax_lower: float | None = None,

    # counts overlay
    show_counts_boxes: bool = True,
    counts_source: str = "lower",       # "lower" or "upper"

    # typography
    tick_label_fontsize: int = 9,
    axis_label_fontsize: int = 11,
    cbar_tick_fontsize: int = 9,
    cbar_label_fontsize: int = 11,
    cbar_label_upper: str = r"$\overline{M_{ij}^{D}}$ (Upper triangle)",
    cbar_label_lower: str = r"$\overline{M_{ij}^{G}}$ (Lower triangle)",
    # NEW:
    cbar_pad: float = 0.6,        # distance between matrix and colorbar
    cbar_label_pad: float = 8.0,  # distance between bar and its label

    # title
    show_title: bool = True,

    # cell annotation fontsize (mean / mean±std)
    cell_annot_fontsize: int = 8,
    cell_value_decimals: int = 1,


    # n= caption fontsize inside boxes
    n_label_fontsize: int = 8,

    # corner annotation box
    show_annotation_box: bool = True,

    # axis label semantics (match your patient-level defaults)
    y_axis_upper_tri_label: str = r"Voxel $i$",
    x_axis_upper_tri_label: str = r"Voxel $j$",
    y_axis_lower_tri_label: str = r"Voxel $j$",
    x_axis_lower_tri_label: str = r"Voxel $i$",

    # rendering controls
    cmap: str = "coolwarm",
    cast_ticklabels_to_int: bool = True,
    color_bar_positions: str = "top_bottom",  # "left_right" or "top_bottom"
):
    """
    Dual-triangle cohort heatmap using cohort-pooled stats:
      - Upper triangle from `upper_df` (direct)
      - Lower triangle from `lower_df` (mirrored into lower half)
      - Independent colorbars for upper/lower
      - Optional per-pair "n=" overlay (kept from v1)
      - Std on/off via upper_std_col/lower_std_col = None
      - i/j axis-label semantics (upper vs lower can be labeled differently)
    """
    plt.ioff()

    # ---------- helpers ----------
    def get_contrasting_color(val, vmin_, vmax_, cmap_):
        if val is None or not np.isfinite(val):
            return "black"
        denom = (vmax_ - vmin_) if (vmax_ is not None and vmin_ is not None) else 1.0
        norm = (val - vmin_) / (denom + 1e-12)
        r, g, b, _ = plt.get_cmap(cmap_)(norm)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "white"

    def _ensure_cols(df, need, tag):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{tag}: missing required columns {missing}")

    def finite_vals(arr):
        v = arr[np.isfinite(arr)]
        return v if v.size else np.array([])

    # ---------- validate inputs ----------
    base_need = ["voxel1", "voxel2", upper_mean_col]
    if upper_std_col is not None:
        base_need.append(upper_std_col)
    _ensure_cols(upper_df, base_need, "Upper")

    base_need = ["voxel1", "voxel2", lower_mean_col]
    if lower_std_col is not None:
        base_need.append(lower_std_col)
    _ensure_cols(lower_df, base_need, "Lower")

    # ---------- minimal frames ----------
    up_cols = ["voxel1", "voxel2", upper_mean_col] + ([upper_std_col] if upper_std_col else []) \
              + ([n_col] if (show_counts_boxes and n_col in upper_df.columns) else [])
    lo_cols = ["voxel1", "voxel2", lower_mean_col] + ([lower_std_col] if lower_std_col else []) \
              + ([n_col] if (show_counts_boxes and n_col in lower_df.columns) else [])

    up_stats = upper_df[up_cols].copy()
    lo_stats = lower_df[lo_cols].copy()

    up_stats = up_stats.rename(columns={upper_mean_col: "mean"})
    lo_stats = lo_stats.rename(columns={lower_mean_col: "mean"})
    if upper_std_col:
        up_stats = up_stats.rename(columns={upper_std_col: "std"})
    else:
        up_stats["std"] = np.nan
    if lower_std_col:
        lo_stats = lo_stats.rename(columns={lower_std_col: "std"})
    else:
        lo_stats["std"] = np.nan

    # ---------- unified voxel set ----------
    voxels = sorted(
        set(up_stats["voxel1"]).union(up_stats["voxel2"])
        .union(lo_stats["voxel1"]).union(lo_stats["voxel2"])
    )
    n = len(voxels)
    if n == 0:
        return

    # ---------- pivot to matrices ----------
    def pivots(stats_df):
        mean_mat = (stats_df.pivot(index="voxel1", columns="voxel2", values="mean")
                    .reindex(index=voxels, columns=voxels))
        std_mat  = (stats_df.pivot(index="voxel1", columns="voxel2", values="std")
                    .reindex(index=voxels, columns=voxels))
        return mean_mat, std_mat

    up_mean, up_std = pivots(up_stats)
    lo_mean, lo_std = pivots(lo_stats)

    UM = up_mean.values
    LM = lo_mean.values
    UPS = up_std.values
    LOS = lo_std.values

    # ---------- counts (optional overlay) ----------
    count_mat = None
    if show_counts_boxes:
        src = up_stats if counts_source == "upper" else lo_stats
        if n_col in src.columns:
            count_mat = (src.pivot(index="voxel1", columns="voxel2", values=n_col)
                         .reindex(index=voxels, columns=voxels))
            count_mat = count_mat.combine_first(count_mat.T)
            if count_mat.values.size:
                np.fill_diagonal(count_mat.values, 0)

    # ---------- build display matrices (independent) ----------
    upper_display = np.full((n, n), np.nan, dtype=float)
    lower_display = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i < j:
                upper_display[i, j] = UM[i, j]
            elif i > j:
                lower_display[i, j] = LM[j, i]  # mirror into lower
            else:
                # diagonal: prefer upper; else lower
                if np.isfinite(UM[i, j]):
                    upper_display[i, j] = UM[i, j]
                elif np.isfinite(LM[i, j]):
                    lower_display[i, j] = LM[i, j]

    # ---------- resolve vmin/vmax per triangle ----------
    if vmin_upper is None or vmax_upper is None:
        up_vals = finite_vals(upper_display)
    if vmin_lower is None or vmax_lower is None:
        lo_vals = finite_vals(lower_display)

    if vmin_upper is None:
        vmin_upper = float(up_vals.min()) if up_vals.size else (vmin if vmin is not None else 0.0)
    if vmax_upper is None:
        vmax_upper = float(up_vals.max()) if up_vals.size else (vmax if vmax is not None else 1.0)
    if vmin_lower is None:
        vmin_lower = float(lo_vals.min()) if lo_vals.size else (vmin if vmin is not None else 0.0)
    if vmax_lower is None:
        vmax_lower = float(lo_vals.max()) if lo_vals.size else (vmax if vmax is not None else 1.0)

    # ---------- annotation strings (mean ± std OR mean-only) ----------
    annot = np.full((n, n), "", dtype=object)

    # small helper for formatting with configurable decimals
    fmt = f"{{:.{int(cell_value_decimals)}f}}"

    for i in range(n):
        for j in range(n):
            if i < j:
                m = UM[i, j]
                if not np.isfinite(m):
                    continue
                s = UPS[i, j]
                if not np.isfinite(s):
                    annot[i, j] = fmt.format(m)
                else:
                    annot[i, j] = f"{fmt.format(m)}\n±\n{fmt.format(s)}"

            elif i > j:
                m = LM[j, i]
                if not np.isfinite(m):
                    continue
                s = LOS[j, i]
                if not np.isfinite(s):
                    annot[i, j] = fmt.format(m)
                else:
                    annot[i, j] = f"{fmt.format(m)}\n±\n{fmt.format(s)}"

            else:
                # diagonal: prefer upper; else lower
                if np.isfinite(UM[i, j]):
                    m = UM[i, j]
                    s = UPS[i, j]
                elif np.isfinite(LM[i, j]):
                    m = LM[i, j]
                    s = LOS[i, j]
                else:
                    continue

                if not np.isfinite(s):
                    annot[i, j] = fmt.format(m)
                else:
                    annot[i, j] = f"{fmt.format(m)}\n±\n{fmt.format(s)}"


    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        lower_display, cmap=cmap,
        vmin=vmin_lower, vmax=vmax_lower,
        mask=~np.isfinite(lower_display),
        cbar=False, linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )
    sns.heatmap(
        upper_display, cmap=cmap,
        vmin=vmin_upper, vmax=vmax_upper,
        mask=~np.isfinite(upper_display),
        cbar=False, linewidths=0.3, linecolor="white",
        square=True, ax=ax,
    )

    # manual annotations (contrast by triangle; diagonal chooses upper if present)
    for i in range(n):
        for j in range(n):
            txt = annot[i, j]
            if not txt:
                continue

            if i < j or (i == j and np.isfinite(upper_display[i, j])):
                color = get_contrasting_color(upper_display[i, j], vmin_upper, vmax_upper, cmap)
            else:
                color = get_contrasting_color(lower_display[i, j], vmin_lower, vmax_lower, cmap)

            ax.text(
                j + 0.5, i + 0.5, txt,
                ha="center", va="center",
                fontsize=cell_annot_fontsize,
                color=color
            )

    if show_title:
        ax.set_title(f"Upper: {eff_size_type_upper}   |   Lower: {eff_size_type_lower}")

    # tick labels
    if cast_ticklabels_to_int:
        try:
            ticklabs = [int(v) for v in voxels]
        except Exception:
            ticklabs = voxels
    else:
        ticklabs = voxels

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    rotation_x = 90  # vertical / sideways labels for x only

    # Bottom x-axis: sideways
    ax.set_xticklabels(
        ticklabs,
        fontsize=tick_label_fontsize,
        rotation=rotation_x,
        ha="center",
        va="top",
    )

    # Left y-axis: horizontal
    ax.set_yticklabels(
        ticklabs,
        fontsize=tick_label_fontsize,
        rotation=0,
    )

    # Lower triangle semantics on bottom/left
    ax.set_xlabel(x_axis_lower_tri_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_axis_lower_tri_label, fontsize=axis_label_fontsize)
    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        direction="out",
        length=4,
        width=1,
        color="black",
    )
    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        direction="out",
        length=4,
        width=1,
        color="black",
    )

    # Upper triangle semantics on top/right
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels(
        ticklabs,
        fontsize=tick_label_fontsize,
        rotation=rotation_x,
        ha="center",
        va="bottom",
    )
    top_ax.set_xlabel(x_axis_upper_tri_label, fontsize=axis_label_fontsize)
    top_ax.tick_params(
        axis="x",
        top=True,
        direction="out",
        length=4,
        width=1,
        color="black",
    )

    right_ax = ax.secondary_yaxis("right")
    right_ax.set_yticks(ax.get_yticks())
    right_ax.set_yticklabels(
        ticklabs,
        fontsize=tick_label_fontsize,
        rotation=0,
    )
    right_ax.set_ylabel(y_axis_upper_tri_label, fontsize=axis_label_fontsize)
    right_ax.tick_params(
        axis="y",
        right=True,
        direction="out",
        length=4,
        width=1,
        color="black",
    )


    # independent colorbars (LEFT=lower, RIGHT=upper)
    if color_bar_positions == "left_right":    
        divider = make_axes_locatable(ax)
        cax_left  = divider.append_axes("left",  size="4%", pad=cbar_pad)
        cax_right = divider.append_axes("right", size="4%", pad=cbar_pad)

        sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap=cmap)
        sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap=cmap)
        sm_lower.set_array([])
        sm_upper.set_array([])

        cbar_lower = plt.colorbar(sm_lower, cax=cax_left)
        cbar_upper = plt.colorbar(sm_upper, cax=cax_right)
        cbar_lower.ax.tick_params(labelsize=cbar_tick_fontsize)
        cbar_upper.ax.tick_params(labelsize=cbar_tick_fontsize)
        cbar_lower.set_label(
            cbar_label_lower,
            fontsize=cbar_label_fontsize,
            labelpad=cbar_label_pad,
        )
        cbar_upper.set_label(
            cbar_label_upper,
            fontsize=cbar_label_fontsize,
            labelpad=cbar_label_pad,
        )

        cax_left.yaxis.set_ticks_position('left')
        cax_left.yaxis.set_label_position('left')
        cax_right.yaxis.set_ticks_position('right')
        cax_right.yaxis.set_label_position('right')
    

    # independent colorbars (BOTTOM=lower, TOP=upper)
    if color_bar_positions == "top_bottom":
        divider = make_axes_locatable(ax)
        cax_bottom = divider.append_axes("bottom", size="4%", pad=cbar_pad)
        cax_top    = divider.append_axes("top",    size="4%", pad=cbar_pad)

        sm_lower = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_lower, vmax=vmax_lower), cmap=cmap)
        sm_upper = ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_upper, vmax=vmax_upper), cmap=cmap)
        sm_lower.set_array([])
        sm_upper.set_array([])

        # lower triangle colorbar - below the matrix
        cbar_lower = plt.colorbar(sm_lower, cax=cax_bottom, orientation="horizontal")
        # upper triangle colorbar - above the matrix
        cbar_upper = plt.colorbar(sm_upper, cax=cax_top,    orientation="horizontal")

        # tick sizes
        cbar_lower.ax.tick_params(axis="x", labelsize=cbar_tick_fontsize)
        cbar_upper.ax.tick_params(axis="x", labelsize=cbar_tick_fontsize)

        # labels
        cbar_lower.set_label(
            cbar_label_lower,
            fontsize=cbar_label_fontsize,
            labelpad=cbar_label_pad,
        )
        cbar_upper.set_label(
            cbar_label_upper,
            fontsize=cbar_label_fontsize,
            labelpad=cbar_label_pad,
        )


        # put ticks and label on the top edge for the upper bar
        cax_top.xaxis.set_ticks_position("top")
        cax_top.xaxis.set_label_position("top")
        # bottom bar keeps default bottom ticks and label


    # ---------- counts overlay (kept) ----------
    if show_counts_boxes and count_mat is not None:
        mask_lt = np.tril(np.ones((n, n), dtype=bool), k=-1)
        vals_lt = count_mat.values.copy()
        vals_lt[~mask_lt] = np.nan
        unique_counts = np.unique(vals_lt[~np.isnan(vals_lt)])

        for c in unique_counts:
            region = (count_mat.values == c) & mask_lt
            coords = np.argwhere(region)
            if coords.size == 0:
                continue

            i_min, j_min = coords.min(axis=0)
            i_max, j_max = coords.max(axis=0)

            ax.hlines(y=i_max + 1, xmin=j_min, xmax=j_max + 2,
                      colors="black", linewidth=1.5, zorder=6)
            ax.vlines(x=j_max + 2, ymin=0, ymax=i_max + 1,
                      colors="black", linewidth=1.5, zorder=6)

            k_in = min(max(min(i_max, n - 1), j_min), min(j_max, n - 1))
            k = min(k_in + 1, n - 1)

            inset = 0.06
            rect = Rectangle(
                (k + inset, k + inset),
                1 - 2*inset, 1 - 2*inset,
                facecolor="white",
                edgecolor="white",
                zorder=4,
            )
            ax.add_patch(rect)

            ax.text(
                k + 0.5, k + 0.5, f"{int(c)}",
                ha="center", va="center",
                fontsize=n_label_fontsize, color="black", zorder=7,
                clip_on=False, transform=ax.transData
            )

    # corner annotation (optional)
    if show_annotation_box and annotation_info:
        text = "\n".join(f"{k}: {v}" for k, v in annotation_info.items())
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
        )

    plt.tight_layout()

    # save/show
    if save_path_base:
        os.makedirs(save_path_base, exist_ok=True)
        base = os.path.join(
            save_path_base,
            f"cohort_dualtri_{save_name_base or 'dose_upper__dosegrad_lower'}_v2"
        )

        # raster + vector outputs
        plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".svg", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".pdf", bbox_inches="tight", dpi=600, format="pdf")

        print(f"Saved dual-triangle cohort heatmap to {base}.png/.svg/.pdf")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


























def dvh_boxplot(cohort_bx_dvh_metrics_df, save_path=None, custom_name=None,
                title: str | None = None,
                axis_label_font_size: int | None = 14,   # default 14
                tick_label_font_size: int | None = 12):  # default 12

    # Filter data for D_x and V_x metrics
    d_metrics = cohort_bx_dvh_metrics_df[cohort_bx_dvh_metrics_df['Metric'].str.startswith('D_')]
    v_metrics = cohort_bx_dvh_metrics_df[cohort_bx_dvh_metrics_df['Metric'].str.startswith('V_')]

    def convert_metric_label(metric):
        """Convert 'D_2' to 'D$_{2\\%}$' for subscript number and percent in mathtext."""
        if metric.startswith(('D_', 'V_')):
            base = metric[0]  # 'D' or 'V'
            number = metric.split('_')[1]
            return f"{base}$_{{{number}\\%}}$"
        return metric

    # Function to create boxplot
    def create_plot(data, metric_type):
        plt.figure(figsize=(12, 8))
        boxplot = data.boxplot(by='Metric', column=['Mean'], grid=True)

        # optional title
        plt.suptitle('')  # keep clearing the pandas-added suptitle
        if title is not None:
            plt.title(title, fontsize=14)
        else:
            # no title printed
            plt.title('')

        # x/y labels with optional font sizes
        if axis_label_font_size is not None:
            plt.xlabel('DVH Metric', fontsize=axis_label_font_size)
        else:
            plt.xlabel('DVH Metric')

        if 'V' in metric_type:
            if axis_label_font_size is not None:
                plt.ylabel('Percent volume', fontsize=axis_label_font_size)
            else:
                plt.ylabel('Percent volume')
        elif 'D' in metric_type:
            if axis_label_font_size is not None:
                plt.ylabel('Dose (Gy)', fontsize=axis_label_font_size)
            else:
                plt.ylabel('Dose (Gy)')

        # initial tick rotation (and optional size)
        if tick_label_font_size is not None:
            plt.xticks(rotation=45, fontsize=tick_label_font_size)
            plt.yticks(fontsize=tick_label_font_size)
        else:
            plt.xticks(rotation=45)

        # After plotting
        xtick_labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]
        custom_labels = [convert_metric_label(label) for label in xtick_labels]
        if tick_label_font_size is not None:
            plt.gca().set_xticklabels(custom_labels, rotation=45, fontsize=tick_label_font_size)
        else:
            plt.gca().set_xticklabels(custom_labels, rotation=45)

        # remove vertical grid lines
        plt.gca().yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().xaxis.grid(False)  # Disable vertical grid lines

        # start y axis range at 0 if metric is dose
        if 'D' in metric_type:
            # Set y-axis limits to start at 0 for dose metrics
            plt.gca().set_ylim(bottom=0)
        else:
            # For volume metrics, set y-axis limits to always range from 0 to 100
            plt.gca().set_ylim(0, 100)

        # Annotation box on top-right corner
        n_patients = data['Patient ID'].nunique()
        n_biopsies = data.groupby(['Patient ID', 'Bx ID']).ngroups
        annotation_text = f"Patients: {n_patients}\nBiopsies: {n_biopsies}"
        # add annotation text box in the top-right corner
        ax = plt.gca()
        ax.text(
            0.98, 0.98,
            annotation_text,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
        )

        if save_path:
            filename_suffix = metric_type.lower().replace(' ', '_')
            if custom_name:
                png_path = save_path.joinpath(f"{custom_name}_{filename_suffix}.png")
                svg_path = save_path.joinpath(f"{custom_name}_{filename_suffix}.svg")
            else:
                png_path = save_path.joinpath(f"{filename_suffix}.png")
                svg_path = save_path.joinpath(f"{filename_suffix}.svg")

            plt.savefig(png_path, format='png', bbox_inches='tight')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Plot saved as PNG: {png_path}")
            print(f"Plot saved as SVG: {svg_path}")
        else:
            plt.show()
        plt.close()

    # Create plots for D_x and V_x metrics
    create_plot(d_metrics, 'D_x')
    create_plot(v_metrics, 'V_x')




def dvh_boxplot_pretty(
    cohort_bx_dvh_metrics_df,
    save_path: Path | None = None,
    custom_name: str | None = None,
    title: str | None = None,
    axis_label_font_size: int | None = 14,
    tick_label_font_size: int | None = 12,
    seaborn_style: str = "ticks",
    seaborn_context: str = "paper",
    font_scale: float = 1.2,
    metric_order_D: Sequence[str] | None = None,  # e.g. ["D_2", "D_50", "D_98"]
    metric_order_V: Sequence[str] | None = None,  # e.g. ["V_150"]
    flier_size: float = 4.0,          # <--- NEW
    flier_alpha: float = 1.0,         # <--- NEW
    annotate: bool = False,            # <--- NEW
):
    plt.ioff()  # <--- this stops figures from popping up


    def convert_metric_label(metric: str) -> str:
        """Convert 'D_2' -> 'D$_{2\\%}$' for mathtext."""
        if metric.startswith(("D_", "V_")):
            base = metric[0]          # 'D' or 'V'
            number = metric.split("_")[1]
            return f"{base}$_{{{number}\\%}}$"
        return metric

    d_metrics = cohort_bx_dvh_metrics_df[
        cohort_bx_dvh_metrics_df["Metric"].str.startswith("D_")
    ].copy()
    v_metrics = cohort_bx_dvh_metrics_df[
        cohort_bx_dvh_metrics_df["Metric"].str.startswith("V_")
    ].copy()

    def _plot_one(data, metric_type: str, metric_order: Sequence[str] | None):
        if data.empty:
            return

        sns.set_style(seaborn_style)
        sns.set_context(seaborn_context, font_scale=font_scale)

        # Decide order (if not given, use sorted unique)
        if metric_order is None:
            metric_order_local = sorted(data["Metric"].unique())
        else:
            metric_order_local = [m for m in metric_order if m in data["Metric"].unique()]

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

        sns.boxplot(
            data=data,
            x="Metric",
            y="Mean",
            order=metric_order_local,
            ax=ax,
            width=0.6,
            fliersize=2,
            linewidth=1.0,
            boxprops=dict(facecolor="lightgray", edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(                      # <--- explicit outlier styling
                marker="o",
                markersize=flier_size,
                markerfacecolor="black",
                markeredgecolor="black",
                alpha=flier_alpha,
            ),
        )


        # Title
        if title is not None:
            ax.set_title(title, fontsize=axis_label_font_size or 14)
        else:
            ax.set_title("")

        # Axis labels
        if axis_label_font_size is not None:
            ax.set_xlabel("DVH metric", fontsize=axis_label_font_size)
        else:
            ax.set_xlabel("DVH metric")

        if "V" in metric_type:
            ylabel = "Percent volume"
        else:
            ylabel = "Dose (Gy)"

        if axis_label_font_size is not None:
            ax.set_ylabel(ylabel, fontsize=axis_label_font_size)
        else:
            ax.set_ylabel(ylabel)

        # Tick labels
        tick_labels = [convert_metric_label(m) for m in metric_order_local]
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=tick_label_font_size or 12,
        )
        if tick_label_font_size is not None:
            ax.tick_params(axis="y", labelsize=tick_label_font_size)

        # Grids & spines
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.grid(visible=False, axis="x")
        sns.despine(ax=ax, left=False, bottom=False)

        # Y limits
        if "D" in metric_type:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=0, top=ymax)
        else:
            ax.set_ylim(0, 100)

        # Annotation box
        if annotate:
            n_patients = data["Patient ID"].nunique()
            n_biopsies = data.groupby(["Patient ID", "Bx ID"]).ngroups
            annotation_text = f"Patients: {n_patients}\nBiopsies: {n_biopsies}"

            ax.text(
                0.98,
                0.98,
                annotation_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=tick_label_font_size or 12,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

        fig.tight_layout()

        if save_path is not None:
            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            metric_suffix = metric_type.lower().replace(" ", "_")
            if custom_name:
                stem = f"{custom_name}_{metric_suffix}"
            else:
                stem = metric_suffix

            png_path = out_dir / f"{stem}.png"
            svg_path = out_dir / f"{stem}.svg"
            pdf_path = out_dir / f"{stem}.pdf"

            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            fig.savefig(svg_path, bbox_inches="tight", dpi=300)
            fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=600)

            print(f"Saved: {png_path}")
            print(f"Saved: {svg_path}")
            print(f"Saved: {pdf_path}")

        plt.close(fig)


    _plot_one(d_metrics, "D_x", metric_order_D)
    _plot_one(v_metrics, "V_x", metric_order_V)




def plot_strip_scatter(
    df,
    x_col: str,
    y_col: str,
    save_dir: str,
    file_name: str,
    title: str,
    xlabel: str = None,
    ylabel: str = None,
    figsize=(10, 6),
    dpi=300
):
    """
    Create and save a seaborn scatter (strip-like) plot of y_col vs x_col.

    Parameters:
    - df: pandas DataFrame
    - x_col: column name for x-axis (e.g., 'length_scale')
    - y_col: column name for y-axis (e.g., 'dose_diff')
    - save_dir: directory to save the output files
    - file_name: base name for saved plot (no extension)
    - title: title for the plot
    - figsize: tuple, default (10, 6)
    - dpi: resolution for saving, default 300
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.3, s=10)
    plt.title(title, fontsize=14)
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()

    # Save as both PNG and SVG
    png_path = os.path.join(save_dir, f"{file_name}.png")
    svg_path = os.path.join(save_dir, f"{file_name}.svg")
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path)
    plt.close()

    print(f"Plot saved as:\n - {png_path}\n - {svg_path}")


def plot_dose_vs_length_with_summary(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        save_dir: str,
        file_name: str,
        title: str,
        figsize=(12, 7),
        dpi=300,
        show_points=False,
        violin_or_box: str = 'violin',  # 'box' or 'violin'
        trend_lines: list = ['mean'],
        annotate_counts=True,
        y_trim=False,
        y_min_quantile=0.05,
        y_max_quantile=0.95,
        y_min_fixed=None,
        y_max_fixed=None,
        xlabel: str = None,
        ylabel: str = None,
        errorbar_lineplot_mean: str = 'ci',  # 'ci' for confidence interval around the estimator
        errorbar_lineplot_median: str = None  # 'ci' for confidence interval around the estimator
    ):

    plt.ioff()

    df = df.copy()
    df[x_col] = df[x_col].astype(str)


    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Determine y-limits
    if y_min_fixed is not None:
        y_min = y_min_fixed
    elif y_trim:
        y_min = df[y_col].quantile(y_min_quantile)
    else:
        y_min = df[y_col].min()

    if y_max_fixed is not None:
        y_max = y_max_fixed
    elif y_trim:
        # calculate the quantile for each strip, and use the max of those quantiles
        max_quantiles_list = []
        for x_val in df[x_col].unique():
            strip = df[df[x_col] == x_val]
            strip_y_max = strip[y_col].quantile(y_max_quantile)
            max_quantiles_list.append(strip_y_max)
        if max_quantiles_list:
            y_max = max(max_quantiles_list)
        else:
            y_max = df[y_col].quantile(y_max_quantile)
    else:
        y_max = df[y_col].max()

    if violin_or_box == 'violin':
        sns.violinplot(data=df, x=x_col, y=y_col, inner=None, color="lightgray", ax=ax)
    elif violin_or_box == 'box':
        sns.boxplot(data=df, x=x_col, y=y_col, color="lightgray", ax=ax, showfliers=False)
    else:
        raise ValueError("`violin_or_box` must be either 'violin' or 'box'")


    # Prepare tick map for annotation and stats overlay
    xticks = ax.get_xticks()
    xlabels = ax.get_xticklabels()
    tick_map = {label.get_text(): tick for label, tick in zip(xlabels, xticks)}


    # Violin summary stats overlay (simplified as per your preference)
    if violin_or_box == 'violin':
        stats = df.groupby(x_col)[y_col].agg(['mean', 'median',
                                            lambda x: x.quantile(0.25),
                                            lambda x: x.quantile(0.75)])
        stats.columns = ['mean', 'median', 'q25', 'q75']

        for x_val, row in stats.iterrows():
            x_pos = tick_map.get(x_val, x_val)
            for stat_y, style in zip(
                [row['q25'], row['median'], row['mean'], row['q75']],
                ['dotted', 'dashed', 'solid', 'dotted']
            ):
                ax.hlines(
                    y=stat_y,
                    xmin=x_pos - 0.2,
                    xmax=x_pos + 0.2,
                    colors='black',
                    linestyles=style,
                    linewidth=1
                )

        legend_elements = [
            Line2D([0], [0], color='black', linestyle='solid', label='Mean'),
            Line2D([0], [0], color='black', linestyle='dashed', label='Median'),
            Line2D([0], [0], color='black', linestyle='dotted', label='Q25 / Q75'),
        ]
    else:
        legend_elements = []



    # Stripplot
    if show_points:
        sns.stripplot(data=df, x=x_col, y=y_col, alpha=0.2, size=2.5, jitter=0.2, color="black", ax=ax)

    # Trend lines
    for trend in trend_lines:
        if trend == 'mean':
            sns.lineplot(
                data=df,
                x=x_col,
                y=y_col,
                estimator='mean',
                errorbar=errorbar_lineplot_mean,
                color='blue',
                label='Mean ± SD',
                ax=ax
            )
        elif trend == 'median':
            sns.lineplot(
                data=df,
                x=x_col,
                y=y_col,
                estimator='median',
                errorbar=errorbar_lineplot_median,
                color='orange',
                label='Median',
                ax=ax
            )
        else:
            raise ValueError(f"Unsupported trend type: {trend}")

    # Annotations below x-axis
    if annotate_counts:
        pair_counts = df.groupby(x_col)[y_col].count()
        biopsy_counts = (
            df[[x_col, 'Patient ID', 'Bx index']]
            .drop_duplicates()
            .groupby(x_col)
            .size()
        )

        y_offset = 0.15 * (y_max - y_min)
        y_annotation = y_min - y_offset

        for x_val in pair_counts.index:
            x_pos = tick_map.get(x_val, x_val)  # ✅ No int() here
            count = pair_counts.get(x_val, 0)
            unique = biopsy_counts.get(x_val, 0)
            annotation = f"n={count}\nu={unique}"
            ax.text(
                x=x_pos,
                y=y_annotation,
                s=annotation,
                ha='center',
                va='top',
                fontsize=8,
                rotation=60,
                alpha=0.7
            )


    ax.set_title(title, fontsize=16)
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    ax.set_xlabel(xlabel)
    x_vals = sorted(map(int, df[x_col].unique()))
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)

    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max * (1.05 if annotate_counts else 1))
    ax.grid(True)

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    if legend_elements:
        handles += legend_elements
        labels += [le.get_label() for le in legend_elements]

    ax.legend(handles=handles, labels=labels, loc='upper right')

    plt.tight_layout()

    png_path = os.path.join(save_dir, f"{file_name}.png")
    svg_path = os.path.join(save_dir, f"{file_name}.svg")
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path)
    plt.close()

    print(f"Plot saved as:\n - {png_path}\n - {svg_path}")









def plot_dose_vs_length_with_summary_mutlibox(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        save_dir: str,
        file_name: str,
        title: str = None,                 # optional title
        figsize=(12, 7),
        dpi=300,
        show_points=False,                 # optional raw points overlay
        violin_or_box: str = 'violin',     # 'box' or 'violin'
        trend_lines: list = ['mean'],      # e.g., ['mean'] or ['median'] or both
        annotate_counts=True,              # enable counts
        annotation_box=False,              # place counts in external box
        y_trim=False,
        y_min_quantile=0.05,
        y_max_quantile=0.95,
        y_min_fixed=None,
        y_max_fixed=None,
        xlabel: str = None,
        ylabel: str = None,
        errorbar_lineplot_mean: str = 'ci',
        errorbar_lineplot_median: str = None,
        title_font_size: int = 16,         # independent font sizes
        axis_label_font_size: int = 14,
        tick_label_font_size: int = 12,
        multi_pairs: list = None,           # list of (Patient ID, Bx index) tuples
        metric_family: str | None = None,  # 'dose' or 'grad'
        y_tick_decimals: int | None = 1,
    ):
    plt.ioff()

    # Decide dose vs gradient
    if metric_family in ('dose', 'grad'):
        is_grad = (metric_family == 'grad')
    else:
        # fallback: try to infer (kept for backward compat)
        is_grad = ('grad' in (y_col or '').lower()) or ('gamma' in (y_col or '').lower())

    # ---- prep ----
    df = df.copy()
    df[x_col] = df[x_col].astype(str)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_axisbelow(True)  # grid under data

    from matplotlib.ticker import StrMethodFormatter



    # Axis labels consistent with your paper
    x_label_latex = r'$\ell_k$ (mm)'
    y_label_latex = r'$\mathcal{S}_b^{G}(\ell_k)\ \mathrm{(Gy\,mm^{-1})}$' if is_grad else r'$\mathcal{S}_b^{D}(\ell_k)\ \mathrm{(Gy)}$'

    # Apply labels (override any plain-text defaults)
    ax.set_xlabel(xlabel if xlabel else x_label_latex, fontsize=axis_label_font_size)
    ax.set_ylabel(ylabel if ylabel else y_label_latex, fontsize=axis_label_font_size)

    # Optional: force tick precision on y (e.g., 1 decimal)
    if y_tick_decimals is not None:
        fmt = f'{{x:.{int(y_tick_decimals)}f}}'
        ax.yaxis.set_major_formatter(StrMethodFormatter(fmt))
        ax.get_yaxis().get_offset_text().set_visible(False)


    # Apply tick font size globally for this figure
    """
    plt.rcParams.update({
        'xtick.labelsize': tick_label_font_size,
        'ytick.labelsize': tick_label_font_size
    })
    """
    ax.tick_params(axis='both', which='both', labelsize=tick_label_font_size)

    # ---- y-limits ----
    if y_min_fixed is not None:
        y_min = y_min_fixed
    elif y_trim:
        y_min = df[y_col].quantile(y_min_quantile)
    else:
        y_min = df[y_col].min()

    if y_max_fixed is not None:
        y_max = y_max_fixed
    elif y_trim:
        max_quantiles_list = [
            df[df[x_col] == x_val][y_col].quantile(y_max_quantile)
            for x_val in df[x_col].unique()
        ]
        y_max = max(max_quantiles_list) if max_quantiles_list else df[y_col].quantile(y_max_quantile)
    else:
        y_max = df[y_col].max()

    # ---- multi-pair filter & id ----
    if multi_pairs:
        df = df[df[['Patient ID', 'Bx index']].apply(tuple, axis=1).isin(multi_pairs)]
        df['pair_id'] = df['Patient ID'].astype(str) + "-" + df['Bx index'].astype(str)

    # ---- base plot (boxes / violins) ----
    if multi_pairs:
        n_pairs = df['pair_id'].nunique()
        box_palette = sns.color_palette("pastel", n_colors=n_pairs)

        if violin_or_box == 'violin':
            vp = sns.violinplot(
                data=df, x=x_col, y=y_col, hue='pair_id',
                inner=None, ax=ax, saturation=0.7, linewidth=0.8,
                palette=box_palette, legend=False  # optional to avoid dup legends
            )
            for c in vp.collections:
                c.set_alpha(0.5)

        elif violin_or_box == 'box':
            bp = sns.boxplot(
                data=df, x=x_col, y=y_col, hue='pair_id',
                ax=ax, showfliers=False, saturation=0.7, linewidth=0.8,
                palette=box_palette, legend=False  # optional
            )
            for patch in bp.artists:
                patch.set_alpha(0.5)
        else:
            raise ValueError("`violin_or_box` must be either 'violin' or 'box'")
    else:
        if violin_or_box == 'violin':
            vp = sns.violinplot(
                data=df, x=x_col, y=y_col, inner=None,
                color="lightgray", ax=ax, linewidth=0.8
            )
            for c in vp.collections:
                c.set_alpha(0.4)
        elif violin_or_box == 'box':
            bp = sns.boxplot(
                data=df, x=x_col, y=y_col, color="lightgray",
                ax=ax, showfliers=False, linewidth=0.8
            )
            for patch in bp.artists:
                patch.set_alpha(0.4)
        else:
            raise ValueError("`violin_or_box` must be either 'violin' or 'box'")




    # ---- trend lines (always on top) ----
    if multi_pairs:
        n_pairs = df['pair_id'].nunique()
        line_colors = sns.color_palette("tab10", n_colors=n_pairs)

        line_kws = dict(
            zorder=10, linewidth=2.5,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
        )

        for idx, pair in enumerate(df['pair_id'].unique()):
            df_pair = df[df['pair_id'] == pair]
            if 'mean' in trend_lines:
                sns.lineplot(
                    data=df_pair, x=x_col, y=y_col,
                    estimator='mean', errorbar=None,
                    label=pair, ax=ax, color=line_colors[idx], **line_kws
                )
            if 'median' in trend_lines:
                sns.lineplot(
                    data=df_pair, x=x_col, y=y_col,
                    estimator='median', errorbar=None,
                    label=f"{pair} (median)", ax=ax,
                    color=line_colors[idx], **line_kws
                )
    else:
        # single-color boxes (lightgray) already fine
        line_kws = dict(
            zorder=10, linewidth=2.5,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
        )
        if 'mean' in trend_lines:
            sns.lineplot(
                data=df, x=x_col, y=y_col,
                estimator='mean', errorbar=errorbar_lineplot_mean,
                color='tab:blue', label='Mean ± CI' if errorbar_lineplot_mean else 'Mean',
                ax=ax, **line_kws
            )
        if 'median' in trend_lines:
            sns.lineplot(
                data=df, x=x_col, y=y_col,
                estimator='median', errorbar=errorbar_lineplot_median,
                color='tab:orange', label='Median', ax=ax, **line_kws
            )



    # ---- optional points (behind trend lines) ----
    if show_points:
        if multi_pairs:
            sns.stripplot(
                data=df, x=x_col, y=y_col, hue='pair_id',
                dodge=True, jitter=0.25, size=3, alpha=0.6,
                linewidth=0.3, edgecolor='black', ax=ax,
                zorder=2, legend=False  # avoid duplicate legend entries
            )
        else:
            sns.stripplot(
                data=df, x=x_col, y=y_col,
                jitter=0.25, size=3, alpha=0.6, color="black",
                linewidth=0.0, ax=ax, zorder=2
            )

    # ---- counts / annotation box ----
    if annotate_counts:
        # Build grouping keys robustly
        group_keys = [x_col] + (['pair_id'] if 'pair_id' in df.columns else [])
        counts = df.groupby(group_keys, dropna=False)[y_col].count()

        if annotation_box:
            if 'pair_id' in df.columns:
                lines = [f"{pid} @ {xv}: n={n}" for (xv, pid), n in counts.items()]
            else:
                lines = [f"{xv}: n={n}" for xv, n in counts.items()]
            ann_text = "\n".join(lines)

            ax.annotate(
                ann_text, xy=(1.02, 0.5), xycoords='axes fraction',
                va='center', ha='left', fontsize=tick_label_font_size,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        # (No inline annotations below the axis)

    # ---- labels, title, limits, grid ----
    if title:
        ax.set_title(title, fontsize=title_font_size)


    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max * 1.05)
    ax.grid(True)

    # Optional: if x are numeric-like (e.g., "5", "10"), show sorted ints as tick labels
    try:
        x_vals_sorted = sorted(map(int, df[x_col].unique()))
        ax.set_xticks(range(len(x_vals_sorted)))
        ax.set_xticklabels(x_vals_sorted)
    except Exception:
        # leave seaborn's categorical ticks as-is
        pass

    # ---- legend de-duplication ----
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        ax.legend(*zip(*uniq), loc='upper right')

    plt.tight_layout()

    # ---- save ----
    png_path = os.path.join(save_dir, f"{file_name}.png")
    svg_path = os.path.join(save_dir, f"{file_name}.svg")
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path)
    plt.close()

    print(f"Plot saved as:\n - {png_path}\n - {svg_path}")











def plot_dose_vs_length_with_summary_cohort_v2(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_dir: str,
    file_name: str,
    title: str = None,
    figsize=(10, 6),
    dpi=300,

    violin_or_box: str = "box",         # "box" or "violin"
    show_points: bool = False,          # optional raw points overlay

    # global trend lines across the entire cohort (pooled)
    trend_lines: tuple[str, ...] = ("mean",),  # ("mean",), ("median",), or both
    errorbar_lineplot_mean: str | None = None,    # e.g. "ci" or None
    errorbar_lineplot_median: str | None = None,  # e.g. "ci" or None

    # NEW: family of mean curves for each (Patient ID, Bx index)
    show_pair_mean_curves: bool = False,
    pair_cols: tuple[str, str] = ("Patient ID", "Bx index"),
    pairs_filter: list[tuple] | None = None,      # optional list of (Patient ID, Bx index)
    pair_mean_estimator: str = "mean",            # "mean" or "median" for the per-pair curve
    pair_line_alpha: float = 0.25,
    pair_line_width: float = 1.2,
    pair_line_palette: str = "tab10",
    pair_line_outline: bool = False,              # NEW: turn off the “white stroke” pop
    show_pair_legend: bool = False,
    max_legend_pairs: int = 12,

    # counts
    annotate_counts: bool = True,
    annotation_box: bool = False,       # external box only
    counts_mode: str = "N_b",          # "both", "N_b", or "N_p"


    # y-limits
    y_trim: bool = False,
    y_min_quantile: float = 0.05,
    y_max_quantile: float = 0.95,
    y_min_fixed: float | None = None,
    y_max_fixed: float | None = None,

    # labeling / typography
    xlabel: str | None = None,
    ylabel: str | None = None,
    metric_family: str | None = None,   # "dose" or "grad"
    y_tick_decimals: int | None = 1,

    title_font_size: int = 20,
    axis_label_font_size: int = 16,
    tick_label_font_size: int = 14,
    annotation_label_font_size: int = 14,


    # absolute label control (auto if None)
    y_is_abs: bool | None = None,

    # NEW: box/violin styling
    box_color: str = "lightgray",
    box_alpha: float = 0.4,

    # NEW: y-limit handling for family curves
    include_pair_curves_in_ylim: bool = True,
    pair_curves_ylim_quantile: float = 1.0,   # 1.0 = include max; try 0.995 to ignore extreme outliers
    ylim_top_pad: float = 0.05,              # 5% headroom (you already effectively do this)

):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as pe
    from matplotlib.ticker import StrMethodFormatter

    plt.ioff()
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()

    # ---- Decide dose vs grad ----
    if metric_family in ("dose", "grad"):
        is_grad = (metric_family == "grad")
    else:
        is_grad = ("grad" in (y_col or "").lower()) or ("gamma" in (y_col or "").lower())

    # ---- Decide abs label ----
    if y_is_abs is None:
        y_is_abs = ("abs" in (y_col or "").lower()) or ("absolute" in (y_col or "").lower())

    # ---- x ordering: enforce ordered categorical (fixes the ℓk=2 shift) ----
    # We plot x as strings everywhere (boxes + lines) so seaborn/matplotlib share the same coordinate system.
    x_vals = df[x_col].dropna().unique()
    try:
        x_order_int = sorted([int(v) for v in x_vals])
        x_order = [str(v) for v in x_order_int]
        df[x_col] = df[x_col].astype(int).astype(str)
    except Exception:
        # fallback categorical
        x_order = sorted([str(v) for v in x_vals])
        x_order_int = None
        df[x_col] = df[x_col].astype(str)

    df[x_col] = pd.Categorical(df[x_col], categories=x_order, ordered=True)

    # ---- Figure/Axes ----
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_axisbelow(True)

    # ---- Paper-consistent LaTeX axis labels ----
    x_label_latex = r'$\ell_k$ (mm)'

    if is_grad:
        base = r'\mathcal{S}_{\mathrm{cohort}}^{G}(\ell_k)'
        units = r'\mathrm{(Gy\,mm^{-1})}'
    else:
        base = r'\mathcal{S}_{\mathrm{cohort}}^{D}(\ell_k)'
        units = r'\mathrm{(Gy)}'

    # this effectively does nothing for now because we dont look at the non-absolute differences 
    if y_is_abs:
        y_label_latex = rf'${base}\ {units}$'
    else:
        y_label_latex = rf'${base}\ {units}$'

    ax.set_xlabel(xlabel if xlabel else x_label_latex, fontsize=axis_label_font_size)
    ax.set_ylabel(ylabel if ylabel else y_label_latex, fontsize=axis_label_font_size)

    # y tick formatting
    if y_tick_decimals is not None:
        fmt = f'{{x:.{int(y_tick_decimals)}f}}'
        ax.yaxis.set_major_formatter(StrMethodFormatter(fmt))
        ax.get_yaxis().get_offset_text().set_visible(False)

    ax.tick_params(axis="both", which="both", labelsize=tick_label_font_size)

    # ---- y-limits ----
    if y_min_fixed is not None:
        y_min = y_min_fixed
    elif y_trim:
        y_min = df[y_col].quantile(y_min_quantile)
    else:
        y_min = df[y_col].min()

    if y_max_fixed is not None:
        y_max = y_max_fixed
    elif y_trim:
        max_qs = [
            df.loc[df[x_col] == xv, y_col].quantile(y_max_quantile)
            for xv in x_order
            if (df[x_col] == xv).any()
        ]
        y_max = max(max_qs) if max_qs else df[y_col].quantile(y_max_quantile)
    else:
        y_max = df[y_col].max()

    # ---- base plot (cohort pooled boxes/violins) ----
    if violin_or_box == "violin":
        vp = sns.violinplot(
            data=df, x=x_col, y=y_col,
            order=x_order,
            inner=None, color=box_color,
            ax=ax, linewidth=0.8
        )
        # violin bodies are in collections
        for c in vp.collections:
            c.set_alpha(box_alpha)



    elif violin_or_box == "box":
        bp = sns.boxplot(
            data=df, x=x_col, y=y_col,
            order=x_order,
            ax=ax,
            showfliers=False,

            # thicker outlines for the box, whiskers, caps
            linewidth=1.3,
            boxprops=dict(
                facecolor=box_color,
                edgecolor="black",
                linewidth=1.3,
            ),
            whiskerprops=dict(
                color="black",
                linewidth=1.3,
            ),
            capprops=dict(
                color="black",
                linewidth=1.3,
            ),

            # really emphasize the median line
            medianprops=dict(
                color="black",
                linewidth=2.0,
            ),
        )

        # keep your alpha control on the fill
        for patch in bp.artists:
            patch.set_alpha(box_alpha)


    else:
        raise ValueError("`violin_or_box` must be either 'violin' or 'box'")

    # ---- optional points (behind lines) ----
    if show_points:
        sns.stripplot(
            data=df, x=x_col, y=y_col,
            order=x_order,
            jitter=0.25, size=3, alpha=0.25, color="black",
            linewidth=0.0, ax=ax, zorder=2
        )

    # ---- global cohort trend line(s) ----
    # keep this prominent (black with white stroke)
    global_line_kws = dict(
        zorder=10, linewidth=2.8,
        path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]
    )

    if trend_lines:
        if "mean" in trend_lines:
            sns.lineplot(
                data=df, x=x_col, y=y_col,
                estimator="mean", errorbar=errorbar_lineplot_mean,
                color="black", label=None,
                ax=ax, sort=False, **global_line_kws
            )
        if "median" in trend_lines:
            sns.lineplot(
                data=df, x=x_col, y=y_col,
                estimator="median", errorbar=errorbar_lineplot_median,
                color="dimgray", label=None,
                ax=ax, sort=False, **global_line_kws
            )

    # ---- per-(Patient ID, Bx index) mean curves (optional) ----
    if show_pair_mean_curves:
        pid_col, bxi_col = pair_cols
        if pid_col not in df.columns or bxi_col not in df.columns:
            raise ValueError(f"pair_cols {pair_cols} not found in df columns")

        df_pairs = df.copy()
        if pairs_filter is not None:
            df_pairs = df_pairs[df_pairs[[pid_col, bxi_col]].apply(tuple, axis=1).isin(pairs_filter)]

        df_pairs["pair_id"] = df_pairs[pid_col].astype(str) + "-" + df_pairs[bxi_col].astype(str)

        agg_func = "mean" if pair_mean_estimator == "mean" else "median"
        pair_curve = (
            df_pairs.groupby(["pair_id", x_col], dropna=False, observed=False)[y_col]
            .agg(agg_func)
            .reset_index()
        )

        if include_pair_curves_in_ylim and not pair_curve.empty:
            # robust control via quantile (1.0 uses max)
            pair_ymax = float(pair_curve[y_col].quantile(pair_curves_ylim_quantile))
            if np.isfinite(pair_ymax):
                y_max = max(y_max, pair_ymax)

        pair_ids = pair_curve["pair_id"].unique().tolist()
        if pair_ids:
            colors = sns.color_palette(pair_line_palette, n_colors=len(pair_ids))

            # optional outline for pair curves (default off)
            pair_pe = (
                [pe.Stroke(linewidth=pair_line_width + 1.2, foreground="white"), pe.Normal()]
                if pair_line_outline else
                None
            )

            for idx, pair in enumerate(pair_ids):
                sub = pair_curve[pair_curve["pair_id"] == pair].copy()
                # ensure same ordered categorical x
                sub[x_col] = pd.Categorical(sub[x_col].astype(str), categories=x_order, ordered=True)
                sub = sub.sort_values(x_col)

                ax.plot(
                    sub[x_col].values,
                    sub[y_col].values,
                    color=colors[idx],
                    alpha=pair_line_alpha,
                    linewidth=pair_line_width,
                    zorder=9,
                    path_effects=pair_pe,
                    label=pair if show_pair_legend else None
                )

            if show_pair_legend:
                handles, labels = ax.get_legend_handles_labels()
                if len(labels) > max_legend_pairs:
                    handles = handles[:max_legend_pairs]
                    labels = labels[:max_legend_pairs]
                    labels[-1] = labels[-1] + " …"
                ax.legend(handles, labels, loc="upper right", fontsize=max(tick_label_font_size - 2, 8))

    """
    # ---- counts / annotation box ----
    if annotate_counts:
        counts = df.groupby([x_col], dropna=False, observed=False)[y_col].count()

        if annotation_box:
            ann_text = "\n".join([f"{xv}: n={int(n)}" for xv, n in counts.items()])
            ax.annotate(
                ann_text, xy=(1.02, 0.5), xycoords="axes fraction",
                va="center", ha="left", fontsize=tick_label_font_size,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    """
    # ---- counts / annotation box ----
    if annotate_counts:
        # Biopsy counts per ℓ_k
        counts_bx = (
            df.dropna(subset=[x_col, "Patient ID", "Bx index"])
              .groupby(x_col, dropna=False, observed=False)[["Patient ID", "Bx index"]]
              .apply(lambda g: g.drop_duplicates().shape[0])
        )

        # Total sample counts per ℓ_k (rows per box)
        counts_rows = df.groupby(x_col, dropna=False, observed=False)[y_col].count()

        # helper: build one label string based on counts_mode
        def _make_label(xv):
            n_bx = int(counts_bx.get(xv, 0))
            n_rows = int(counts_rows.get(xv, 0))

            if counts_mode == "N_b":
                return rf"$N_b={n_bx}$"
            elif counts_mode == "N_p":
                return rf"$N_p={n_rows}$"
            else:  # "both" (default)
                return rf"$N_b={n_bx}$" + ", " + rf"$N_p={n_rows}$"

        if annotation_box:
            # side-box behaviour: one line per ℓ_k
            ann_text = "\n".join(
                [rf"{xv}: {_make_label(xv)}" for xv in counts_rows.index]
            )
            ax.annotate(
                ann_text, xy=(1.02, 0.5), xycoords="axes fraction",
                va="center", ha="left", fontsize=annotation_label_font_size,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        else:
            # text directly under each ℓ_k tick
            x_positions = {cat: i for i, cat in enumerate(x_order)}
            trans = ax.get_xaxis_transform()
            y_text = -0.16

            for xv in x_order:
                cat = str(xv)
                if cat not in x_positions:
                    continue

                x_idx = x_positions[cat]
                label_text = _make_label(xv)

                ax.text(
                    x_idx,
                    y_text,
                    label_text,
                    transform=trans,
                    ha="right",
                    va="top",
                    fontsize=annotation_label_font_size,
                    rotation=35,
                    rotation_mode="anchor",
                    clip_on=False,
                )




    # ---- title / limits / grid ----
    if title:
        ax.set_title(title, fontsize=title_font_size)

    y_span = (y_max - y_min) if np.isfinite(y_max - y_min) else 1.0

    # If user pinned the bottom (y_min_fixed), do NOT pad below
    if y_min_fixed is not None:
        bottom = y_min
    else:
        bottom = y_min - 0.05 * y_span

    top = y_max * (1.0 + ylim_top_pad)

    ax.set_ylim(bottom, top)


    ax.grid(True)

    # display integer tick labels if we have them
    if x_order_int is not None:
        ax.set_xticks(np.arange(len(x_order)))
        ax.set_xticklabels(x_order_int)


    plt.tight_layout()

    # ---- save ----
    png_path = os.path.join(save_dir, f"{file_name}.png")
    svg_path = os.path.join(save_dir, f"{file_name}.svg")
    pdf_path = os.path.join(save_dir, f"{file_name}.pdf")  # <--- NEW

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight", format="pdf")  # <--- NEW

    plt.close()

    print(
        "Plot saved as:\n"
        f" - {png_path}\n"
        f" - {svg_path}\n"
        f" - {pdf_path}"  # <--- NEW
    )
















def plot_global_dosimetry_boxplot(
        df: pd.DataFrame,
        dose_key: str,  # e.g. 'Dose (Gy)'
        subindices: list,  # e.g. ['mean', 'min', 'max', 'quantile_05']
        save_dir: str,
        file_name: str,
        title: str = '',
        figsize=(10, 6),
        dpi=300,
        xlabel: str = '',
        ylabel: str = '',
        showfliers: bool = False,
        label_map: dict = None,
        horizontal: bool = False

    ):
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract and reshape data
    selected_df = df.loc[:, (dose_key, subindices)]
    selected_df.columns = selected_df.columns.droplevel(0)  # Drop 'Dose (Gy)' level
    long_df = selected_df.melt(var_name='Metric', value_name='Value')

    # Apply label mapping if provided
    if label_map:
        long_df['Metric'] = long_df['Metric'].map(label_map).fillna(long_df['Metric'])

    n_patients = df[('Patient ID', '')].nunique()
    # find the number of unique biopsies by finding the number of unique 'Bx ID' and Patient ID pairs by using grouping
    df_alt = df.copy()  # Avoid modifying the original DataFrame
    df_alt['Bx ID'] = df_alt['Bx ID'].astype(str)  # Ensure 'Bx ID' is string for grouping
    n_biopsies = df_alt.groupby(['Patient ID', 'Bx ID']).ngroups
    # Set title if not provided
    if not title:
        title = f"Global Dosimetry Boxplot for {dose_key}"

    annotation_text = f"Patients: {n_patients}\nBiopsies: {n_biopsies}"


    # Create the boxplot
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        box = sns.boxplot(
            data=long_df,
            y='Metric',
            x='Value',
            showfliers=showfliers,
            #boxprops=dict(edgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black', markersize=2),
            ax=ax,
            fill=False,  # No fill color for boxes,
            orient='h'  # Horizontal orientation
        )
    else:
        box = sns.boxplot(
            data=long_df,
            x='Metric',
            y='Value',
            showfliers=showfliers,
            #boxprops=dict(edgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black', markersize=2),
            ax=ax,
            fill=False  # No fill color for boxes
            )
    
    # Annotate mean as a dot and median as a line + label
    means = long_df.groupby('Metric')['Value'].mean()

    for metric, mean_val in means.items():
        
        # Mean as dot
        if horizontal:
            ax.plot(mean_val, metric, 'x', color='black', markersize=6, label='_nolegend_')
        else:
            ax.plot(metric, mean_val, 'x', color='black', markersize=6, label='_nolegend_')

        
        
        


    plt.title(title, fontsize=16)

    plt.xlabel(xlabel or 'Metric')
    plt.ylabel(ylabel or dose_key)
    plt.grid(True)

    # Annotation box on top-right corner
    ax.text(
        0.98, 0.98,
        annotation_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
    )

    # remove horizontal grid lines if horizontal
    if horizontal:
        ax.yaxis.grid(False)
    else:
        ax.xaxis.grid(False)

    # make font size of y and x tick labels 14
    ax.tick_params(axis='both', which='major', labelsize=14)

    # make font size of x and y labels 14
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)

    plt.tight_layout()

    # Save
    png_path = os.path.join(save_dir, f"{file_name}.png")
    svg_path = os.path.join(save_dir, f"{file_name}.svg")
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path)
    plt.close()

    print(f"Boxplot saved to:\n - {png_path}\n - {svg_path}")










def production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring_no_dose_cohort_v2(sp_bx_dose_distribution_all_trials_df,
                                                                                          sp_patient_and_sp_bx_dose_dataframe_by_voxel,
                                                                                          sp_patient_binom_df,
                                                                                          svg_image_width,
                                                                                          svg_image_height,
                                                                                          dpi,
                                                                                          ridge_line_dose_and_binom_general_plot_name_string,
                                                                                          patient_sp_output_figures_dir_dict,
                                                                                          cancer_tissue_label):
    plt.ioff()
    

    df_dose = sp_bx_dose_distribution_all_trials_df
    df_dose = misc_tools.convert_categorical_columns(df_dose, ['Voxel index', "Dose (Gy)"], [int, float])

    df_dose_stats_by_voxel = sp_patient_and_sp_bx_dose_dataframe_by_voxel
    df_tissue = sp_patient_binom_df[sp_patient_binom_df["Tissue class"] == cancer_tissue_label]

    colors = ["green", "blue", "black"]
    cmap = LinearSegmentedColormap.from_list("GreenBlueRed", colors, N=10)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    def annotate_and_color_v2(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = df_dose_stats_by_voxel[(df_dose_stats_by_voxel['Voxel index'] == label_float) & 
                           (df_dose_stats_by_voxel['Patient ID'] == patient_id) & 
                           (df_dose_stats_by_voxel['Bx index'] == bx_index)].iloc[0]

        nominal_dose = voxel_row[('Dose (Gy)','nominal')]
        mean = voxel_row[('Dose (Gy)','mean')]
        std = voxel_row[('Dose (Gy)','std')]
        max_density_dose = voxel_row[('Dose (Gy)','argmax_density')]
        voxel_begin = voxel_row[('Voxel begin (Z)', '')]
        voxel_end = voxel_row[('Voxel end (Z)', '')]

        q05_dose = voxel_row[('Dose (Gy)','quantile_05')]
        q25_dose = voxel_row[('Dose (Gy)','quantile_25')]
        q50_dose = voxel_row[('Dose (Gy)','quantile_50')]
        q75_dose = voxel_row[('Dose (Gy)','quantile_75')]
        q95_dose = voxel_row[('Dose (Gy)','quantile_95')]

        tissue_voxel = df_tissue[(df_tissue['Voxel index'] == label_float) & 
                                 (df_tissue['Patient ID'] == patient_id) & 
                                 (df_tissue['Bx index'] == bx_index)]
        binom_mean = tissue_voxel["Binomial estimator"].mean()

        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin:.1f}, {voxel_end:.1f}) | Tumor tissue score: {binom_mean:.2f}\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(1.03, 0.7, annotation_text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        ax.axvline(x=max_density_dose, color='magenta', linestyle='-', label='Max Density (Gy)')
        ax.axvline(x=mean, color='orange', linestyle='-', label='Mean (Gy)')
        ax.axvline(x=nominal_dose, color='red', linestyle='-', label='Nominal (Gy)')

        # Added loop for plotting dotted gray vertical lines for each quantile
        for quantile_value in [q05_dose, q25_dose, q50_dose, q75_dose, q95_dose]:
            ax.axvline(x=quantile_value, color='gray', linestyle='--', linewidth=1)

        density_color = cmap(norm(binom_mean))

        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        y_density = kde(x_grid)

        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        scaled_density = y_density * scaling_factor

        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)

    max_95th_quantile = df_dose_stats_by_voxel[('Dose (Gy)','quantile_95')].max()
    min_5th_quantile = df_dose_stats_by_voxel[('Dose (Gy)','quantile_05')].min()

    # Define legend handles
    legend_handles = [
        Line2D([0], [0], color='magenta', lw=2, linestyle='-', label='Max Density (Gy)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='-', label='Mean (Gy)'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Nominal (Gy)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Quantiles (05, 25, 50, 75, 95)')
    ]

    for (patient_id, bx_index), group in df_dose.groupby(['Patient ID', 'Bx index']):
        bx_id = group.iloc[0]['Bx ID']

        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}


        # Ensure that the FacetGrid is only created for valid voxel indices
        valid_voxel_indices = [voxel for voxel in group['Voxel index'].unique() if voxel in palette_black]
        if len(valid_voxel_indices) == 0:
            print(f"No valid voxel indices for Patient ID: {patient_id}, Bx index: {bx_index}")
            continue
        
        #g = sns.FacetGrid(group[group['Voxel index'].isin(valid_voxel_indices)], row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)

        g.map(annotate_and_color_v2, "Dose (Gy)")

        g.set(xlim=(min_5th_quantile, max_95th_quantile))

        g.fig.subplots_adjust(right=0.53, left=0.07, top=0.95, bottom=0.05)
        cbar_ax = g.fig.add_axes([0.9, 0.2, 0.03, 0.6])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)
        g.fig.text(0.88, 0.5, 'Tumor tissue score', va='center', rotation='vertical', fontsize=12)

        for ax in g.axes.flat:
            ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
            ax.set_axisbelow(True)

        # Update legend to include quantiles
        g.fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.9, 1), ncol=1, frameon=True, facecolor='white')

        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_and_binom_general_plot_name_string+' - '+str(patient_id)+' - '+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)

    #df_dose = dataframe_builders.convert_columns_to_categorical_and_downcast(df_dose, threshold=0.25)


def plot_dose_ridge_for_single_biopsy(
    dose_df,
    dose_stats_df,
    binom_df,
    save_dir,
    fig_title_suffix,
    fig_name_suffix,
    cancer_tissue_label,
    fig_scale=1.0,
    dpi=300,
    add_text_annotations=True,
    x_label="Dose (Gy)",
    y_label="Biopsy Axial Dimension (mm)",
    space_between_ridgeline_padding_multiplier = 1.2,
    ridgeline_vertical_padding_value = 0.25
):


    plt.ioff()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    dose_df = misc_tools.convert_categorical_columns(dose_df, ['Voxel index', 'Dose (Gy)'], [int, float])
    coloring_enabled = binom_df is not None

    if coloring_enabled:
        binom_df = binom_df[binom_df["Tissue class"] == cancer_tissue_label]
        cmap = LinearSegmentedColormap.from_list("GreenBlueBlack", ["green", "blue", "black"], N=10)
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=cmap)

    patient_id = dose_df['Patient ID'].iloc[0]
    bx_index = dose_df['Bx index'].iloc[0]
    bx_id = dose_df['Bx ID'].iloc[0]
    num_trials = dose_df["MC trial num"].max() + 1

    def annotate_and_fill(x, color, label, **kwargs):
        label_val = float(label)
        voxel_stats = dose_stats_df[
            (dose_stats_df['Voxel index'] == label_val) &
            (dose_stats_df['Patient ID'] == patient_id) &
            (dose_stats_df['Bx index'] == bx_index)
        ].iloc[0]

        nominal = voxel_stats[('Dose (Gy)', 'nominal')]
        mean = voxel_stats[('Dose (Gy)', 'mean')]
        std = voxel_stats[('Dose (Gy)', 'std')]
        max_dens = voxel_stats[('Dose (Gy)', 'argmax_density')]
        z_start = voxel_stats[('Voxel begin (Z)', '')]
        z_end = voxel_stats[('Voxel end (Z)', '')]
        q_vals = [voxel_stats[('Dose (Gy)', f'quantile_{q:02}')] for q in [5, 25, 50, 75, 95]]

        ax = plt.gca()
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        y_vals = kde(x_grid)
        y_scaled = y_vals / np.max(y_vals) if np.max(y_vals) > 0 else y_vals

        # Fill
        if coloring_enabled:
            binom_mean = binom_df[
                (binom_df['Voxel index'] == label_val) &
                (binom_df['Patient ID'] == patient_id) &
                (binom_df['Bx index'] == bx_index)
            ]["Binomial estimator"].mean()
            fill_color = cmap(norm(binom_mean))
        else:
            fill_color = "gray"

        ax.fill_between(x_grid, y_scaled, alpha=0.5, color=fill_color)

        # Vertical lines
        ax.axvline(x=max_dens, color='magenta', linestyle='-', linewidth=1)
        ax.axvline(x=mean, color='orange', linestyle='-', linewidth=1)
        ax.axvline(x=nominal, color='red', linestyle='-', linewidth=1)
        for qv in q_vals:
            ax.axvline(x=qv, color='gray', linestyle='--', linewidth=1)

        # Annotations
        if add_text_annotations:
            annotation = (
                f"Segment: ({z_start:.1f}, {z_end:.1f}) mm"
                + (f" | Tumor score: {binom_mean:.2f}" if coloring_enabled else "")
                + f"\nMean: {mean:.2f} Gy | SD: {std:.2f} | Max Dens: {max_dens:.2f} | Nominal: {nominal:.2f}"
            )
            ax.text(1.02, 0.5, annotation, transform=ax.transAxes, ha='left', va='center', fontsize=8, color=color)

        # Y-axis tick for ridge
        ax.set_yticks([0.5])
        #ax.set_yticklabels([f"{z_start:.1f}–{z_end:.1f} mm"])
        ax.set_yticklabels([f""]) # clear y-tick labels for ridgeline effect
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', which='both', labelbottom=True, length=3, width=0.8)
        

    # Plot setup
    max_q95 = dose_stats_df[('Dose (Gy)', 'quantile_95')].max()
    min_q05 = dose_stats_df[('Dose (Gy)', 'quantile_05')].min()
    voxel_ids = dose_df['Voxel index'].unique()
    palette = {v: 'black' for v in voxel_ids}

    #
    space_between_ridgeline_padding_multiplier = 1.2
    g = sns.FacetGrid(dose_df, row='Voxel index', hue='Voxel index', aspect=15, height=1*space_between_ridgeline_padding_multiplier, palette=palette)

    g.map(annotate_and_fill, 'Dose (Gy)')

    # Build y-tick positions and labels (axial mm)
    sorted_voxels = sorted(voxel_ids)
    z_ticks = []
    z_labels = []

    # Voxel begin of the first voxel
    first_voxel = sorted_voxels[0]
    first_z_begin = dose_stats_df[
        (dose_stats_df['Voxel index'] == first_voxel) &
        (dose_stats_df['Patient ID'] == patient_id) &
        (dose_stats_df['Bx index'] == bx_index)
    ]['Voxel begin (Z)'].values[0]
    z_ticks.append(0)
    z_labels.append(f"{first_z_begin:.1f} mm")

    # For each voxel (except first), get voxel end
    for i, voxel in enumerate(sorted_voxels):
        z_end = dose_stats_df[
            (dose_stats_df['Voxel index'] == voxel) &
            (dose_stats_df['Patient ID'] == patient_id) &
            (dose_stats_df['Bx index'] == bx_index)
        ]['Voxel end (Z)'].values[0]
        z_ticks.append(i + 1)
        z_labels.append(f"{z_end:.1f} mm")

    # Set ticks on the shared y-axis (reversed so 0 mm at top)
    g.fig.subplots_adjust(hspace=ridgeline_vertical_padding_value) # 0.1 is default
    #for i, ax in enumerate(g.axes.flat):  # show current segment label on each ridge
    #ax.set_yticks([0.5])
    #ax.set_yticklabels([f"V{sorted_voxels[i]} ({z_labels[i]}-{z_labels[i+1]})" for i in range(len(sorted_voxels) - 1)])
    #y_tick_labels = [f"V{sorted_voxels[i]} ({z_labels[i]}-{z_labels[i+1]})" for i in range(len(sorted_voxels) - 1)]
    #g.set(yticks=[0.5])
    #g.set(yticklabels=y_tick_labels)

    """
    for i, ax in enumerate(g.axes.flat):
        tick_label = f"V{sorted_voxels[i]} ({z_labels[i]}–{z_labels[i+1]})"
        ax.set_yticks([0.5])
        ax.set_yticklabels([tick_label], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='y', labelsize=9)
    """
    for i, ax in enumerate(g.axes.flat):
        tick_label = f"V{sorted_voxels[i]} ({z_labels[i]}–{z_labels[i+1]})"
        ax.text(-0.1, 0.5, tick_label, transform=ax.transAxes,
                ha='right', va='center', fontsize=12, family='monospace')


    #
        #ax.invert_yaxis()

    # Set full y-axis tick labels (spine level, leftmost)
    g.fig.subplots_adjust(left=0.15)

    # Custom global y-axis (only if you want full scale ticks, optional)
    # ax = g.fig.axes[0]
    # ax.set_yticks(z_ticks)
    # ax.set_yticklabels(z_labels)


    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.1)

    g.set(xlim=(min_q05, max_q95))
    g.set_titles("")
    #g.set(yticks=[])
    g.despine(left=True)
    g.set_axis_labels(x_label, "", fontsize=14)
    g.fig.text(0, 0.5, y_label, va='center', rotation='vertical', fontsize=14)

    if coloring_enabled:
        g.fig.text(0.87, 0.5, 'Tumor tissue score', va='center', rotation='vertical', fontsize=10)
        cbar_ax = g.fig.add_axes([0.88, 0.2, 0.015, 0.6])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

    # Layout
    g.fig.subplots_adjust(left=0.07, right=0.85 if coloring_enabled else 0.93, top=0.9, bottom=0.05)

    # Legend
    legend_lines = [
        Line2D([0], [0], color='magenta', lw=1, label='Max Density'),
        Line2D([0], [0], color='orange', lw=1, label='Mean Dose'),
        Line2D([0], [0], color='red', lw=1, label='Nominal Dose'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Quantiles (5%, 25%, 50%, 75%, 95%)')
    ]
    g.fig.legend(
        handles=legend_lines,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor='white',
        fontsize=12
    )

    # Title and patient box
    g.fig.text(0.07, 0.93, f"Patient ID: {patient_id} | Bx ID: {bx_id} | Trials: {num_trials}",
               ha='left', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.suptitle(f"{fig_title_suffix}", fontsize=16, fontweight='bold', y=0.98, x=0.1)

    # Size
    base_height_per_ridge = 1.0
    base_width = 10.0
    fig_h = base_height_per_ridge * len(voxel_ids) * fig_scale
    fig_w = base_width * fig_scale
    g.fig.set_size_inches(fig_w, fig_h)

    # Save
    save_path = os.path.join(save_dir, f"{patient_id}_{bx_id}_ridge_plot_{fig_name_suffix}.svg")
    g.fig.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')

    png_path = save_path.replace(".svg", ".png")
    g.fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')

    plt.close(g.fig)




def plot_dose_ridge_cohort_by_voxel(
    dose_df,
    save_dir,
    fig_title_suffix,
    fig_name_suffix,
    fig_scale=1.0,
    dpi=300,
    add_text_annotations=True,
    x_label="Dose (Gy)",
    y_label="Axial Dimension (mm)",
    space_between_ridgeline_padding_multiplier=1.2,
    ridgeline_vertical_padding_value=0.25
):

    plt.ioff()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    voxel_ids = dose_df['Voxel index'].unique()
    palette = {v: 'black' for v in voxel_ids}

    def annotate_and_fill(x, color, label, **kwargs):
        label_val = float(label)
        voxel_data = dose_df[dose_df['Voxel index'] == label_val]['Dose (Gy)']

        kde = gaussian_kde(voxel_data)
        x_grid = np.linspace(voxel_data.min(), voxel_data.max(), 1000)
        y_vals = kde(x_grid)
        y_scaled = y_vals / np.max(y_vals) if np.max(y_vals) > 0 else y_vals

        mean = voxel_data.mean()
        std = voxel_data.std()
        max_dens = x_grid[np.argmax(y_vals)]
        q_vals = [np.percentile(voxel_data, q) for q in [5, 25, 50, 75, 95]]

        ax = plt.gca()
        ax.fill_between(x_grid, y_scaled, alpha=0.5, color='gray')

        ax.axvline(x=max_dens, color='magenta', linestyle='-', linewidth=1)
        ax.axvline(x=mean, color='orange', linestyle='-', linewidth=1)
        for qv in q_vals:
            ax.axvline(x=qv, color='gray', linestyle='--', linewidth=1)

        if add_text_annotations:
            annotation = (
                f"Mean: {mean:.2f} Gy | SD: {std:.2f} | Max Dens: {max_dens:.2f}"
            )
            ax.text(1.02, 0.5, annotation, transform=ax.transAxes, ha='left', va='center', fontsize=8, color=color)

        ax.set_yticks([0.5])
        ax.set_yticklabels([""])
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', which='both', labelbottom=True, length=3, width=0.8)

    g = sns.FacetGrid(dose_df, row='Voxel index', hue='Voxel index', aspect=15,
                      height=1 * space_between_ridgeline_padding_multiplier, palette=palette)
    g.map(annotate_and_fill, 'Dose (Gy)')

    sorted_voxels = sorted(voxel_ids)
    z_ticks = []
    z_labels = []

    for i, voxel in enumerate(sorted_voxels):
        tick_label = f"Voxel {voxel}"
        z_ticks.append(i + 1)
        z_labels.append(tick_label)

    g.fig.subplots_adjust(hspace=ridgeline_vertical_padding_value)
    for i, ax in enumerate(g.axes.flat):
        tick_label = z_labels[i]
        ax.text(-0.1, 0.5, tick_label, transform=ax.transAxes,
                ha='right', va='center', fontsize=12, family='monospace')

    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.1)

    min_q05 = dose_df.groupby('Voxel index')['Dose (Gy)'].quantile(0.05).min()
    max_q95 = dose_df.groupby('Voxel index')['Dose (Gy)'].quantile(0.95).max()
    g.set(xlim=(min_q05, max_q95))
    g.set_titles("")
    g.despine(left=True)
    g.set_axis_labels(x_label, "", fontsize=14)
    g.fig.text(0, 0.5, y_label, va='center', rotation='vertical', fontsize=14)

    g.fig.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.05)

    legend_lines = [
        Line2D([0], [0], color='magenta', lw=1, label='Max Density'),
        Line2D([0], [0], color='orange', lw=1, label='Mean Dose'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Quantiles (5%, 25%, 50%, 75%, 95%)')
    ]
    g.fig.legend(
        handles=legend_lines,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor='white',
        fontsize=12
    )

    g.fig.text(0.07, 0.93, f"Cohort-Wide Voxel Dose Distribution",
               ha='left', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.suptitle(f"{fig_title_suffix}", fontsize=16, fontweight='bold', y=0.98, x=0.1)

    base_height_per_ridge = 1.0
    base_width = 10.0
    fig_h = base_height_per_ridge * len(voxel_ids) * fig_scale
    fig_w = base_width * fig_scale
    g.fig.set_size_inches(fig_w, fig_h)

    save_path = os.path.join(save_dir, f"cohort_ridge_plot_{fig_name_suffix}.svg")
    g.fig.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')

    png_path = save_path.replace(".svg", ".png")
    g.fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')

    plt.close(g.fig)



























###################### nominal - mean dose comparison plots #########################



def plot_biopsy_deltas_line(
    deltas_df: pd.DataFrame,
    patient_id,
    bx_index,
    save_dir,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',   # must match what you passed to compute_biopsy_nominal_deltas
    x_axis: str = 'Voxel index',               # or 'Voxel begin (Z)'
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    title: str | None = None,
):
    plt.ioff()

    """Line plot of Nominal−(Mean/Mode/Q50) along core for one (Patient ID, Bx index). Saves SVG."""
    # filter biopsy
    m = (deltas_df[('Patient ID','')] == patient_id) & (deltas_df[('Bx index','')] == bx_index)
    sub = deltas_df.loc[m].copy()
    if sub.empty:
        raise ValueError(f"No rows for Patient ID={patient_id!r}, Bx index={bx_index!r}")

    # choose x-axis
    if x_axis == 'Voxel begin (Z)':
        x = sub[('Voxel begin (Z)','')]
        x_label = 'Voxel begin (Z) [mm]'
    else:
        x = sub[('Voxel index','')]
        x_label = 'Voxel index (along core)'

    sub = sub.assign(_x=pd.to_numeric(x, errors='coerce')).sort_values('_x')

    # pick delta block
    block = f"{zero_level_index_str} deltas"
    cols = [(block,'nominal_minus_mean'), (block,'nominal_minus_mode'), (block,'nominal_minus_q50')]
    for c in cols:
        if c not in sub.columns:
            raise KeyError(f"Missing {c}. Did you compute deltas with zero_level_index_str='{zero_level_index_str}'?")

    tidy = sub.loc[:, cols].copy()
    tidy.columns = ['Nominal - Mean', 'Nominal - Mode', 'Nominal - Median (Q50)']
    tidy = tidy.assign(x=sub['_x'].values).melt(id_vars='x', var_name='Delta', value_name='Value')

    # y-label based on metric
    y_label = 'Delta (Gy/mm)' if 'grad' in zero_level_index_str else 'Delta (Gy)'

    sns.set(style='whitegrid')
    ax = sns.lineplot(data=tidy, x='x', y='Value', hue='Delta', linewidth=2)
    ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.legend(title='', fontsize=tick_label_fontsize)
    ax.set_title(title or f"Patient {patient_id}, Bx {bx_index} — {zero_level_index_str} deltas",
                 fontsize=axes_label_fontsize)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / f"{fig_name}.svg"
    plt.tight_layout()
    plt.savefig(out, format='svg')
    plt.close()
    return out


def plot_biopsy_deltas_line_both_signed_and_abs(
    deltas_df: pd.DataFrame,
    patient_id,
    bx_index,
    save_dir,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',   # must match compute_biopsy_nominal_deltas*
    x_axis: str = 'Voxel index',               # or 'Voxel begin (Z)'
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    title: str | None = None,
    # --- options ---
    include_abs: bool = True,
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,
    label_style: str = 'math',                 # kept for backward compat (ignored for y/legend)
    median_superscript: str = 'Q50',
    order_kinds: tuple = ('mean', 'mode', 'median'),
    show_points: bool = False,
    point_size: int = 3,
    alpha: float = 0.5,
    linewidth_signed: float = 2.0,
    linewidth_abs: float = 2.0,
    # --- NEW customization ---
    show_title: bool = True,                   # set False to remove title completely
    linestyle_signed: tuple | str = 'solid',
    linestyle_absolute: tuple | str = (0, (3, 2)),  # dotted
):




    def _as_dashpattern(val):
        """Return a Seaborn-friendly dash pattern tuple (on, off, ...)."""
        if isinstance(val, tuple):
            # If user passed a 2-tuple like (0, (3,2)), unwrap it
            if len(val) == 2 and isinstance(val[1], (tuple, list)):
                return tuple(val[1])
            return val  # assume already a pattern-only tuple like (3,2)
        if isinstance(val, str):
            v = val.lower()
            if v in ("solid", "-"):
                return ()                # solid -> empty tuple (pattern-only)
            if v in ("dashed", "--"):
                return (6, 6)
            if v in ("dotted", ":"):
                return (1, 3)
            if v in ("dashdot", "-."):
                return (6, 4, 1, 4)
            raise ValueError(f"Unrecognized linestyle string: {val!r}")
        # default to solid
        return ()




    plt.ioff()

    # --- filter the biopsy (MultiIndex metadata expected)
    mask = (deltas_df[('Patient ID','')] == patient_id) & (deltas_df[('Bx index','')] == bx_index)
    sub = deltas_df.loc[mask].copy()
    if sub.empty:
        raise ValueError(f"No rows for Patient ID={patient_id!r}, Bx index={bx_index!r}")

    # --- choose x-axis
    if x_axis == 'Voxel begin (Z)':
        x = sub[('Voxel begin (Z)', '')]
        x_label = 'Voxel begin (Z) [mm]'
    else:
        x = sub[('Voxel index', '')]
        x_label = 'Voxel index'   # <- final label
    sub = sub.assign(_x=pd.to_numeric(x, errors='coerce')).sort_values('_x')


    sub = sub.assign(_x=pd.to_numeric(x, errors='coerce')).sort_values('_x')

    # --- mathy helpers
    def _latex_j(kind: str) -> str:
        if kind == 'mean':
            sup = r'\mathrm{mean}'
        elif kind == 'mode':
            sup = r'\mathrm{mode}'
        elif kind == 'median':
            sup = r'\mathrm{' + (median_superscript.replace('%', r'\%')) + r'}'
        else:
            sup = r'\mathrm{' + kind + r'}'
        return rf'$\Delta_{{b,v}}^{{{sup}}}$'

    # --- signed delta columns
    block = f"{zero_level_index_str} deltas"
    signed_map = {
        'mean':   (block, 'nominal_minus_mean'),
        'mode':   (block, 'nominal_minus_mode'),
        'median': (block, 'nominal_minus_q50'),
    }
    missing_signed = [pair for pair in signed_map.values() if pair not in sub.columns]
    if missing_signed:
        raise KeyError(
            f"Missing signed delta columns for zero_level_index_str='{zero_level_index_str}'. "
            f"Missing: {missing_signed}"
        )

    # tidy signed
    signed_cols = [signed_map[k] for k in order_kinds]
    tidy_signed = sub.loc[:, signed_cols].copy()
    # attach a canonical j column ('mean','mode','median') and a display label column
    tidy_signed.columns = order_kinds
    tidy_signed = tidy_signed.assign(x=sub['_x'].values).melt(id_vars='x', var_name='j', value_name='Value')
    tidy_signed['Kind'] = 'Signed'
    tidy_signed['j_label'] = tidy_signed['j'].map(_latex_j)

    # --- absolute deltas
    tidy_abs = None
    if include_abs:
        abs_block = f"{zero_level_index_str} abs deltas"
        abs_map = {
            'mean':   (abs_block, 'abs_nominal_minus_mean'),
            'mode':   (abs_block, 'abs_nominal_minus_mode'),
            'median': (abs_block, 'abs_nominal_minus_q50'),
        }
        has_abs = all(pair in sub.columns for pair in abs_map.values())
        if not has_abs and require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError(
                "Absolute delta columns are missing and recomputation is disabled. "
                f"Expected abs columns: {list(abs_map.values())}. "
                "Use *_with_abs DataFrames or set fallback_recompute_abs=True."
            )

        if has_abs:
            abs_cols = [abs_map[k] for k in order_kinds]
            tidy_abs = sub.loc[:, abs_cols].copy()
            tidy_abs.columns = order_kinds
            tidy_abs = tidy_abs.assign(x=sub['_x'].values).melt(id_vars='x', var_name='j', value_name='Value')
            tidy_abs['Kind'] = 'Absolute'
            tidy_abs['j_label'] = tidy_abs['j'].map(_latex_j)
        elif fallback_recompute_abs:
            tidy_abs = tidy_signed.copy()
            tidy_abs['Value'] = tidy_abs['Value'].abs()
            tidy_abs['Kind'] = 'Absolute'
            tidy_abs['j_label'] = tidy_abs['j'].map(_latex_j)


    # --- compose plotting frame
    tidy_plot = pd.concat([tidy_signed, tidy_abs], ignore_index=True) if (include_abs and tidy_abs is not None) else tidy_signed

    # --- palette: 3 stable colors for the three j's in the requested order
    j_order = [k for k in order_kinds]  # e.g. ('mean','mode','median')
    # Convert 'median' to the display-order label \Delta^{Q50} in legend; palette keyed by j (not label)
    base_palette = sns.color_palette("tab10", n_colors=len(j_order))
    palette = {j_order[i]: base_palette[i] for i in range(len(j_order))}


    # --- units + axis labels
    is_gradient = ('grad' in zero_level_index_str.lower()) or ('gradient' in zero_level_index_str.lower())
    unit_text = r'Gy mm$^{-1}$' if is_gradient else r'Gy'
    delta_type_text = r'G' if is_gradient else r'D' 
    y_label = rf'$\Delta {delta_type_text}_{{b,v}}^{{j}}$ ({unit_text})'




    # --- plot (color by j, linestyle by Signed/Absolute)
    sns.set(style='whitegrid')

    dash_signed   = _as_dashpattern(linestyle_signed)
    dash_absolute = _as_dashpattern(linestyle_absolute)
    dashes_map = {"Signed": dash_signed, "Absolute": dash_absolute} if (include_abs and tidy_abs is not None) else {"Signed": dash_signed}

    ax = sns.lineplot(
        data=tidy_plot,
        x='x', y='Value',
        hue='j',
        hue_order=j_order,
        palette=palette,
        style='Kind',
        dashes=dashes_map,
        linewidth=linewidth_signed,   # base width for signed; we'll bump absolute next
        legend=False
    )

    # Make absolute lines thicker (robust to ordering)
    groups = list(tidy_plot.groupby(['j','Kind'], sort=False))
    for line, ((jv, kv), _df) in zip(ax.lines, groups):
        if kv == 'Absolute':
            line.set_linewidth(linewidth_abs)



    # optional points (same hue & style)
    if show_points:
        sns.scatterplot(
            data=tidy_plot,
            x='x', y='Value',
            hue='j',
            hue_order=j_order,
            palette=palette,
            style='Kind',
            alpha=alpha, s=point_size**2,
            legend=False
        )

    # labels, title
    ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    if show_title and title:
        ax.set_title(title, fontsize=axes_label_fontsize)
    elif show_title and not title:
        ax.set_title(f"Patient {patient_id}, Bx {bx_index} — {zero_level_index_str} deltas", fontsize=axes_label_fontsize)
    # else: no title at all

    # --- ONE unified legend ---
    from matplotlib.lines import Line2D

    # 1) Δ kinds (colors)
    j_labels_display = [_latex_j(j) for j in j_order]
    color_handles = [Line2D([0],[0], color=palette[j], lw=linewidth_signed, linestyle='solid', label=lab)
                    for j, lab in zip(j_order, j_labels_display)]

    # 2) Signed vs |Absolute| (linestyles)
    def _legend_ls_from_pattern(pat): return ('solid' if pat == () else (0, pat))
    style_handles = [
        Line2D([0],[0], color='black', lw=linewidth_signed, linestyle=_legend_ls_from_pattern(dash_signed),   label=r'$\Delta$'),
        Line2D([0],[0], color='black', lw=linewidth_abs,   linestyle=_legend_ls_from_pattern(dash_absolute), label=r'$|\Delta|$'),
    ]

    # stitch into a single legend
    handles = color_handles + style_handles
    leg = ax.legend(handles=handles, title=None, frameon=True, fontsize=tick_label_fontsize, loc='best')
    for t in leg.get_texts():
        t.set_fontsize(tick_label_fontsize)


    # save both SVG + PNG
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, format='png')
    plt.close()
    return svg_path, png_path






def plot_biopsy_deltas_line_multi(
    deltas_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],     # [(patient_id, bx_index), ...] (same patient or across)
    save_dir: str | Path,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    x_axis: str = 'Voxel index',
    axes_label_fontsize: int = 16,
    tick_label_fontsize: int = 14,
    title: str | None = None,
    include_abs: bool = True,
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,
    median_superscript: str = 'Q50',
    order_kinds: tuple = ('mean','mode','median'),
    linewidth_signed: float = 2.0,
    linewidth_abs: float = 2.6,              # thicker dotted by default
    linestyle_signed: tuple | str = 'solid',
    linestyle_absolute: tuple | str = (0, (2, 2)),
    show_markers: bool = False,
    marker_size: int = 36,                   # matplotlib points^2
    marker_edgewidth: float = 1.0,
    marker_every: int | None = None,         # e.g., every 2 points
    palette: str | Iterable = 'tab10',
    y_tick_decimals: int | None = 1,
    legend_fontsize: int | None = None,   # default: use tick_label_fontsize

):
    """
    Overlay multiple biopsies on one plot:
      hue = Δ kind (colors)
      style = Signed vs |Absolute| (linestyle)
      markers = Biopsy (optional)
    """
    import seaborn as sns
    from matplotlib.lines import Line2D
    from pathlib import Path

    sns.set(style='whitegrid')

    is_gradient = ('grad' in zero_level_index_str.lower())
    delta_type_text = r'G' if is_gradient else r'D'

    # helpers
    def _mi(name: str):
        return (name, '') if isinstance(deltas_df.columns, pd.MultiIndex) and (name, '') in deltas_df.columns else name

    def _latex_j(kind: str) -> str:
        if kind == 'mean':      sup = r'\mathrm{mean}'
        elif kind == 'mode':    sup = r'\mathrm{mode}'
        elif kind == 'median':  sup = r'\mathrm{' + (median_superscript.replace('%', r'\%')) + r'}'
        else:                   sup = r'\mathrm{' + kind + r'}'
        # include D/G just like the y-axis
        return rf'$\Delta {delta_type_text}_{{b,v}}^{{{sup}}}$'

    def _as_dashpattern(val):
        if isinstance(val, tuple):
            if len(val) == 2 and isinstance(val[1], (tuple, list)):
                return tuple(val[1])
            return val
        if isinstance(val, str):
            v = val.lower()
            if v in ("solid", "-"):  return ()
            if v in ("dashed","--"): return (6,6)
            if v in ("dotted",":"):  return (1,3)
            if v in ("dashdot","-."):return (6,4,1,4)
            raise ValueError(f"bad linestyle {val!r}")
        return ()

    pid_c = _mi('Patient ID'); bxi_c = _mi('Bx index')
    x_c   = _mi(x_axis if x_axis=='Voxel begin (Z)' else 'Voxel index')
    if x_axis == 'Voxel begin (Z)':
        x_label = 'Voxel begin (Z) [mm]'
    else:
        x_label = 'Voxel index'

    # collect tidy rows for all biopsies
    frames = []
    for pid, bx in biopsies:
        sub = deltas_df[(deltas_df[pid_c]==pid) & (deltas_df[bxi_c]==bx)].copy()
        if sub.empty:
            continue
        sub = sub.assign(_x=pd.to_numeric(sub[x_c], errors='coerce')).sort_values('_x')

        block = f"{zero_level_index_str} deltas"
        signed_map = {
            'mean':   (block, 'nominal_minus_mean'),
            'mode':   (block, 'nominal_minus_mode'),
            'median': (block, 'nominal_minus_q50'),
        }
        if not all(pair in sub.columns for pair in signed_map.values()):
            raise KeyError(f"Missing signed delta cols for {zero_level_index_str}")

        tidy_s = sub.loc[:, [signed_map[k] for k in order_kinds]].copy()
        tidy_s.columns = order_kinds
        tidy_s = tidy_s.assign(x=sub['_x'].values, Biopsy=f"{pid}, Bx {bx}", Kind='Signed')
        tidy_s = tidy_s.melt(id_vars=['x','Biopsy','Kind'], var_name='j', value_name='Value')

        tidy_a = None
        if include_abs:
            abs_block = f"{zero_level_index_str} abs deltas"
            abs_map = {
                'mean':   (abs_block, 'abs_nominal_minus_mean'),
                'mode':   (abs_block, 'abs_nominal_minus_mode'),
                'median': (abs_block, 'abs_nominal_minus_q50'),
            }
            has_abs = all(pair in sub.columns for pair in abs_map.values())
            if has_abs:
                tidy_a = sub.loc[:, [abs_map[k] for k in order_kinds]].copy()
                tidy_a.columns = order_kinds
                tidy_a = tidy_a.assign(x=sub['_x'].values, Biopsy=f"{pid}, Bx {bx}", Kind='Absolute')
                tidy_a = tidy_a.melt(id_vars=['x','Biopsy','Kind'], var_name='j', value_name='Value')
            elif require_precomputed_abs and not fallback_recompute_abs:
                raise KeyError("Abs delta block missing and recompute disabled.")
            else:
                tidy_a = tidy_s.copy()
                tidy_a['Value'] = tidy_a['Value'].abs()
                tidy_a['Kind'] = 'Absolute'

        frames.append(tidy_s)
        if tidy_a is not None:
            frames.append(tidy_a)

    if not frames:
        raise ValueError("No data for requested biopsies.")
    tidy = pd.concat(frames, ignore_index=True)
    tidy['j_label'] = tidy['j'].map(_latex_j)

    # palettes/styles
    j_order = list(order_kinds)
    if isinstance(palette, str):
        cols = sns.color_palette(palette, n_colors=len(j_order))
    else:
        cols = list(palette)
    color_map = {j_order[i]: cols[i] for i in range(len(j_order))}
    dashes_map = {'Signed': _as_dashpattern(linestyle_signed),
                  'Absolute': _as_dashpattern(linestyle_absolute)}

    # ---- split the tidy data for controlled layering ----
    signed_df = tidy[tidy['Kind'] == 'Signed']
    abs_df    = tidy[tidy['Kind'] == 'Absolute'] if include_abs else tidy.iloc[0:0]

    dash_signed   = _as_dashpattern(linestyle_signed)      # usually ()
    dash_absolute = _as_dashpattern(linestyle_absolute)    # e.g. (1,3)
    # (optional) make dotted more visible than (1,3)
    if dash_absolute == (1, 3):
        dash_absolute = (4, 3)

    # compute final linestyles once
    ls_signed   = 'solid' if dash_signed == () else (0, dash_signed)
    ls_absolute = 'solid' if dash_absolute == () else (0, dash_absolute)

    # 1) Signed first (under)
    ax = sns.lineplot(
        data=signed_df, x='x', y='Value',
        hue='j', hue_order=j_order, palette=color_map,
        units='Biopsy', estimator=None, errorbar=None, sort=False,
        linestyle=ls_signed,                 # <-- use linestyle, not dashes
        linewidth=linewidth_signed, legend=False,
        zorder=2
    )

    # 2) Absolute second (on top)
    sns.lineplot(
        data=abs_df, x='x', y='Value',
        hue='j', hue_order=j_order, palette=color_map,
        units='Biopsy', estimator=None, errorbar=None, sort=False,
        linestyle=ls_absolute,               # <-- use linestyle, not dashes
        linewidth=linewidth_abs, legend=False,
        zorder=3, ax=ax
    )

    # --- enforce integer x ticks spanning min..max ---
    # collect x from whichever data exists
    if include_abs and not abs_df.empty:
        x_vals = np.concatenate([signed_df['x'].values, abs_df['x'].values])
    else:
        x_vals = signed_df['x'].values

    # guard against NaNs
    x_vals = np.asarray(x_vals)
    x_vals = x_vals[~np.isnan(x_vals)]
    if x_vals.size:
        x_min = int(np.floor(x_vals.min()))
        x_max = int(np.ceil(x_vals.max()))

        # prefer step = 1, but cap number of ticks to ~25 if needed
        max_ticks = 25
        span = max(1, x_max - x_min)
        step = max(1, int(np.ceil(span / max_ticks)))

        ax.set_xlim(x_min, x_max)  # hard-limit to min..max
        ticks = np.arange(x_min, x_max + 1, step, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:d}" for t in ticks])



    # Round caps so thicker dotted lines look properly thicker
    for ln in ax.lines:
        ln.set_solid_capstyle('round')
        ln.set_dash_capstyle('round')


    # optional biopsy markers
    if show_markers:
        # give each biopsy a distinct marker
        marker_cycle = ['o','s','^','D','P','X','v','>','<']
        biopsy_to_marker = {b: marker_cycle[i % len(marker_cycle)] for i, b in enumerate(tidy['Biopsy'].unique())}
        # plot markers as a second pass so they sit on top
        for (jv, kv, bv), g in tidy.groupby(['j','Kind','Biopsy']):
            m = biopsy_to_marker[bv]
            sc = ax.scatter(g['x'], g['Value'], marker=m, s=marker_size,
                            facecolors='none' if kv=='Absolute' else color_map[jv],
                            edgecolors=color_map[jv], linewidths=marker_edgewidth, zorder=3)
            if marker_every is not None:
                # thin markers along the line
                sc.set_offsets(sc.get_offsets()[::max(1, int(marker_every))])

    # labels
    ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
    is_gradient = ('grad' in zero_level_index_str.lower())
    unit_text = r'Gy mm$^{-1}$' if is_gradient else r'Gy'
    if include_abs:
        ax.set_ylabel(rf'$\Delta {delta_type_text}^{{j}}_{{b,v}}$ and $|\Delta {delta_type_text}^{{j}}_{{b,v}}|$ ({unit_text})', fontsize=axes_label_fontsize)
    else:
        ax.set_ylabel(rf'$\Delta {delta_type_text}^{{j}}_{{b,v}}$ ({unit_text})', fontsize=axes_label_fontsize)

    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    if title:
        ax.set_title(title, fontsize=axes_label_fontsize)

    # unified legend (Δ kinds + Signed/|Abs| + Biopsies when markers shown)
    from matplotlib.lines import Line2D

    # --- unified legend ---
    handles = []
    handles += [Line2D([0],[0], color=color_map[j], lw=linewidth_signed, label=_latex_j(j))
                for j in j_order]

    # use ls_signed / ls_absolute computed earlier
    handles.append(
        Line2D([0],[0], color='black', lw=linewidth_signed, linestyle=ls_signed,
            label=rf'$\Delta {delta_type_text}$')
    )
    if include_abs:
        handles.append(
            Line2D([0],[0], color='black', lw=linewidth_abs, linestyle=ls_absolute,
                label=rf'$|\Delta {delta_type_text}|$')
        )



    # biopsies (markers) — only if markers are on
    if show_markers:
        marker_cycle = ['o','s','^','D','P','X','v','>','<']  # ensure defined here too
        for i, b in enumerate(tidy['Biopsy'].unique()):
            m = marker_cycle[i % len(marker_cycle)]
            handles.append(Line2D([0],[0], marker=m, color='black', linestyle='None', label=b,
                                markerfacecolor='none', markeredgewidth=1.2))

    fs = legend_fontsize if legend_fontsize is not None else tick_label_fontsize
    leg = ax.legend(handles=handles, frameon=True, fontsize=fs, loc='best')



    if y_tick_decimals is not None:
        fmt = f'{{x:.{y_tick_decimals}f}}'
        ax.yaxis.set_major_formatter(StrMethodFormatter(fmt))
        ax.get_yaxis().get_offset_text().set_visible(False)  # optional

    # save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, format='png')
    plt.close()
    return svg_path, png_path













def plot_biopsy_voxel_trial_boxplots_dual(
    deltas_df: pd.DataFrame,
    patient_id: str,
    bx_index: int,
    output_dir: str | Path,
    plot_name_base: str,
    *,
    metric: Literal['Dose (Gy)', 'Dose grad (Gy/mm)'] = 'Dose (Gy)',
    x_axis: Literal['Voxel index', 'Voxel begin (Z)'] = 'Voxel index',
    # layout & style
    figsize: tuple[float, float] = (11.0, 5.0),
    dpi: int = 200,
    axes_label_fontsize: int = 14,
    tick_label_fontsize: int = 11,
    show_title: bool = True,
    title: Optional[str] = None,
    tight_layout: bool = True,
    y0_refline: bool = True,          # helpful for signed deltas
    ylim: Optional[tuple[float, float]] = None,
    # boxplot controls
    whis: float | tuple[float, float] = 1.5,    # e.g., (5,95) for percentile whiskers
    showfliers: bool = True,
    box_offset: float = 0.18,                   # signed at x-δ, abs at x+δ
    # voxel axis density
    sort_voxels_by: Optional[Literal['median', 'mean']] = 'median',
    max_voxels: Optional[int] = None,
    xtick_stride: Optional[int] = None,         # show every k-th label; auto if None
    # trial-point overlays
    show_points_signed: bool = True,
    show_points_abs: bool = True,
    point_size_signed: float = 7.0,
    point_size_abs: float = 7.0,
    point_alpha_signed: float = 0.25,
    point_alpha_abs: float = 0.25,
    jitter_width: float = 0.25,
    # abs handling
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,
    # labeling mode
    label_style: Literal['math', 'text'] = 'math',
    legend_loc: str = 'best',
    # saving
    save_formats: Iterable[str] = ('png', 'pdf'),
) -> list[Path]:
    """
    Per-voxel dual boxplot: for each voxel on the x-axis, draw TWO boxes:
      - left = signed Δ distribution across trials,
      - right = |Δ| distribution across trials.
    Call once for 'Dose (Gy)' and once for 'Dose grad (Gy/mm)'.
    """
    # ---- helpers ----
    def _mi(name: str):
        return (name, '') if isinstance(deltas_df.columns, pd.MultiIndex) and (name, '') in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _resolve_trial_col(metric: str, use_abs: bool):
        if use_abs:
            mi_key = (f"{metric} abs deltas", "abs_nominal_minus_trial")
            flat = f"{metric} abs deltas_abs_nominal_minus_trial"
        else:
            mi_key = (f"{metric} deltas", "nominal_minus_trial")
            flat = f"{metric} deltas_nominal_minus_trial"
        if _has(mi_key): return mi_key
        if _has(flat):   return flat
        if use_abs and (not require_precomputed_abs) and fallback_recompute_abs:
            return ('__compute_abs_from__', _resolve_trial_col(metric, use_abs=False))
        raise KeyError(f"Missing column for metric={metric!r}, {'abs' if use_abs else 'signed'}.")

    # ---- filter ----
    pid_c = _mi('Patient ID')
    bxi_c = _mi('Bx index')
    x_c   = _mi(x_axis)
    sub = deltas_df[(deltas_df[pid_c] == patient_id) & (deltas_df[bxi_c] == bx_index)].copy()
    if sub.empty:
        raise ValueError(f"No data for patient={patient_id!r}, biopsy={bx_index}.")
    if x_c not in sub.columns:
        raise KeyError(f"X-axis column {x_axis!r} not found.")

    # resolve columns
    signed_key = _resolve_trial_col(metric, use_abs=False)
    abs_key    = _resolve_trial_col(metric, use_abs=True)
    compute_abs_from = None
    if isinstance(abs_key, tuple) and abs_key[0] == '__compute_abs_from__':
        compute_abs_from = abs_key[1]

    # order by x for nicer axis
    sub = sub.sort_values(by=x_c).reset_index(drop=True)

    # group per voxel
    vox_labels = []
    signed_groups, abs_groups = [], []
    for vox, g in sub.groupby(x_c, sort=False):
        s_vals = pd.to_numeric(g[signed_key], errors='coerce').dropna().values
        if compute_abs_from is not None:
            a_vals = pd.to_numeric(g[compute_abs_from], errors='coerce').abs().dropna().values
        else:
            a_vals = pd.to_numeric(g[abs_key], errors='coerce').dropna().values
        if s_vals.size or a_vals.size:
            vox_labels.append(vox)
            signed_groups.append(s_vals)
            abs_groups.append(a_vals)

    if not signed_groups and not abs_groups:
        raise ValueError("No usable trial values found.")

    # optional voxel ordering
    if sort_voxels_by in {'median', 'mean'}:
        stat_fn = np.median if sort_voxels_by == 'median' else np.mean
        # Use signed to sort; if empty, fall back to abs
        base_stats = np.array([stat_fn(g) if g.size else (stat_fn(h) if h.size else 0.0)
                               for g, h in zip(signed_groups, abs_groups)])
        order = np.argsort(base_stats)
        vox_labels   = [vox_labels[i]   for i in order]
        signed_groups= [signed_groups[i] for i in order]
        abs_groups   = [abs_groups[i]    for i in order]

    # cap voxel count
    if max_voxels is not None and len(vox_labels) > max_voxels:
        vox_labels    = vox_labels[:max_voxels]
        signed_groups = signed_groups[:max_voxels]
        abs_groups    = abs_groups[:max_voxels]

    n = len(vox_labels)
    centers = np.arange(1, n + 1)
    pos_signed = centers - box_offset
    pos_abs    = centers + box_offset

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bp_signed = ax.boxplot(
        signed_groups, positions=pos_signed, widths=0.6,
        manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=False
    )
    bp_abs = ax.boxplot(
        abs_groups, positions=pos_abs, widths=0.6,
        manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=False
    )

    # overlay jittered points
    rng = np.random.default_rng(12345)  # stable jitter across runs
    if show_points_signed:
        for i, arr in enumerate(signed_groups):
            if arr.size:
                xj = pos_signed[i] + (rng.random(arr.size) - 0.5) * jitter_width
                ax.scatter(xj, arr, s=point_size_signed, alpha=point_alpha_signed, marker='o')
    if show_points_abs:
        for i, arr in enumerate(abs_groups):
            if arr.size:
                xj = pos_abs[i] + (rng.random(arr.size) - 0.5) * jitter_width
                ax.scatter(xj, arr, s=point_size_abs, alpha=point_alpha_abs, marker='x')

    # labels
    is_grad = 'grad' in metric.lower()
    if label_style == 'math':
        core_signed = r"$\Delta^G_{b,v,t}$" if is_grad else r"$\Delta_{b,v,t}$"
        core_abs    = r"$|\Delta^G_{b,v,t}|$" if is_grad else r"$|\Delta_{b,v,t}|$"
        unit = r"$(\mathrm{Gy/mm})$" if is_grad else r"$(\mathrm{Gy})$"
        ylab = f"{core_signed} / {core_abs} {unit}"
    else:
        core_signed = "Δ^G" if is_grad else "Δ"
        core_abs    = "|Δ^G|" if is_grad else "|Δ|"
        unit = "(Gy/mm)" if is_grad else "(Gy)"
        ylab = f"{core_signed} / {core_abs} {unit}"

    ax.set_xlabel(x_axis, fontsize=axes_label_fontsize)
    ax.set_ylabel(ylab, fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    # x tick thinning
    if xtick_stride is None:
        stride = max(1, int(np.ceil(n / 40)))
    else:
        stride = max(1, int(xtick_stride))
    show_mask = ((centers - 1) % stride) == 0
    ax.set_xticks(centers[show_mask])
    ax.set_xticklabels([str(v) for v in np.array(vox_labels)[show_mask]], rotation=90)

    if y0_refline:
        ax.axhline(0.0, linestyle=':', linewidth=1.0, alpha=0.8)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_title:
        if title is None:
            title = f"{metric} — Δ & |Δ| per voxel — Patient {patient_id}, Bx {bx_index}"
        ax.set_title(title, fontsize=axes_label_fontsize)

    # legend (proxy artists)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], linestyle='-', marker='o', alpha=0.8, label='Δ trials',  linewidth=1.0),
        Line2D([], [], linestyle='-', marker='x', alpha=0.8, label='|Δ| trials', linewidth=1.0),
    ]
    ax.legend(handles=handles, loc=legend_loc, fontsize=max(axes_label_fontsize - 2, 8))

    if tight_layout:
        plt.tight_layout()

    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for ext in save_formats:
        fn = f"{plot_name_base}.{ext.lstrip('.')}"
        outp = output_dir / fn
        fig.savefig(outp, bbox_inches='tight', dpi=dpi)
        out_paths.append(outp)

    plt.close(fig)
    return out_paths









def plot_dual_boxplots_by_voxel_for_biopsies(
        deltas_df: pd.DataFrame,
        biopsies: Sequence[Tuple[str, int]],         # list of (patient_id, bx_index)
        output_dir: str | Path,
        plot_name_base: str,
        *,
        metric: Literal['Dose (Gy)', 'Dose grad (Gy/mm)'] = 'Dose (Gy)',
        x_axis: Literal['Voxel index', 'Voxel begin (Z)'] = 'Voxel index',
        # appearance & layout
        figsize: tuple[float, float] = (12.0, 5.0),
        dpi: int = 200,
        seaborn_style: str = 'whitegrid',
        seaborn_context: str = 'talk',               # 'paper'|'notebook'|'talk'|'poster'
        palette: str | Iterable = 'deep',            # seaborn palette name or list of colors
        axes_label_fontsize: int = 14,
        tick_label_fontsize: int = 11,
        show_title: bool = False,
        title: Optional[str] = None,
        tight_layout: bool = True,
        y0_refline: bool = True,                     # useful for signed deltas
        ylim: Optional[tuple[float, float]] = None,
        # box placement & style
        voxel_box_width: float = 0.55,               # width for each (biopsy) box
        delta_pair_offset: float = 0.28,             # center shift: signed at x-δ, abs at x+δ
        biopsy_spread: float = 0.20,                 # spread biopsies within each delta position
        whisker_mode: Literal['iqr1.5','q05q95'] = 'iqr1.5',
        showfliers: bool = True,
        signed_fill_alpha: float = 0.35,             # filled boxes for signed
        abs_edge_only: bool = True,                  # abs as edge-only boxes (no facecolor)
        # points overlay
        show_points: bool = False,                   # default OFF per your request
        point_size: float = 8.0,
        point_alpha: float = 0.25,
        jitter_width: float = 0.20,
        # abs handling
        require_precomputed_abs: bool = True,
        fallback_recompute_abs: bool = False,
        # labeling
        label_style: Literal['math','text'] = 'math',
        biopsy_label_map: Optional[Dict[Tuple[str,int], str]] = None,  # custom legend labels
        legend_loc: str = 'best',
        # saving
        save_formats: Iterable[str] = ('png','svg'),
        layout: str = "overlay",           # keep old behavior by default
        draw_voxel_guides: bool = True,
        voxel_guide_alpha: float = 0.06,

    ) -> list[Path]:
    """
    One figure for a single metric (Dose or Dose grad) comparing multiple biopsies.
    At each voxel x-position, draws TWO groups of boxes:
      - signed Δ (left) for all biopsies (colored)
      - absolute |Δ| (right) for all biopsies (colored)
    """

    # ---------------- seaborn look ----------------
    sns.set_theme(style=seaborn_style, context=seaborn_context)
    colors = sns.color_palette(palette, n_colors=max(1, len(biopsies)))

    # ---------------- helpers ----------------
    def _mi(name: str):
        return (name, '') if isinstance(deltas_df.columns, pd.MultiIndex) and (name, '') in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _trial_col(metric: str, use_abs: bool):
        if use_abs:
            mi_key = (f"{metric} abs deltas", "abs_nominal_minus_trial")
            flat  = f"{metric} abs deltas_abs_nominal_minus_trial"
        else:
            mi_key = (f"{metric} deltas", "nominal_minus_trial")
            flat  = f"{metric} deltas_nominal_minus_trial"
        if _has(mi_key): return mi_key, None
        if _has(flat):   return flat, None
        if use_abs and (not require_precomputed_abs) and fallback_recompute_abs:
            signed_key, _ = _trial_col(metric, use_abs=False)
            return ('__compute_abs_from__', signed_key), signed_key
        raise KeyError(f"Missing column for metric={metric!r}, {'abs' if use_abs else 'signed'}.")

    def _ylabel():
        # Proper mathtext units: Gy mm^{-1} for gradient
        if 'grad' in metric.lower():
            unit = r"$\mathrm{Gy}\ \mathrm{mm}^{-1}$"
            return r"$\Delta^G_{b,v,t}$ / $|\Delta^G_{b,v,t}|$ " + unit if label_style == 'math' else "Δ/|Δ| (Gy mm^-1)"
        else:
            unit = r"$\mathrm{Gy}$"
            return r"$\Delta_{b,v,t}$ / $|\Delta_{b,v,t}|$ " + unit if label_style == 'math' else "Δ/|Δ| (Gy)"


    # ---------------- collect data ----------------
    pid_c = _mi('Patient ID')
    bxi_c = _mi('Bx index')
    x_c   = _mi(x_axis)

    # filter once to speed up
    df = deltas_df[(deltas_df[pid_c].isin([p for p,_ in biopsies])) & (deltas_df[bxi_c].isin([b for _,b in biopsies]))].copy()
    if df.empty:
        raise ValueError("No rows for requested biopsies.")

    if x_c not in df.columns:
        raise KeyError(f"X-axis column {x_axis!r} not found in dataframe.")

    # resolve trial columns
    signed_key, _ = _trial_col(metric, use_abs=False)
    abs_key, compute_abs_from = _trial_col(metric, use_abs=True)

    # numeric, ordered x-values per-biopsy
    def _cast_numeric(s):
        return pd.to_numeric(s, errors='coerce')

    # global voxel order (sorted unique union)
    all_x = (
        pd.concat([
            _cast_numeric(
                df[(df[pid_c]==pid) & (df[bxi_c]==bx)][x_c]
            )
            for pid,bx in biopsies
        ])
        .dropna().unique()
    )
    x_vals_sorted = np.sort(all_x)

    # build groups: for each voxel and each biopsy, signed and abs arrays
    per_voxel_signed = { (pid,bx): [] for pid,bx in biopsies }
    per_voxel_abs    = { (pid,bx): [] for pid,bx in biopsies }

    for pid, bx in biopsies:
        sub = df[(df[pid_c]==pid) & (df[bxi_c]==bx)].copy()
        sub['_x'] = _cast_numeric(sub[x_c])
        sub = sub.dropna(subset=['_x'])

        # ensure order by voxel
        sub = sub.sort_values('_x')

        for xv in x_vals_sorted:
            g = sub[sub['_x'] == xv]

            s_vals = pd.to_numeric(g[signed_key], errors='coerce').dropna().values
            if isinstance(abs_key, tuple) and abs_key[0] == '__compute_abs_from__':
                a_vals = pd.to_numeric(g[compute_abs_from], errors='coerce').abs().dropna().values
            else:
                a_vals = pd.to_numeric(g[abs_key], errors='coerce').dropna().values

            per_voxel_signed[(pid,bx)].append(s_vals)  # can be empty
            per_voxel_abs[(pid,bx)].append(a_vals)     # can be empty

    # ---------------- plotting ----------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if draw_voxel_guides:
        # centers must be your integer x tick positions
        for c in centers:  # centers = np.arange(1, len(x_vals_sorted)+1)
            ax.axvspan(c-0.5, c+0.5, color='k', alpha=voxel_guide_alpha, zorder=0)


    n_vox = len(x_vals_sorted)
    centers = np.arange(1, n_vox + 1)

    # whisker style
    whis = 1.5 if whisker_mode == 'iqr1.5' else (5, 95)

    # for legend proxies
    from matplotlib.patches import Patch
    legend_patches = []

    # loop biopsies in color
    for bi, (pid,bx) in enumerate(biopsies):
        color = colors[bi % len(colors)]
        label = biopsy_label_map.get((pid,bx), f"{pid}, Bx {bx}") if biopsy_label_map else f"{pid}, Bx {bx}"

        # positions for this biopsy within each delta side
        # spread multiple biopsies within signed and within abs sides
        if len(biopsies) == 1:
            offsets_in_group = [0.0]
        else:
            # symmetric small spread around group center
            span = biopsy_spread
            offsets_in_group = np.linspace(-span, span, len(biopsies))
        inner_off = offsets_in_group[bi]

        pos_signed = centers - delta_pair_offset + inner_off
        pos_abs    = centers + delta_pair_offset + inner_off

        # signed boxes (filled with alpha)
        groups_s = per_voxel_signed[(pid,bx)]
        if any(len(a) for a in groups_s):
            bp_s = ax.boxplot(
                groups_s,
                positions=pos_signed,
                widths=voxel_box_width,
                manage_ticks=False,
                whis=whis,
                showfliers=showfliers,
                patch_artist=True
            )
            for box in bp_s['boxes']:
                box.set_facecolor(color)
                box.set_alpha(signed_fill_alpha)
                box.set_edgecolor(color)
                box.set_linewidth(1.2)
            for element in ['whiskers','caps','medians']:
                for line in bp_s[element]:
                    line.set_color(color)
                    line.set_linewidth(1.2)

            if show_points:
                rng = np.random.default_rng(12345 + bi)
                for x0, arr in zip(pos_signed, groups_s):
                    if arr.size:
                        xj = x0 + (rng.random(arr.size) - 0.5) * jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='o')

        # absolute boxes (edge only)
        groups_a = per_voxel_abs[(pid,bx)]
        if any(len(a) for a in groups_a):
            bp_a = ax.boxplot(
                groups_a,
                positions=pos_abs,
                widths=voxel_box_width,
                manage_ticks=False,
                whis=whis,
                showfliers=showfliers,
                patch_artist=True
            )
            for box in bp_a['boxes']:
                if abs_edge_only:
                    box.set_facecolor('white')
                    box.set_alpha(1.0)
                else:
                    box.set_facecolor(color)
                    box.set_alpha(0.15)
                box.set_edgecolor(color)
                box.set_linewidth(1.2)
                # distinguish abs: dashed box edges
                box.set_linestyle('--')
            for element in ['whiskers','caps','medians']:
                for line in bp_a[element]:
                    line.set_color(color)
                    line.set_linewidth(1.2)
                    line.set_linestyle('--')

            if show_points:
                rng = np.random.default_rng(54321 + bi)
                for x0, arr in zip(pos_abs, groups_a):
                    if arr.size:
                        xj = x0 + (rng.random(arr.size) - 0.5) * jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='x')

        # legend proxy for this biopsy
        legend_patches.append(Patch(facecolor=color, edgecolor=color, alpha=signed_fill_alpha, label=label))

    # axis labels, ticks, etc.
    # x ticks at centers labeled by voxel index (sorted numeric)
    ax.set_xticks(centers)
    # show integers if they are (1,2,3,...)
    # if x_axis is 'Voxel index', labels are integers; else show float
    if x_axis == 'Voxel index':
        ax.set_xticklabels([str(int(v)) for v in x_vals_sorted], rotation=0)
        xlab = 'Voxel index'
    else:
        ax.set_xticklabels([f"{v:g}" for v in x_vals_sorted], rotation=0)
        xlab = 'Voxel begin (Z) [mm]'

    # y-label based on metric
    ax.set_xlabel(xlab, fontsize=axes_label_fontsize)
    ax.set_ylabel(_ylabel(), fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    # zero reference
    if y0_refline:
        ax.axhline(0.0, linestyle=':', linewidth=1.0, alpha=0.8, color='0.3')

    if ylim is not None:
        ax.set_ylim(*ylim)

    # legend
    # add two line styles proxies for signed vs abs
    from matplotlib.lines import Line2D
    signed_proxy = Line2D([], [], color='black', linewidth=1.2, alpha=0.8, label='Δ (filled)')
    abs_proxy    = Line2D([], [], color='black', linewidth=1.2, alpha=0.8, linestyle='--', label='|Δ| (dashed)')
    leg = ax.legend(handles=legend_patches + [signed_proxy, abs_proxy], loc=legend_loc, frameon=True)
    for txt in leg.get_texts():
        txt.set_fontsize(max(tick_label_fontsize - 0, 9))

    # title
    if show_title:
        if title is None:
            title = f"{metric} — Δ & |Δ| per voxel"
        ax.set_title(title, fontsize=axes_label_fontsize)

    if tight_layout:
        plt.tight_layout()

    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for ext in save_formats:
        fp = output_dir / f"{plot_name_base}.{ext.lstrip('.')}"
        fig.savefig(fp, bbox_inches='tight', dpi=dpi)
        out_paths.append(fp)

    plt.close(fig)
    return out_paths













def plot_biopsy_voxel_dualboxes(
    deltas_df: pd.DataFrame,
    biopsies: Sequence[Tuple[str, int]],          # e.g., [("184 (F2)",1), ("184 (F2)",2)]
    output_dir: str | Path,
    plot_name_base: str,
    *,
    metric: Literal['Dose (Gy)', 'Dose grad (Gy/mm)'] = 'Dose (Gy)',
    x_axis: Literal['Voxel index', 'Voxel begin (Z)'] = 'Voxel index',

    # Layout: 'overlay' = multiple biopsies on one axes; 'facet' = one column per biopsy (no overlap)
    layout: Literal['overlay', 'facet'] = 'overlay',

    # Seaborn & figure style
    figsize: tuple[float, float] = (12.0, 5.0),
    dpi: int = 200,
    seaborn_style: str = 'whitegrid',
    seaborn_context: str = 'talk',               # 'paper'|'notebook'|'talk'|'poster'
    palette: str | Iterable = 'deep',

    # Axes text
    axes_label_fontsize: int = 14,
    tick_label_fontsize: int = 11,
    show_title: bool = False,
    title: Optional[str] = None,
    tight_layout: bool = True,
    y0_refline: bool = True,
    ylim: Optional[tuple[float, float]] = None,

    # Box geometry (overlay mode)
    signed_offset: float = 0.28,                 # signed at x - signed_offset
    abs_offset: float    = 0.28,                 # abs at    x + abs_offset
    box_width: float = 0.42,                     # width for each biopsy box in overlay
    biopsy_spread: float = 0.14,                 # tiny intra-side spread among biopsies (overlay)
    whisker_mode: Literal['iqr1.5','q05q95'] = 'q05q95',
    showfliers: bool = False,

    # Visual clarity helpers
    show_voxel_guides: bool = True,
    voxel_guide_alpha: float = 0.08,

    # Points (OFF by default)
    show_points: bool = False,
    point_size: float = 7.0,
    point_alpha: float = 0.25,
    jitter_width: float = 0.16,

    # Abs handling
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,

    # Labels / legend
    label_style: Literal['math','text'] = 'math',
    biopsy_label_map: Optional[Dict[Tuple[str,int], str]] = None,
    legend_loc: str = 'upper right',

    # Save
    save_formats: Iterable[str] = ('png','svg'),
) -> list[Path]:
    """
    Draw per-voxel boxplots that show BOTH signed Δ and absolute |Δ| on the same plot.
    - In 'overlay' mode, multiple biopsies share one axes (colored by biopsy).
    - In 'facet'  mode, each biopsy gets its own axes column (no overlap), still with Δ (left) & |Δ| (right).
    """

    # ---------- seaborn theme ----------
    sns.set_theme(style=seaborn_style, context=seaborn_context)
    colors = sns.color_palette(palette, n_colors=max(1, len(biopsies)))

    # ---------- helpers ----------
    def _mi(name: str):
        return (name, '') if isinstance(deltas_df.columns, pd.MultiIndex) and (name, '') in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _trial_col(metric: str, use_abs: bool):
        if use_abs:
            mi_key = (f"{metric} abs deltas", "abs_nominal_minus_trial")
            flat  = f"{metric} abs deltas_abs_nominal_minus_trial"
        else:
            mi_key = (f"{metric} deltas", "nominal_minus_trial")
            flat  = f"{metric} deltas_nominal_minus_trial"
        if _has(mi_key): return mi_key, None
        if _has(flat):   return flat, None
        if use_abs and (not require_precomputed_abs) and fallback_recompute_abs:
            signed_key, _ = _trial_col(metric, use_abs=False)
            return ('__compute_abs_from__', signed_key), signed_key
        raise KeyError(f"Missing col for metric={metric!r}, {'abs' if use_abs else 'signed'}.")

    def _ylabel():
        if label_style == 'math':
            # Dose grad unit in LaTeX: Gy mm^{-1}
            unit = r"$\mathrm{Gy}$" if 'grad' not in metric.lower() else r"$\mathrm{Gy}\ \mathrm{mm}^{-1}$"
            if 'grad' in metric.lower():
                return r"$\Delta^G_{b,v,t}\ /\ |\Delta^G_{b,v,t}|$" + f" {unit}"
            else:
                return r"$\Delta_{b,v,t}\ /\ |\Delta_{b,v,t}|$" + f" {unit}"
        else:
            return "Δ / |Δ| (Gy mm^-1)" if 'grad' in metric.lower() else "Δ / |Δ| (Gy)"

    # ---------- filter once ----------
    pid_c = _mi('Patient ID')
    bxi_c = _mi('Bx index')
    x_c   = _mi(x_axis)

    # keep only needed biopsies
    df = deltas_df[
        (deltas_df[pid_c].isin([p for p,_ in biopsies])) &
        (deltas_df[bxi_c].isin([b for _,b in biopsies]))
    ].copy()
    if df.empty:
        raise ValueError("No rows for requested biopsies.")
    if x_c not in df.columns:
        raise KeyError(f"X-axis column {x_axis!r} not in dataframe.")

    # resolve columns
    signed_key, _ = _trial_col(metric, use_abs=False)
    abs_key, compute_abs_from = _trial_col(metric, use_abs=True)

    # numeric x; global ordered voxel positions
    def _to_num(s): return pd.to_numeric(s, errors='coerce')
    all_xvals = (
        pd.concat([ _to_num(df[(df[pid_c]==pid) & (df[bxi_c]==bx)][x_c]) for pid,bx in biopsies ])
        .dropna().unique()
    )
    x_vals_sorted = np.sort(all_xvals)
    centers = np.arange(1, len(x_vals_sorted) + 1)

    # convenience for legend labels
    def _biopsy_label(pair):
        return biopsy_label_map.get(pair, f"{pair[0]}, Bx {pair[1]}") if biopsy_label_map else f"{pair[0]}, Bx {pair[1]}"

    # ---------- build per-voxel groups ----------
    def _groups_for_biopsy(pid, bx):
        sub = df[(df[pid_c]==pid) & (df[bxi_c]==bx)].copy()
        sub['_x'] = _to_num(sub[x_c])
        sub = sub.dropna(subset=['_x']).sort_values('_x')
        gs, ga = [], []
        for xv in x_vals_sorted:
            g = sub[sub['_x'] == xv]
            s_vals = pd.to_numeric(g[signed_key], errors='coerce').dropna().values
            if isinstance(abs_key, tuple) and abs_key[0] == '__compute_abs_from__':
                a_vals = pd.to_numeric(g[compute_abs_from], errors='coerce').abs().dropna().values
            else:
                a_vals = pd.to_numeric(g[abs_key], errors='coerce').dropna().values
            gs.append(s_vals)  # may be empty
            ga.append(a_vals)
        return gs, ga

    per_signed = {}
    per_abs    = {}
    for pair in biopsies:
        gs, ga = _groups_for_biopsy(*pair)
        per_signed[pair] = gs
        per_abs[pair]    = ga

    whis = 1.5 if whisker_mode == 'iqr1.5' else (5, 95)

    out_paths: list[Path] = []
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- overlay layout ----------
    if layout == 'overlay':
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # vertical voxel guides to make alignment obvious
        if show_voxel_guides:
            for x in centers:
                ax.axvspan(x-0.5, x+0.5, color='grey', alpha=voxel_guide_alpha, zorder=0)

        # deterministic intra-side offsets for biopsies
        if len(biopsies) == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-biopsy_spread, biopsy_spread, len(biopsies))

        # draw boxes per biopsy
        for bi, pair in enumerate(biopsies):
            color = colors[bi % len(colors)]
            label = _biopsy_label(pair)
            off   = offsets[bi]

            # signed at left
            pos_s = centers - signed_offset + off
            gs = per_signed[pair]
            if any(len(a) for a in gs):
                bp = ax.boxplot(gs, positions=pos_s, widths=box_width,
                                manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
                for box in bp['boxes']:
                    box.set_facecolor(color); box.set_alpha(0.35)
                    box.set_edgecolor(color); box.set_linewidth(1.1)
                for k in ['whiskers','caps','medians']:
                    for line in bp[k]:
                        line.set_color(color); line.set_linewidth(1.1)
                if show_points:
                    rng = np.random.default_rng(2025 + bi)
                    for x0, arr in zip(pos_s, gs):
                        if arr.size:
                            xj = x0 + (rng.random(arr.size) - 0.5)*jitter_width
                            ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='o', zorder=3)

            # abs at right (dashed edges / hollow face)
            pos_a = centers + abs_offset + off
            ga = per_abs[pair]
            if any(len(a) for a in ga):
                bp = ax.boxplot(ga, positions=pos_a, widths=box_width,
                                manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
                for box in bp['boxes']:
                    box.set_facecolor('white'); box.set_alpha(1.0)
                    box.set_edgecolor(color);  box.set_linewidth(1.1); box.set_linestyle('--')
                for k in ['whiskers','caps','medians']:
                    for line in bp[k]:
                        line.set_color(color); line.set_linewidth(1.1); line.set_linestyle('--')
                if show_points:
                    rng = np.random.default_rng(4040 + bi)
                    for x0, arr in zip(pos_a, ga):
                        if arr.size:
                            xj = x0 + (rng.random(arr.size) - 0.5)*jitter_width
                            ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='x', zorder=3)

        # ticks & labels
        ax.set_xticks(centers)
        if x_axis == 'Voxel index':
            ax.set_xticklabels([str(int(v)) for v in x_vals_sorted], rotation=0)
            xlab = 'Voxel index'
        else:
            ax.set_xticklabels([f"{v:g}" for v in x_vals_sorted], rotation=0)
            xlab = 'Voxel begin (Z) [mm]'

        ax.set_xlabel(xlab, fontsize=axes_label_fontsize)
        ax.set_ylabel(_ylabel(), fontsize=axes_label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        if y0_refline:
            ax.axhline(0.0, linestyle=':', linewidth=1.0, alpha=0.85, color='0.25')

        if ylim is not None:
            ax.set_ylim(*ylim)

        # legend: biopsy colors + line style keys
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        patches = [Patch(facecolor=colors[i % len(colors)], edgecolor=colors[i % len(colors)],
                         alpha=0.35, label=_biopsy_label(biopsies[i])) for i in range(len(biopsies))]
        signed_key = Line2D([], [], color='black', linewidth=1.1, label='Δ (left)')
        abs_key    = Line2D([], [], color='black', linewidth=1.1, linestyle='--', label='|Δ| (right)')
        leg = ax.legend(handles=patches + [signed_key, abs_key], loc=legend_loc, frameon=True)
        for t in leg.get_texts():
            t.set_fontsize(max(tick_label_fontsize-0, 9))

        if show_title:
            if title is None:
                title = f"{metric} — Δ & |Δ| per voxel"
            ax.set_title(title, fontsize=axes_label_fontsize)

        if tight_layout:
            plt.tight_layout()

        for ext in save_formats:
            fp = output_dir / f"{plot_name_base}.{ext.lstrip('.')}"
            fig.savefig(fp, bbox_inches='tight', dpi=dpi)
            out_paths.append(fp)
        plt.close(fig)

    # ---------- facet layout ----------
    else:
        # one column per biopsy; inside each, Δ on left and |Δ| on right
        n_b = len(biopsies)
        fig, axes = plt.subplots(1, n_b, figsize=(max(8.0, 6.0*n_b), figsize[1]), dpi=dpi, sharey=True)

        if n_b == 1:
            axes = [axes]

        for ax, (pair, color) in zip(axes, zip(biopsies, colors)):
            pid, bx = pair
            gs = per_signed[pair]
            ga = per_abs[pair]

            if show_voxel_guides:
                for x in centers:
                    ax.axvspan(x-0.5, x+0.5, color='grey', alpha=voxel_guide_alpha, zorder=0)

            pos_s = centers - signed_offset
            pos_a = centers + abs_offset

            bp_s = ax.boxplot(gs, positions=pos_s, widths=box_width,
                              manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
            for box in bp_s['boxes']:
                box.set_facecolor(color); box.set_alpha(0.35)
                box.set_edgecolor(color); box.set_linewidth(1.1)
            for k in ['whiskers','caps','medians']:
                for line in bp_s[k]:
                    line.set_color(color); line.set_linewidth(1.1)

            bp_a = ax.boxplot(ga, positions=pos_a, widths=box_width,
                              manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
            for box in bp_a['boxes']:
                box.set_facecolor('white'); box.set_alpha(1.0)
                box.set_edgecolor(color);  box.set_linewidth(1.1); box.set_linestyle('--')
            for k in ['whiskers','caps','medians']:
                for line in bp_a[k]:
                    line.set_color(color); line.set_linewidth(1.1); line.set_linestyle('--')

            if show_points:
                rng1 = np.random.default_rng(2025)
                rng2 = np.random.default_rng(4040)
                for x0, arr in zip(pos_s, gs):
                    if arr.size:
                        xj = x0 + (rng1.random(arr.size)-0.5)*jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='o', zorder=3)
                for x0, arr in zip(pos_a, ga):
                    if arr.size:
                        xj = x0 + (rng2.random(arr.size)-0.5)*jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='x', zorder=3)

            ax.set_xticks(centers)
            if x_axis == 'Voxel index':
                ax.set_xticklabels([str(int(v)) for v in x_vals_sorted], rotation=0)
                xlab = 'Voxel index'
            else:
                ax.set_xticklabels([f"{v:g}" for v in x_vals_sorted], rotation=0)
                xlab = 'Voxel begin (Z) [mm]'

            ax.set_xlabel(xlab, fontsize=axes_label_fontsize)
            ax.tick_params(axis='both', labelsize=tick_label_fontsize)
            ax.set_title(_biopsy_label(pair), fontsize=axes_label_fontsize)

            if y0_refline:
                ax.axhline(0.0, linestyle=':', linewidth=1.0, alpha=0.85, color='0.25')

        axes[0].set_ylabel(_ylabel(), fontsize=axes_label_fontsize)
        if ylim is not None:
            axes[0].set_ylim(*ylim)

        if show_title:
            if title is None:
                title = f"{metric} — Δ & |Δ| per voxel"
            fig.suptitle(title, fontsize=axes_label_fontsize)
            fig.subplots_adjust(top=0.88)

        if tight_layout:
            plt.tight_layout()

        for ext in save_formats:
            fp = output_dir / f"{plot_name_base}.{ext.lstrip('.')}"
            fig.savefig(fp, bbox_inches='tight', dpi=dpi)
            out_paths.append(fp)
        plt.close(fig)

    return out_paths





def plot_voxel_dualboxes_by_biopsy_lanes(
    deltas_df: pd.DataFrame,
    biopsies: Sequence[Tuple[str, int]],          # e.g. [("184 (F2)",1), ("184 (F2)",2)]
    output_dir: str | Path,
    plot_name_base: str,
    *,
    metric: Literal['Dose (Gy)', 'Dose grad (Gy/mm)'] = 'Dose (Gy)',
    x_axis: Literal['Voxel index', 'Voxel begin (Z)'] = 'Voxel index',

    # Styling / seaborn
    seaborn_style: str = 'whitegrid',
    seaborn_context: str = 'talk',
    palette: str | Iterable = 'deep',

    # Figure & axes
    figsize: tuple[float, float] = (12.0, 5.0),
    dpi: int = 200,
    axes_label_fontsize: int = 16,
    tick_label_fontsize: int = 14,
    show_title: bool = False,
    title: Optional[str] = None,
    tight_layout: bool = True,
    y0_refline: bool = True,
    ylim: Optional[tuple[float, float]] = None,

    # Lane layout (all are in "x units"; the function spaces lanes on a custom axis)
    lane_gap: float = 1.2,          # spacing between voxel lanes (increase to spread out lanes)
    box_width: float = 0.35,        # width of each box
    pair_gap: float = 0.08,         # gap between Δ and |Δ| for the SAME biopsy
    biopsy_gap: float = 0.18,       # extra gap between biopsy pairs inside a lane

    # Boxplot specifics
    whisker_mode: Literal['iqr1.5','q05q95'] = 'q05q95',
    showfliers: bool = False,

    # Trial points (OFF by default)
    show_points: bool = False,
    point_size: float = 7.0,
    point_alpha: float = 0.25,
    jitter_width: float = 0.12,

    # Abs handling
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,

    # Labels / legend
    label_style: Literal['math','text'] = 'math',
    biopsy_label_map: Optional[Dict[Tuple[str,int], str]] = None,
    legend_loc: str = 'best',
    y_label_mode: str = 'comma',   # 'and'|'comma'|'slash'


    # Save
    save_formats: Iterable[str] = ('png','svg'),
) -> list[Path]:
    """
    Per-voxel 'lane' layout:
      lane = voxel v
      inside each lane: for each biopsy in `biopsies`:
         [ Δ (solid) ] --pair_gap--> [ |Δ| (dashed) ] --biopsy_gap--> next biopsy pair
    """
    sns.set_theme(style=seaborn_style, context=seaborn_context)
    colors = sns.color_palette(palette, n_colors=max(1, len(biopsies)))

    # ----- helpers -----
    def _mi(name: str):
        return (name, '') if isinstance(deltas_df.columns, pd.MultiIndex) and (name, '') in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _trial_col(metric: str, use_abs: bool):
        if use_abs:
            mi_key = (f"{metric} abs deltas", "abs_nominal_minus_trial")
            flat  = f"{metric} abs deltas_abs_nominal_minus_trial"
        else:
            mi_key = (f"{metric} deltas", "nominal_minus_trial")
            flat  = f"{metric} deltas_nominal_minus_trial"
        if _has(mi_key): return mi_key, None
        if _has(flat):   return flat, None
        if use_abs and (not require_precomputed_abs) and fallback_recompute_abs:
            signed_key, _ = _trial_col(metric, use_abs=False)
            return ('__compute_abs_from__', signed_key), signed_key
        raise KeyError(f"Missing col for metric={metric!r}, {'abs' if use_abs else 'signed'}.")

    def _ylabel(metric: str, mode: str = 'and'):
        is_grad = 'grad' in metric.lower()
        unit = r"(Gy mm$^{-1}$)" if is_grad else r"(Gy)"
        if is_grad:
            left, right = r"$\Delta G_{b,v,t}$", r"$|\Delta G_{b,v,t}|$"
        else:
            left, right = r"$\Delta D_{b,v,t}$", r"$|\Delta D_{b,v,t}|$"
        if mode == 'and':
            return f"{left} and {right}  {unit}"
        elif mode == 'comma':
            return f"{left}, {right}  {unit}"
        else:  # fallback
            return f"{left} / {right}  {unit}"


    pid_c = _mi('Patient ID')
    bxi_c = _mi('Bx index')
    x_c   = _mi(x_axis)

    df = deltas_df[
        (deltas_df[pid_c].isin([p for p,_ in biopsies])) &
        (deltas_df[bxi_c].isin([b for _,b in biopsies]))
    ].copy()
    if df.empty:
        raise ValueError("No rows for requested biopsies.")
    if x_c not in df.columns:
        raise KeyError(f"X-axis column {x_axis!r} missing.")

    signed_key, _ = _trial_col(metric, use_abs=False)
    abs_key, compute_abs_from = _trial_col(metric, use_abs=True)

    def _to_num(s): return pd.to_numeric(s, errors='coerce')

    # global voxel order (sorted union)
    all_xvals = (
        pd.concat([ _to_num(df[(df[pid_c]==pid) & (df[bxi_c]==bx)][x_c]) for pid,bx in biopsies ])
        .dropna().unique()
    )
    x_vals_sorted = np.sort(all_xvals)
    n_vox = len(x_vals_sorted)

    # gather per-voxel per-biopsy trial arrays
    per_signed: Dict[Tuple[str,int], list] = {}
    per_abs:    Dict[Tuple[str,int], list] = {}
    for pair in biopsies:
        pid, bx = pair
        sub = df[(df[pid_c]==pid) & (df[bxi_c]==bx)].copy()
        sub['_x'] = _to_num(sub[x_c])
        sub = sub.dropna(subset=['_x']).sort_values('_x')
        gs, ga = [], []
        for xv in x_vals_sorted:
            g = sub[sub['_x'] == xv]
            s_vals = pd.to_numeric(g[signed_key], errors='coerce').dropna().values
            if isinstance(abs_key, tuple) and abs_key[0] == '__compute_abs_from__':
                a_vals = pd.to_numeric(g[compute_abs_from], errors='coerce').abs().dropna().values
            else:
                a_vals = pd.to_numeric(g[abs_key], errors='coerce').dropna().values
            gs.append(s_vals)  # can be empty
            ga.append(a_vals)  # can be empty
        per_signed[pair] = gs
        per_abs[pair]    = ga

    # lane centers spaced by lane_gap
    lane_centers = np.arange(n_vox, dtype=float) * lane_gap

    # build within-lane offsets: for each biopsy j -> two boxes (Δ then |Δ|)
    # pattern: [Δ_j], [|Δ|_j], then biopsy_gap before next biopsy’s Δ
    # offsets start from a negative value so the whole lane is centered on lane_center
    # compute total width of one full biopsy pair
    pair_width = 2*box_width + pair_gap
    # total width for all biopsy pairs + gaps between pairs (biopsy_gap between pairs)
    lane_span = len(biopsies) * pair_width + (len(biopsies)-1) * biopsy_gap
    # start so that lane is centered
    start = -lane_span / 2.0
    # offsets for each biopsy j
    biopsy_pair_starts = [start + j*(pair_width + biopsy_gap) for j in range(len(biopsies))]
    # final offsets for Δ and |Δ| within a lane, per biopsy
    rel_offsets_signed = [s for s in biopsy_pair_starts]
    rel_offsets_abs    = [s + box_width + pair_gap for s in biopsy_pair_starts]

    # begin plotting
    sns.set_style(seaborn_style)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    whis = 1.5 if whisker_mode == 'iqr1.5' else (5, 95)

    # draw per biopsy
    for bi, pair in enumerate(biopsies):
        color = sns.color_palette(palette, n_colors=len(biopsies))[bi % len(biopsies)]
        label = biopsy_label_map.get(pair, f"{pair[0]}, Bx {pair[1]}") if biopsy_label_map else f"{pair[0]}, Bx {pair[1]}"

        # positions for this biopsy across lanes
        pos_signed = lane_centers + rel_offsets_signed[bi] + box_width/2.0
        pos_abs    = lane_centers + rel_offsets_abs[bi]    + box_width/2.0

        gs = per_signed[pair]
        ga = per_abs[pair]

        # Δ (solid filled)
        if any(len(a) for a in gs):
            bp = ax.boxplot(gs, positions=pos_signed, widths=box_width,
                            manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor(color); box.set_alpha(0.35)
                box.set_edgecolor(color); box.set_linewidth(1.15)
            for k in ['whiskers','caps','medians']:
                for line in bp[k]:
                    line.set_color(color); line.set_linewidth(1.15)
            if show_points:
                rng = np.random.default_rng(1000 + bi)
                for x0, arr in zip(pos_signed, gs):
                    if arr.size:
                        xj = x0 + (rng.random(arr.size)-0.5)*jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='o', zorder=3)

        # |Δ| (dashed, hollow)
        if any(len(a) for a in ga):
            bp = ax.boxplot(ga, positions=pos_abs, widths=box_width,
                            manage_ticks=False, whis=whis, showfliers=showfliers, patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor('white'); box.set_alpha(1.0)
                box.set_edgecolor(color);  box.set_linewidth(1.15); box.set_linestyle('--')
            for k in ['whiskers','caps','medians']:
                for line in bp[k]:
                    line.set_color(color); line.set_linewidth(1.15); line.set_linestyle('--')
            if show_points:
                rng = np.random.default_rng(2000 + bi)
                for x0, arr in zip(pos_abs, ga):
                    if arr.size:
                        xj = x0 + (rng.random(arr.size)-0.5)*jitter_width
                        ax.scatter(xj, arr, s=point_size, alpha=point_alpha, color=color, marker='x', zorder=3)

    # x ticks at lane centers with voxel labels
    ax.set_xticks(lane_centers)
    if x_axis == 'Voxel index':
        ax.set_xticklabels([str(int(v)) for v in x_vals_sorted], rotation=0)
        xlab = 'Voxel index'
    else:
        ax.set_xticklabels([f"{v:g}" for v in x_vals_sorted], rotation=0)
        xlab = 'Voxel begin (Z) [mm]'

    # subtle vertical guides to clarify lanes
    for c in lane_centers:
        ax.axvline(c, color='0.85', linewidth=0.6, alpha=0.6, zorder=0)

    ax.set_xlabel(xlab, fontsize=axes_label_fontsize)
    ax.set_ylabel(_ylabel(metric, mode=y_label_mode), fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    if y0_refline:
        ax.axhline(0.0, linestyle=':', linewidth=1.0, alpha=0.85, color='0.25')

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Legend = biopsies only
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # biopsy color swatches
    patches = [
        Patch(facecolor=colors[i % len(colors)],
            edgecolor=colors[i % len(colors)],
            alpha=0.35,
            label=(biopsy_label_map.get(biopsies[i], f"{biopsies[i][0]}, Bx {biopsies[i][1]}")
                    if biopsy_label_map else f"{biopsies[i][0]}, Bx {biopsies[i][1]}"))
        for i in range(len(biopsies))
    ]

    # Δ / |Δ| style keys — no extra text
    delta_key   = Line2D([], [], color='black', linewidth=1.2, label=r'$\Delta$')
    absdelta_key= Line2D([], [], color='black', linewidth=1.2, linestyle='--', label=r'$|\Delta|$')

    leg = ax.legend(handles=patches + [delta_key, absdelta_key], loc=legend_loc, frameon=True)
    for t in leg.get_texts():
        t.set_fontsize(max(tick_label_fontsize - 0, 9))


    if show_title:
        if title is None:
            title = f"{metric} — Δ & |Δ| per voxel"
        ax.set_title(title, fontsize=axes_label_fontsize)

    if tight_layout:
        plt.tight_layout()

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for ext in save_formats:
        fp = output_dir / f"{plot_name_base}.{ext.lstrip('.')}"
        fig.savefig(fp, bbox_inches='tight', dpi=dpi)
        out_paths.append(fp)
    plt.close(fig)
    return out_paths












def plot_cohort_deltas_boxplot_by_voxel(
    deltas_df: pd.DataFrame,
    save_dir,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    x_axis: str = 'Voxel index',               
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    title: str | None = None,
    show_points: bool = True,   # new flag
    point_size: int = 3,        # adjust visibility
    alpha: float = 0.5,         # transparency for points
):
    """
    Boxplots of Nominal−(Mean/Mode/Q50) across the entire cohort.
    For each voxel position on the chosen x-axis, shows the distribution
    of delta values over all rows (patients/biopsies).
    Optionally overlays jittered points.
    """
    plt.ioff()

    # choose x-axis
    if x_axis == 'Voxel begin (Z)':
        x = deltas_df[('Voxel begin (Z)', '')]
        x_label = 'Voxel begin (Z) [mm]'
    else:
        x = deltas_df[('Voxel index', '')]
        x_label = 'Voxel index (along core)'

    x_num = pd.to_numeric(x, errors='coerce')
    df = deltas_df.copy()
    df = df.assign(_x=x_num)

    block = f"{zero_level_index_str} deltas"
    cols = [
        (block, 'nominal_minus_mean'),
        (block, 'nominal_minus_mode'),
        (block, 'nominal_minus_q50'),
    ]
    for c in cols:
        if c not in df.columns:
            raise KeyError(
                f"Missing {c}. Did you compute deltas with zero_level_index_str='{zero_level_index_str}'?"
            )

    tidy = df.loc[:, cols].copy()
    tidy.columns = ['Nominal - Mean', 'Nominal - Mode', 'Nominal - Median (Q50)']
    tidy = tidy.assign(x=df['_x'].values)
    tidy = tidy.melt(id_vars='x', var_name='Delta', value_name='Value')
    tidy = tidy.dropna(subset=['x', 'Value'])

    x_order = sorted(tidy['x'].unique())
    y_label = 'Delta (Gy/mm)' if 'grad' in zero_level_index_str else 'Delta (Gy)'

    sns.set(style='whitegrid')
    ax = sns.boxplot(
        data=tidy,
        x='x',
        y='Value',
        hue='Delta',
        order=x_order,
        showfliers=False,
    )

    # overlay points
    if show_points:
        sns.stripplot(
            data=tidy,
            x='x',
            y='Value',
            hue='Delta',
            order=x_order,
            dodge=True,        # separate by hue
            size=point_size,
            alpha=alpha,
            ax=ax,
            legend=False,      # avoid duplicate legend
        )

    ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.legend(title='', fontsize=tick_label_fontsize)

    if title is None:
        pass
    else:
        ax.set_title(title, fontsize=axes_label_fontsize)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, format='png')
    plt.close()
    return svg_path, png_path




def plot_cohort_deltas_boxplot_by_voxel(
    deltas_df: pd.DataFrame,
    save_dir,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    x_axis: str = 'Voxel index',               
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    title: str | None = None,
    show_points: bool = True,
    point_size: int = 3,
    alpha: float = 0.5,
    # ---- NEW options ----
    include_abs: bool = True,                 # include |Δ|
    abs_as_hue: bool = True,                  # hue=Signed vs Absolute & facet by Δ kind
    require_precomputed_abs: bool = True,     # expect abs block present
    fallback_recompute_abs: bool = False,     # compute |Δ| from signed if abs block missing
    label_style: str = 'math',                # 'math' -> Δ^{mean}/Δ^{mode}/Δ^{Q50}; 'text' -> Nominal - Mean, etc.
    median_superscript: str = 'Q50',          # used in math labels for median
    order_kinds: tuple = ('mean', 'mode', 'median'),  # Δ order across panels / legend
):
    """
    Cohort boxplots of Nominal−(Mean/Mode/Q50) per voxel position on the x-axis.
    Optionally overlays points and includes |Δ|, with math or text labels.
    Saves SVG & PNG.
    """
    plt.ioff()

    df = deltas_df.copy()

    # --- choose x-axis
    if x_axis == 'Voxel begin (Z)':
        x = df[('Voxel begin (Z)', '')]
        x_label = 'Voxel begin (Z) [mm]'
    else:
        x = df[('Voxel index', '')]
        x_label = 'Voxel index (along core)'

    x_num = pd.to_numeric(x, errors='coerce')
    df = df.assign(_x=x_num)

    # --- helpers for labels
    def _math_sup(kind: str) -> str:
        if kind == 'mean':   return r'\mathrm{mean}'
        if kind == 'mode':   return r'\mathrm{mode}'
        if kind == 'median': return r'\mathrm{' + (median_superscript.replace('%', r'\%')) + r'}'
        return r'\mathrm{' + kind + r'}'

    def _label(kind: str, absolute: bool) -> str:
        if label_style == 'math':
            base = rf'$\Delta^{{{_math_sup(kind)}}}$'
            return rf'$|{base[1:-1]}|$' if absolute else base
        else:
            txt = {'mean':'Nominal - Mean','mode':'Nominal - Mode','median':'Nominal - Median (Q50)'}[kind]
            return f'|{txt}|' if absolute else txt

    # --- signed delta columns
    block = f"{zero_level_index_str} deltas"
    signed_map = {
        'mean':   (block, 'nominal_minus_mean'),
        'mode':   (block, 'nominal_minus_mode'),
        'median': (block, 'nominal_minus_q50'),
    }
    missing_signed = [v for v in signed_map.values() if v not in df.columns]
    if missing_signed:
        raise KeyError(
            f"Missing signed delta columns for zero_level_index_str='{zero_level_index_str}'. "
            f"Missing: {missing_signed}"
        )

    signed_cols = [signed_map[k] for k in order_kinds]
    tidy_signed = df.loc[:, signed_cols].copy()
    tidy_signed.columns = [_label(k, absolute=False) for k in order_kinds]
    tidy_signed = tidy_signed.assign(x=df['_x'].values)
    tidy_signed = tidy_signed.melt(id_vars='x', var_name='Delta', value_name='Value').dropna(subset=['x','Value'])
    tidy_signed['Kind'] = 'Signed'

    # --- absolute delta columns (prefer precomputed)
    tidy_abs = None
    if include_abs:
        abs_block = f"{zero_level_index_str} abs deltas"
        abs_map = {
            'mean':   (abs_block, 'abs_nominal_minus_mean'),
            'mode':   (abs_block, 'abs_nominal_minus_mode'),
            'median': (abs_block, 'abs_nominal_minus_q50'),
        }
        has_abs = all(v in df.columns for v in abs_map.values())
        if not has_abs and require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError(
                "Absolute delta columns are missing and recomputation is disabled. "
                f"Expected abs columns: {list(abs_map.values())}. "
                "Use *_with_abs DataFrames or set fallback_recompute_abs=True."
            )

        if has_abs:
            abs_cols = [abs_map[k] for k in order_kinds]
            tidy_abs = df.loc[:, abs_cols].copy()
            tidy_abs.columns = [_label(k, absolute=True) for k in order_kinds]
            tidy_abs = tidy_abs.assign(x=df['_x'].values)
            tidy_abs = tidy_abs.melt(id_vars='x', var_name='Delta', value_name='Value').dropna(subset=['x','Value'])
            tidy_abs['Kind'] = 'Absolute'
        elif fallback_recompute_abs:
            tidy_abs = tidy_signed.copy()
            tidy_abs['Value'] = tidy_abs['Value'].abs()
            # convert labels to absolute
            tidy_abs['Delta'] = tidy_abs['Delta'].map(
                lambda s: _label('mean', True)   if ('mean'   in s or 'Mean'   in s) else
                          _label('mode', True)   if ('mode'   in s or 'Mode'   in s) else
                          _label('median', True)
            )
            tidy_abs['Kind'] = 'Absolute'

    # --- build plotting frame
    y_label = 'Delta (Gy/mm)' if 'grad' in zero_level_index_str.lower() else 'Delta (Gy)'
    x_order = sorted(tidy_signed['x'].unique())

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"

    sns.set(style='whitegrid')

    if include_abs and abs_as_hue and tidy_abs is not None:
        # Facet by Δ kind, hue = Signed vs Absolute
        # Normalize Δ labels to signed form for facet titles
        if label_style == 'math':
            def _to_signed(lbl: str) -> str:
                return lbl.replace('|$', '$').replace('$|', '$')
        else:
            def _to_signed(lbl: str) -> str:
                return lbl.strip('|')

        signed_h = tidy_signed.copy()
        abs_h = tidy_abs.copy()
        signed_h['DeltaShort'] = signed_h['Delta']
        abs_h['DeltaShort'] = abs_h['Delta'].map(_to_signed)

        tidy_plot = pd.concat([signed_h, abs_h], ignore_index=True)

        # order of facets
        facet_order = [_label(k, absolute=False) for k in order_kinds]

        g = sns.catplot(
            data=tidy_plot,
            x='x', y='Value',
            hue='Kind', hue_order=['Signed','Absolute'],
            col='DeltaShort', col_order=facet_order,
            order=x_order,
            kind='box',
            showfliers=False,
            height=3.2, aspect=1.4, sharey=True, sharex=True
        )
        if show_points:
            # overlay per facet
            def _strip(data, color, **kwargs):
                sns.stripplot(
                    data=data, x='x', y='Value', hue='Kind',
                    dodge=True, order=x_order, size=point_size, alpha=alpha, linewidth=0
                )
            g.map_dataframe(_strip)
            # dedupe legends (created twice by strip + box)
            for ax in g.axes.flatten():
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles[:2], labels[:2], title=None, fontsize=tick_label_fontsize)
        else:
            # keep one legend
            g.add_legend(title=None)
            g._legend.set_frame_on(True)
            for txt in g._legend.texts:
                txt.set_fontsize(tick_label_fontsize)

        # titles/labels
        for ax in g.axes.flatten():
            ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
            ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
            ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        if title:
            g.fig.suptitle(title, fontsize=axes_label_fontsize)
            g.fig.subplots_adjust(top=0.85)

        g.fig.tight_layout()
        g.fig.savefig(svg_path, format='svg')
        g.fig.savefig(png_path, format='png')
        plt.close(g.fig)

    else:
        # Single axes: hue distinguishes Δ kinds; include_abs=False → 3; include_abs=True → 6 categories
        tidy_plot = tidy_signed if (not include_abs or tidy_abs is None) else pd.concat([tidy_signed, tidy_abs], ignore_index=True)

        ax = sns.boxplot(
            data=tidy_plot,
            x='x', y='Value',
            hue='Delta', order=x_order,
            showfliers=False
        )
        if show_points:
            sns.stripplot(
                data=tidy_plot,
                x='x', y='Value',
                hue='Delta', order=x_order,
                dodge=True, size=point_size, alpha=alpha, ax=ax, legend=False
            )

        ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
        ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        if title:
            ax.set_title(title, fontsize=axes_label_fontsize)

        # style legend
        leg = ax.legend(title='', frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(tick_label_fontsize)

        plt.tight_layout()
        plt.savefig(svg_path, format='svg')
        plt.savefig(png_path, format='png')
        plt.close()

    return svg_path, png_path









def plot_cohort_deltas_boxplot(
    deltas_df: pd.DataFrame,
    save_dir,
    fig_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    include_patient_ids: list | None = None,
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    title: str | None = None,
    show_points: bool = False,
    show_counts_in_title: bool = True,
    # ---- absolute controls ----
    include_abs: bool = True,
    abs_as_hue: bool = True,
    require_precomputed_abs: bool = True,
    fallback_recompute_abs: bool = False,
    # ---- labeling controls ----
    label_style: str = 'math',          # 'math' -> Δ^{mean} / |Δ^{mean}| ; 'text' -> Nominal - Mean / |Nominal - Mean|
    median_superscript: str = 'Q50',    # used when label_style='math': 'Q50' or 'median'
    order_kinds: tuple = ('mean', 'mode', 'median'),  # order along the x-axis
):
    """
    Cohort boxplot of Nominal−(Mean/Mode/Q50) with optional absolutes.
    - Math labels: Δ^{mean}, Δ^{mode}, Δ^{Q50}; abs as |Δ^{⋅}| when abs_as_hue=False.
    - With abs_as_hue=True, x shows Δ^{⋅} and legend shows Signed vs Absolute.
    Saves both SVG and PNG.
    """
    plt.ioff()

    data = deltas_df
    if include_patient_ids is not None:
        pid_key = ('Patient ID','')
        if pid_key not in data.columns:
            raise KeyError("Patient filter requested but ('Patient ID','') column not found.")
        data = data[data[pid_key].isin(include_patient_ids)]
        if data.empty:
            raise ValueError("Patient filter returned no rows.")

    # counts
    n_biopsies = (
        data.loc[:, [('Patient ID',''), ('Bx index','')]]
        .drop_duplicates()
        .shape[0]
    )
    n_voxels = data.shape[0]

    # ----- helpers for labels -----
    def _math_sup(kind: str) -> str:
        if kind == 'mean':
            return r'\mathrm{mean}'
        if kind == 'mode':
            return r'\mathrm{mode}'
        if kind == 'median':
            return r'\mathrm{' + (median_superscript.replace('%', r'\%')) + r'}'
        return r'\mathrm{' + kind + r'}'

    def _label(kind: str, absolute: bool) -> str:
        # kind ∈ {'mean','mode','median'}
        if label_style == 'math':
            base = rf'$\Delta^{{{_math_sup(kind)}}}$'
            return rf'$|{base[1:-1]}|$' if absolute else base
        else:  # 'text'
            txt = {'mean': 'Nominal - Mean', 'mode': 'Nominal - Mode', 'median': 'Nominal - Median (Q50)'}[kind]
            return f'|{txt}|' if absolute else txt

    # ----- columns -----
    block = f"{zero_level_index_str} deltas"
    signed_map = {
        'mean':   (block, 'nominal_minus_mean'),
        'mode':   (block, 'nominal_minus_mode'),
        'median': (block, 'nominal_minus_q50'),
    }
    missing_signed = [v for k,v in signed_map.items() if v not in data.columns]
    if missing_signed:
        raise KeyError(
            f"Missing signed delta columns for zero_level_index_str='{zero_level_index_str}'. "
            f"Missing: {missing_signed}"
        )

    # build tidy signed
    signed_cols = [signed_map[k] for k in order_kinds]
    tidy_signed = data.loc[:, signed_cols].copy()
    tidy_signed.columns = [_label(k, absolute=False) for k in order_kinds]
    tidy_signed = tidy_signed.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])
    tidy_signed['Kind'] = 'Signed'

    # abs block (prefer precomputed)
    tidy_abs = None
    if include_abs:
        abs_block = f"{zero_level_index_str} abs deltas"
        abs_map = {
            'mean':   (abs_block, 'abs_nominal_minus_mean'),
            'mode':   (abs_block, 'abs_nominal_minus_mode'),
            'median': (abs_block, 'abs_nominal_minus_q50'),
        }
        has_abs = all(v in data.columns for v in abs_map.values())
        if not has_abs and require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError(
                "Absolute delta columns are missing and recomputation is disabled. "
                f"Expected abs columns: {list(abs_map.values())}. "
                "Use *_with_abs DataFrames or set fallback_recompute_abs=True."
            )

        if has_abs:
            abs_cols = [abs_map[k] for k in order_kinds]
            tidy_abs = data.loc[:, abs_cols].copy()
            tidy_abs.columns = [_label(k, absolute=True) for k in order_kinds]
            tidy_abs = tidy_abs.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])
            tidy_abs['Kind'] = 'Absolute'
        elif fallback_recompute_abs:
            tidy_abs = tidy_signed.copy()
            tidy_abs['Value'] = tidy_abs['Value'].abs()
            tidy_abs['Delta'] = tidy_abs['Delta'].map(lambda s: _label(
                {'mean': 'mean', 'mode': 'mode', 'median': 'median'}[
                    'mean' if 'Mean' in s else ('mode' if 'Mode' in s else 'median')
                ], absolute=True
            ))
            tidy_abs['Kind'] = 'Absolute'

    # compose plotting frame
    if include_abs and abs_as_hue and tidy_abs is not None:
        # Use same x tick labels (Δ^{⋅}); hue distinguishes Signed vs Absolute
        # Convert abs labels back to signed form for x so categories match
        def _to_signed_xtick(lbl: str) -> str:
            if label_style == 'math':
                return lbl.replace('|$', '$').replace('$|', '$')  # strip surrounding |…|
            else:
                return lbl.strip('|')
        signed_for_hue = tidy_signed.copy()
        abs_for_hue = tidy_abs.copy()
        abs_for_hue['Delta'] = abs_for_hue['Delta'].map(_to_signed_xtick)
        tidy_plot = pd.concat([signed_for_hue, abs_for_hue], ignore_index=True)

        x_order = [_label(k, absolute=False) for k in order_kinds]
        hue_order = ['Signed', 'Absolute']
    elif include_abs and tidy_abs is not None:
        # six boxes: Δ^{⋅} and |Δ^{⋅}| appear as separate x categories (tick shows bars)
        tidy_plot = pd.concat([tidy_signed, tidy_abs], ignore_index=True)
        x_order = [_label(k, False) for k in order_kinds] + [_label(k, True) for k in order_kinds]
        hue_order = None
    else:
        tidy_plot = tidy_signed.copy()
        x_order = [_label(k, absolute=False) for k in order_kinds]
        hue_order = None

    # y-axis label
    y_label = 'Delta (Gy/mm)' if 'grad' in zero_level_index_str.lower() else 'Delta (Gy)'

    # plot
    sns.set(style='whitegrid')
    if include_abs and abs_as_hue and tidy_abs is not None:
        ax = sns.boxplot(data=tidy_plot, x='Delta', y='Value', hue='Kind', order=x_order, hue_order=hue_order, showfliers=False)
        if show_points:
            sns.stripplot(data=tidy_plot, x='Delta', y='Value', hue='Kind', dodge=True, order=x_order, hue_order=hue_order,
                          color='k', alpha=0.20, jitter=0.25, size=2)
            # dedupe legend
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 2:
                ax.legend(handles[:2], labels[:2], title=None)
    else:
        ax = sns.boxplot(data=tidy_plot, x='Delta', y='Value', order=x_order, showfliers=False)
        if show_points:
            sns.stripplot(data=tidy_plot, x='Delta', y='Value', order=x_order, color='k', alpha=0.20, jitter=0.25, size=2)

    # title
    if title is None:
        title = f"Cohort — {zero_level_index_str} deltas"
    if show_counts_in_title:
        title = f"{title}  (biopsies: {n_biopsies}, voxels: {n_voxels})"
    ax.set_title(title, fontsize=axes_label_fontsize)

    # axes
    ax.set_xlabel('', fontsize=axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    # legend styling
    if include_abs and abs_as_hue and tidy_abs is not None:
        leg = ax.legend(title=None, frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(tick_label_fontsize)

    # save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, format='png')
    plt.close()
    return svg_path, png_path


























########## delta D vs grad plots################





# ---------- helper: robust binned trend (with safer fallbacks) ----------
def compute_binned_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    bins: int = 24,
    binning: str = "quantile",   # "quantile" or "width"
    min_per_bin: int = 20,
    qs: Tuple[float,...] = (0.05, 0.25, 0.50, 0.75, 0.95)
) -> pd.DataFrame:
    """Return per-bin robust summary: n, q05/q25/median/q75/q95, and x_center."""
    dd = df[[x, y]].copy()
    dd[x] = pd.to_numeric(dd[x], errors='coerce')
    dd[y] = pd.to_numeric(dd[y], errors='coerce')
    dd = dd.dropna()
    if dd.empty:
        return pd.DataFrame(columns=["x_center","n","q05","q25","q50","q75","q95"])

    # choose edges
    def _edges_quantile(d: pd.Series, k: int) -> np.ndarray:
        qs_edges = np.linspace(0, 1, k + 1)
        e = d.quantile(qs_edges).to_numpy()
        e = np.unique(e)
        if e.size < 3:  # not enough unique edges → fallback to width
            lo, hi = d.min(), d.max()
            e = np.linspace(lo, hi, min(k, 10) + 1)
        return e

    if binning == "quantile":
        edges = _edges_quantile(dd[x], bins)
    else:
        lo, hi = dd[x].min(), dd[x].max()
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            # no spread, return empty
            return pd.DataFrame(columns=["x_center","n","q05","q25","q50","q75","q95"])
        edges = np.linspace(lo, hi, bins + 1)

    dd["__bin__"] = pd.cut(dd[x], bins=edges, include_lowest=True, duplicates="drop")
    gb = dd.groupby("__bin__", observed=True, sort=True)

    rows = []
    for b, g in gb:
        if g.shape[0] < min_per_bin:
            continue
        q = g[y].quantile(qs)
        lo5, q25, med, q75, hi95 = q.iloc[0], q.iloc[1], q.iloc[2], q.iloc[3], q.iloc[4]
        if isinstance(b, pd.Interval):
            x_center = 0.5 * (float(b.left) + float(b.right))
        else:
            try:
                left, right = [float(v) for v in str(b).strip("[]()").split(",")]
                x_center = 0.5 * (left + right)
            except Exception:
                x_center = np.nan
        rows.append({"x_center": x_center, "n": int(g.shape[0]),
                     "q05": lo5, "q25": q25, "q50": med, "q75": q75, "q95": hi95})

    out = pd.DataFrame(rows).dropna(subset=["x_center"])
    return out


# ---------- main plotter (fixed grouping + optional regression) ----------
def plot_delta_vs_gradient(
    long_df: pd.DataFrame,
    save_dir,
    fig_name: str,
    *,
    # what to plot
    gradient_cols: Optional[Iterable[str]] = None,   # None -> auto: all 'Grad[...]'
    delta_kinds: Iterable[str] = ("Δ_mode", "Δ_median", "Δ_mean"),  # by prefix match
    y_variant: str = "both",          # 'signed' | 'abs' | 'both'
    use_log1p_abs: bool = False,      # when plotting abs, use log1p(|Δ|)
    # visuals
    show_scatter: bool = True,
    scatter_sample: int = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 8.0,
    # trend grouping & legend
    hue_by: str = "Measure",          # 'Measure' or 'Delta kind'  (default fixed to 'Measure' for 'both')
    trend_split: str = "auto",        # 'auto' | 'Measure' | 'Delta kind'
    # binned trend
    bins: int = 24,
    binning: str = "quantile",
    min_per_bin: int = 20,
    show_iqr_band: bool = True,
    show_90_band: bool = True,
    # optional regression over raw points (per panel & group)
    regression: str = "none",         # 'none' | 'ols' | 'loess'
    poly_order: int = 1,              # for 'ols' (1 = linear)
    loess_frac: float = 0.2,          # for 'loess' if statsmodels is available
    max_points_for_reg: int = 50000,  # downsample for regression fit
    # labels & fonts
    label_style: str = "math",
    median_superscript: str = "Q50",
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    legend_fontsize: int = 11,
    title: Optional[str] = None,
    # layout & export
    facet_cols: int = 2,
    height: float = 3.2,
    aspect: float = 1.4,
):
    """
    Δ (and/or |Δ|) vs dose-gradient with robust binned trends (+ optional regression).
    Expects the long table from build_deltas_vs_gradient_df_with_abs(..., return_long=True):
        ['Patient ID','Bx index','Voxel index', 'Grad[⋯]', 'Delta kind', 'Delta (signed)', '|Delta|','log1p|Delta|']
    """
    plt.ioff()
    df = long_df.copy()

    # --- choose gradient columns ---
    if gradient_cols is None:
        gradient_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Grad[")]
    gradient_cols = list(gradient_cols)
    if not gradient_cols:
        raise KeyError("No gradient columns found. Ensure columns like 'Grad[nominal] (Gy/mm)' exist.")

    # --- normalize 'Delta kind' to prefixes (Δ_mode / Δ_median / Δ_mean) ---
    def _kind_of(s: str) -> str:
        if isinstance(s, str) and "Δ_mode" in s: return "Δ_mode"
        if isinstance(s, str) and "Δ_median" in s: return "Δ_median"
        if isinstance(s, str) and "Δ_mean" in s: return "Δ_mean"
        return str(s)

    df["__kind__"] = df["Delta kind"].map(_kind_of)
    df = df[df["__kind__"].isin(delta_kinds)].copy()
    if df.empty:
        raise ValueError("No rows after filtering to requested delta_kinds.")

    # --- build 'Value' + 'Measure' (Signed/Absolute/Absolute (log1p)) ---
    parts = []
    if y_variant in ("signed", "both"):
        tmp = df.copy()
        tmp["Value"] = pd.to_numeric(tmp["Delta (signed)"], errors='coerce')
        tmp["Measure"] = "Signed"
        parts.append(tmp)

    if y_variant in ("abs", "both"):
        if use_log1p_abs:
            if "log1p|Delta|" not in df.columns:
                raise KeyError("use_log1p_abs=True but 'log1p|Delta|' column is missing.")
            tmp = df.copy()
            tmp["Value"] = pd.to_numeric(tmp["log1p|Delta|"], errors='coerce')
            tmp["Measure"] = "Absolute (log1p)"
        else:
            if "|Delta|" not in df.columns:
                raise KeyError("y_variant includes abs but '|Delta|' column is missing.")
            tmp = df.copy()
            tmp["Value"] = pd.to_numeric(tmp["|Delta|"], errors='coerce')
            tmp["Measure"] = "Absolute"
        parts.append(tmp)

    plot_df = pd.concat(parts, ignore_index=True).dropna(subset=["Value"])

    # --- trend grouping: never mix Signed with Absolute in same trend ---
    if trend_split == "auto":
        split_col = "Measure" if y_variant == "both" else "__kind__"
    else:
        split_col = trend_split
        if y_variant == "both" and split_col != "Measure":
            # force a safe choice: separate trends by Signed vs Absolute
            split_col = "Measure"

    # hue choice for scatter
    if hue_by not in ("Measure", "Delta kind", "__kind__"):
        hue_by = "Measure" if y_variant == "both" else "Delta kind"

    # layout
    n = len(gradient_cols)
    ncols = min(facet_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(aspect*height*ncols, height*nrows), squeeze=False)
    sns.set(style="whitegrid")

    # optional regression helpers
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
        _has_loess = True
    except Exception:
        _has_loess = False

    # panels
    for i, gcol in enumerate(gradient_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        pane = plot_df[[gcol, "Value", "__kind__", "Delta kind", "Measure"]].dropna().rename(columns={gcol: "Grad"})

        # scatter (downsample for speed)
        if show_scatter and not pane.empty:
            pp = pane
            if scatter_sample and pp.shape[0] > scatter_sample:
                pp = pp.sample(scatter_sample, random_state=42)
            sns.scatterplot(
                data=pp,
                x="Grad", y="Value",
                hue=(hue_by if hue_by in pp.columns else None),
                alpha=scatter_alpha, s=scatter_size, ax=ax, legend=False
            )

        # trend groups
        groups = pane[split_col].dropna().unique()
        palette = sns.color_palette(n_colors=len(groups))

        for color, grp in zip(palette, groups):
            sub = pane[pane[split_col] == grp]
            if sub.empty:
                continue

            # robust binned trend
            trend = compute_binned_trend(
                sub.rename(columns={"Grad":"x","Value":"y"}),
                x="x", y="y",
                bins=bins, binning=binning, min_per_bin=min_per_bin
            )

            # if no bins survive, relax criteria once
            if trend.empty and binning == "quantile":
                trend = compute_binned_trend(
                    sub.rename(columns={"Grad":"x","Value":"y"}),
                    x="x", y="y", bins=max(8, bins//2), binning="width", min_per_bin=max(5, min_per_bin//2)
                )

            if not trend.empty:
                if show_90_band:
                    ax.fill_between(trend["x_center"], trend["q05"], trend["q95"], color=color, alpha=0.08, lw=0)
                if show_iqr_band:
                    ax.fill_between(trend["x_center"], trend["q25"], trend["q75"], color=color, alpha=0.18, lw=0)
                ax.plot(trend["x_center"], trend["q50"], color=color, lw=2.0, label=str(grp))

            # optional regression (over raw points)
            if regression != "none" and not sub.empty:
                rr = sub
                if max_points_for_reg and rr.shape[0] > max_points_for_reg:
                    rr = rr.sample(max_points_for_reg, random_state=123)

                x = pd.to_numeric(rr["Grad"], errors='coerce').to_numpy()
                y = pd.to_numeric(rr["Value"], errors='coerce').to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if x.size >= 5:
                    xs = np.linspace(x.min(), x.max(), 200)
                    if regression == "ols":
                        # simple polynomial fit
                        coefs = np.polyfit(x, y, deg=poly_order)
                        ys = np.polyval(coefs, xs)
                        ax.plot(xs, ys, color=color, lw=2.0, ls='--')
                    elif regression == "loess" and _has_loess:
                        sm = _lowess(y, x, frac=loess_frac, it=1, return_sorted=True)
                        ax.plot(sm[:,0], sm[:,1], color=color, lw=2.0, ls='--')

        # labels
        ax.set_xlabel(gcol, fontsize=axes_label_fontsize)
        # y label (generic; legend carries the group meaning)
        if y_variant == "signed":
            ylab = r"$\Delta$"
        elif use_log1p_abs:
            ylab = r"$\log(1+|\Delta|)$"
        else:
            ylab = r"$|\Delta|$"
        ax.set_ylabel(ylab, fontsize=axes_label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        # legend
        if len(groups):
            leg = ax.legend(title=None, frameon=True, fontsize=legend_fontsize)
            if leg:
                for txt in leg.get_texts():
                    txt.set_fontsize(legend_fontsize)

    # hide unused axes
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # title
    if title:
        fig.suptitle(title, fontsize=axes_label_fontsize)
        fig.subplots_adjust(top=0.90)

    # save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    svg_path = Path(save_dir) / f"{fig_name}.svg"
    png_path = Path(save_dir) / f"{fig_name}.png"
    fig.tight_layout()
    fig.savefig(svg_path, format="svg", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return svg_path, png_path




def _stat_tex(stat: str, overrides: dict | None) -> str:
    """
    Map a gradient 'stat' token to LaTeX-friendly text.
    """
    defaults = {
        "nominal": r"\mathrm{nom}",
        "mean":    r"\mathrm{mean}",
        "mode":    r"\mathrm{mode}",
        "median":  r"\mathrm{median}",
        "q50":     r"\mathrm{Q50}",
        "quantile_50": r"\mathrm{Q50}",
    }
    if overrides and stat in overrides:
        return overrides[stat]
    return defaults.get(stat, rf"\mathrm{{{stat}}}")

def _x_label_for(gcol: str, label_style: str, grad_stat_tex: dict | None) -> str:
    """
    'Grad[nominal] (Gy/mm)' -> LaTeX '||∇D||_{nom} (Gy mm^{-1})'
    Keeps working even if the units part is missing.
    """
    m = re.search(r"Grad\[(.*?)\]", gcol)
    stat = m.group(1) if m else "nominal"

    if label_style != "latex":
        # Plain text, but fix the unit formatting to Gy mm^-1
        return f"Grad[{stat}] (Gy mm^-1)"

    s = _stat_tex(stat, grad_stat_tex)
    # Mathtext LaTeX; do not require usetex=True
    return rf"$\|\nabla D\|_{{{s}}}\ \mathrm{{(Gy\ mm^{{-1}})}}$"

def _y_label_for(y_col: str, label_style: str, j_symbol: str, idx_sub: tuple[str,str]) -> str:
    """
    Map internal y_col → axis label with indices/superscripts.
    """
    b, v = idx_sub
    is_abs = (y_col == "|Delta|")
    is_logabs = (y_col == "log1p|Delta|")

    if label_style != "latex":
        if is_abs:     return f"|Delta D|^{j_symbol}_{{{b},{v}}}"
        if is_logabs:  return f"log(1+|Delta D|)^{j_symbol}_{{{b},{v}}}"
        return f"Delta D^{j_symbol}_{{{b},{v}}}"

    if is_abs:
        return rf"$|\Delta D|^{{{j_symbol}}}_{{{b},{v}}}$"
    if is_logabs:
        return rf"$\log(1+|\Delta D|)^{{{j_symbol}}}_{{{b},{v}}}$"
    return rf"$\Delta D^{{{j_symbol}}}_{{{b},{v}}}$"

def _legend_label_for(kind: str, y_col: str, label_style: str, idx_sub: tuple[str,str]) -> str:
    """
    Δ_mode → Δ^{mode}_{b,v}  (or |Δ|^{mode}_{b,v} if abs plot)
    """
    if label_style != "latex":
        return kind

    b, v = idx_sub
    is_abs = (y_col == "|Delta|") or (y_col == "log1p|Delta|")
    # which stat?
    stat = "mode" if "mode" in kind else ("median" if "median" in kind else ("mean" if "mean" in kind else kind))
    if is_abs:
        return rf"$|\Delta|^{{\mathrm{{{stat}}}}}_{{{b},{v}}}$"
    else:
        return rf"$\Delta^{{\mathrm{{{stat}}}}}_{{{b},{v}}}$"

def _relabel_legend_texts(legend, y_col: str, label_style: str, idx_sub: tuple[str,str]) -> None:
    """
    Mutate legend texts in-place to LaTeX forms.
    """
    if not legend: 
        return
    for txt in legend.get_texts():
        raw = txt.get_text()
        txt.set_text(_legend_label_for(raw, y_col, label_style, idx_sub))





def _kind_prefix(s: str) -> str:
    if isinstance(s, str) and "Δ_mode"   in s: return "Δ_mode"
    if isinstance(s, str) and "Δ_median" in s: return "Δ_median"
    if isinstance(s, str) and "Δ_mean"   in s: return "Δ_mean"
    return str(s)


def _ols_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors='coerce'); y = pd.to_numeric(y, errors='coerce')
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(), y[m].to_numpy()
    n = x.size
    out = {'n': int(n), 'slope': np.nan, 'slope_lo': np.nan, 'slope_hi': np.nan,
           'intercept': np.nan, 'r2': np.nan, 'rho': np.nan, 'rho_p': np.nan}
    if n < 3: return out
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    out['intercept'] = float(model.params[0])
    out['slope']     = float(model.params[1])
    ci = np.asarray(model.conf_int(alpha=0.05))
    out['slope_lo']  = float(ci[1,0])
    out['slope_hi']  = float(ci[1,1])
    out['r2']        = float(model.rsquared)
    rho, p = spearmanr(x, y)
    out['rho'], out['rho_p'] = float(rho), float(p)
    return out


def _panel_regplot(ax, df, xcol, ycol, hue, ci=95, scatter=True,
                   scatter_sample=20000, scatter_alpha=0.15, scatter_size=10.0):
    # scatter (downsample for speed/overplotting)
    if scatter:
        dd = df if (scatter_sample is None or len(df) <= scatter_sample) else df.sample(scatter_sample, random_state=42)
        sns.scatterplot(data=dd, x=xcol, y=ycol, hue=hue, ax=ax,
                        alpha=scatter_alpha, s=scatter_size, legend=False)

    # regression per hue with seaborn (OLS + CI)
    for k, sub in df.groupby(hue, observed=True, sort=False):
        sns.regplot(data=sub, x=xcol, y=ycol, ax=ax, scatter=False, ci=ci, label=str(k))

def _plot_delta_vs_gradient_pkg_core(
    long_df: pd.DataFrame,
    save_dir,
    file_prefix: str,
    *,
    y_col: str,
    gradient_cols=None,
    delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
    scatter: bool = True,
    scatter_sample: int = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 10.0,
    ci: int = 95,
    annotate_stats: bool = False,
    write_stats_csv: bool = True,
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    legend_fontsize: int = 11,
    height: float = 3.2,
    aspect: float = 1.4,
    facet_cols: int = 2,
    title: str | None = None,
    # label/style knobs
    label_style: str = "latex",         # "plain" or "latex"
    idx_sub: tuple[str,str] = ("b","v"),
    j_symbol: str = "j",
    grad_stat_tex: dict | None = None,
):
    """
    Core engine used by the two wrappers below.
    Expects long_df with columns:
      ['Delta kind', 'Delta (signed)', '|Delta|', 'log1p|Delta|', 'Grad[...] (Gy/mm)', ...]
    """
    df = long_df.copy()

    # pick gradient columns
    if gradient_cols is None:
        gradient_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Grad[")]
    gradient_cols = list(gradient_cols)
    if not gradient_cols:
        raise KeyError("No gradient columns detected (need columns like 'Grad[nominal] (Gy/mm)')")

    # normalize Δ-kind and filter
    df["__kind__"] = df["Delta kind"].map(_kind_prefix)
    df = df[df["__kind__"].isin(delta_kinds)].copy()
    if df.empty:
        raise ValueError("No rows after filtering to requested delta_kinds")

    # |Δ| should be non-negative
    if y_col == "|Delta|" and (pd.to_numeric(df[y_col], errors='coerce').dropna() < 0).any():
        raise ValueError("|Delta| contains negatives — upstream column mixing issue.")

    # layout
    n = len(gradient_cols)
    ncols = min(facet_cols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(aspect*height*ncols, height*nrows), squeeze=False)
    sns.set(style="whitegrid")

    # collect stats for CSV
    stats_rows = []

    for i, gcol in enumerate(gradient_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        pane = df[[gcol, "Delta kind", y_col]].dropna().rename(columns={gcol: "Grad", y_col: "Value"})
        if pane.empty:
            ax.set_visible(False)
            continue

        # plot
        _panel_regplot(ax, pane, xcol="Grad", ycol="Value", hue="Delta kind",
                       ci=ci, scatter=scatter, scatter_sample=scatter_sample,
                       scatter_alpha=scatter_alpha, scatter_size=scatter_size)

        # labels
        ax.set_xlabel(_x_label_for(gcol, label_style, grad_stat_tex), fontsize=axes_label_fontsize)
        ax.set_ylabel(_y_label_for(y_col, label_style, j_symbol, idx_sub), fontsize=axes_label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        # single legend call + relabel (do NOT call legend() again after this)
        leg = ax.legend(title=None, frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(legend_fontsize)
            _relabel_legend_texts(leg, y_col, label_style, idx_sub)

        # stats per Δ-kind
        panel_lines = []
        for kind, sub in pane.groupby("Delta kind", observed=True, sort=False):
            s = _ols_stats(sub["Grad"], sub["Value"])
            stats_rows.append({
                'gradient_col': gcol,
                'delta_kind': kind,
                'y_col': y_col,
                **s
            })
            panel_lines.append(f"{kind}: slope={s['slope']:.3f} [{s['slope_lo']:.3f},{s['slope_hi']:.3f}], "
                               f"ρ={s['rho']:.3f}, R²={s['r2']:.3f}, n={s['n']}")

        if annotate_stats and panel_lines:
            ax.text(0.02, 0.98, "\n".join(panel_lines[:3]), transform=ax.transAxes,
                    va='top', ha='left', fontsize=max(legend_fontsize-1, 9),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, lw=0))

    # hide unused axes
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=axes_label_fontsize)
        fig.subplots_adjust(top=0.90)

    # save images
    out_dir = Path(save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{file_prefix}.svg"
    png = out_dir / f"{file_prefix}.png"
    fig.tight_layout()
    fig.savefig(svg, format="svg", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)

    # optional CSV of slopes/ρ/R²
    stats_df = pd.DataFrame(stats_rows)
    csv_path = None
    if write_stats_csv and not stats_df.empty:
        csv_path = out_dir / f"{file_prefix}__stats.csv"
        stats_df.to_csv(csv_path, index=False)

    return svg, png, csv_path, stats_df


# ---------------- public wrappers ----------------

def plot_abs_delta_vs_gradient_pkg(
    long_df: pd.DataFrame,
    save_dir,
    file_prefix: str,
    label_style="latex",      # "plain" or "latex"
    idx_sub=("b","v"),        # indices shown as _{b,v}
    j_symbol="j",             # y-axis generic superscript
    grad_stat_tex=None,       # optional dict to customize stat labels
    *,
    gradient_cols=None,
    delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
    use_log1p: bool = False,
    **kwargs
):
    """
    |Δ| vs gradient (or log1p(|Δ|) if use_log1p=True), with OLS+95% CI per Δ-kind, per gradient panel.
    Also returns/saves per-panel stats (slope±CI, ρ, R², n).
    """
    y_col = "log1p|Delta|" if use_log1p else "|Delta|"
    return _plot_delta_vs_gradient_pkg_core(
        long_df, save_dir, file_prefix,
        y_col=y_col,
        gradient_cols=gradient_cols,
        delta_kinds=delta_kinds,
        label_style=label_style,
        idx_sub=idx_sub,
        j_symbol=j_symbol,
        grad_stat_tex=grad_stat_tex,
        **kwargs
    )



def plot_signed_delta_vs_gradient_pkg(
    long_df: pd.DataFrame,
    save_dir,
    file_prefix: str,
    label_style="latex",      # "plain" or "latex"
    idx_sub=("b","v"),        # indices shown as _{b,v}
    j_symbol="j",             # y-axis generic superscript
    grad_stat_tex=None,       # optional dict to customize stat labels
    *,
    gradient_cols=None,
    delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
    **kwargs
):
    """
    Signed Δ vs gradient, with OLS+95% CI per Δ-kind, per gradient panel.
    Also returns/saves per-panel stats (slope±CI, ρ, R², n).
    """
    return _plot_delta_vs_gradient_pkg_core(
        long_df, save_dir, file_prefix,
        y_col="Delta (signed)",
        gradient_cols=gradient_cols,
        delta_kinds=delta_kinds,
        label_style=label_style,
        idx_sub=idx_sub,
        j_symbol=j_symbol,
        grad_stat_tex=grad_stat_tex,
        **kwargs
    )





from pathlib import Path
from typing import Iterable, Optional, Tuple, List
def plot_abs_delta_vs_gradient_pkg_batch(
    long_df: pd.DataFrame,
    save_dir,
    base_prefix: str,
    *,
    gradient_cols: Optional[Iterable[str]] = None,      # None -> auto all Grad[⋯]
    delta_kinds: Iterable[str] = ("Δ_mode","Δ_median","Δ_mean"),
    use_log1p: bool = False,
    # viz defaults (explicit so you see them at the callsite)
    scatter: bool = True,
    scatter_sample: int = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 10.0,
    ci: int = 95,
    annotate_stats: bool = False,
    write_stats_csv: bool = True,
    axes_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_fontsize: int = 12,
    height_single: float = 3.0,
    aspect_single: float = 1.6,
    height_combined: float = 3.0,
    aspect_combined: float = 1.5,
    facet_cols_combined: int = 2,
    # label/style knobs
    label_style="latex",
    idx_sub=("b","v"),
    j_symbol="j",
    grad_stat_tex=None,
) -> Tuple[List[Path], List[Path], Optional[Path]]:
    """
    Returns (per_grad_svgs, per_grad_pngs, combined_stats_csv_path)
    """
    if gradient_cols is None:
        gradient_cols = [c for c in long_df.columns if isinstance(c, str) and c.startswith("Grad[")]
    gradient_cols = list(gradient_cols)
    if not gradient_cols:
        raise KeyError("No gradient columns detected for batch plotting.")

    per_svgs, per_pngs = [], []

    # 1) per-gradient single-panel figs
    for g in gradient_cols:
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", g).strip("_")
        file_prefix = f"{base_prefix}__{suffix}"
        svg, png, stats_csv, _stats_df = plot_abs_delta_vs_gradient_pkg(
            long_df=long_df,
            save_dir=save_dir,
            file_prefix=file_prefix,
            gradient_cols=[g],
            delta_kinds=delta_kinds,
            use_log1p=use_log1p,
            scatter=scatter, scatter_sample=scatter_sample,
            scatter_alpha=scatter_alpha, scatter_size=scatter_size,
            ci=ci,
            annotate_stats=annotate_stats,
            write_stats_csv=write_stats_csv,
            axes_label_fontsize=axes_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            legend_fontsize=legend_fontsize,
            height=height_single,
            aspect=aspect_single,
            facet_cols=1,
            title=None,
            # pass label knobs through
            label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol, grad_stat_tex=grad_stat_tex,
        )
        per_svgs.append(svg); per_pngs.append(png)

    # 2) combined multi-panel fig (all gradients)
    svg_c, png_c, stats_csv_c, _ = plot_abs_delta_vs_gradient_pkg(
        long_df=long_df,
        save_dir=save_dir,
        file_prefix=f"{base_prefix}__combined",
        gradient_cols=gradient_cols,
        delta_kinds=delta_kinds,
        use_log1p=use_log1p,
        scatter=scatter, scatter_sample=scatter_sample,
        scatter_alpha=scatter_alpha, scatter_size=scatter_size,
        ci=ci,
        annotate_stats=annotate_stats,
        write_stats_csv=write_stats_csv,
        axes_label_fontsize=axes_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        height=height_combined,
        aspect=aspect_combined,
        facet_cols=facet_cols_combined,
        title=None,
        # pass label knobs through
        label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol, grad_stat_tex=grad_stat_tex,
    )

    return per_svgs + [svg_c], per_pngs + [png_c], stats_csv_c

def plot_signed_delta_vs_gradient_pkg_batch(
    long_df: pd.DataFrame,
    save_dir,
    base_prefix: str,
    *,
    gradient_cols: Optional[Iterable[str]] = None,
    delta_kinds: Iterable[str] = ("Δ_mode","Δ_median","Δ_mean"),
    scatter: bool = True,
    scatter_sample: int = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 10.0,
    ci: int = 95,
    annotate_stats: bool = False,
    write_stats_csv: bool = True,
    axes_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_fontsize: int = 12,
    height_single: float = 3.0,
    aspect_single: float = 1.6,
    height_combined: float = 3.0,
    aspect_combined: float = 1.5,
    facet_cols_combined: int = 2,
    # label/style knobs
    label_style="latex",
    idx_sub=("b","v"),
    j_symbol="j",
    grad_stat_tex=None,
) -> Tuple[List[Path], List[Path], Optional[Path]]:
    if gradient_cols is None:
        gradient_cols = [c for c in long_df.columns if isinstance(c, str) and c.startswith("Grad[")]
    gradient_cols = list(gradient_cols)
    if not gradient_cols:
        raise KeyError("No gradient columns detected for batch plotting.")

    per_svgs, per_pngs = [], []

    for g in gradient_cols:
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", g).strip("_")
        file_prefix = f"{base_prefix}__{suffix}"
        svg, png, stats_csv, _stats_df = plot_signed_delta_vs_gradient_pkg(
            long_df=long_df,
            save_dir=save_dir,
            file_prefix=file_prefix,
            gradient_cols=[g],
            delta_kinds=delta_kinds,
            scatter=scatter, scatter_sample=scatter_sample,
            scatter_alpha=scatter_alpha, scatter_size=scatter_size,
            ci=ci,
            annotate_stats=annotate_stats,
            write_stats_csv=write_stats_csv,
            axes_label_fontsize=axes_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            legend_fontsize=legend_fontsize,
            height=height_single,
            aspect=aspect_single,
            facet_cols=1,
            title=None,
            # pass label knobs through
            label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol, grad_stat_tex=grad_stat_tex,
        )
        per_svgs.append(svg); per_pngs.append(png)

    svg_c, png_c, stats_csv_c, _ = plot_signed_delta_vs_gradient_pkg(
        long_df=long_df,
        save_dir=save_dir,
        file_prefix=f"{base_prefix}__combined",
        gradient_cols=gradient_cols,
        delta_kinds=delta_kinds,
        scatter=scatter, scatter_sample=scatter_sample,
        scatter_alpha=scatter_alpha, scatter_size=scatter_size,
        ci=ci,
        annotate_stats=annotate_stats,
        write_stats_csv=write_stats_csv,
        axes_label_fontsize=axes_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        height=height_combined,
        aspect=aspect_combined,
        facet_cols=facet_cols_combined,
        title=None,
        # pass label knobs through
        label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol, grad_stat_tex=grad_stat_tex,
    )
    return per_svgs + [svg_c], per_pngs + [png_c], stats_csv_c



def production_plot_path1_threshold_qa_summary(
    df: pd.DataFrame,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    qa_class_col: str = "qa_class",
    p_pass_col: str = "p_pass",
    n_trials_col: str = "n_trials",
    figsize=(8, 8),
    dpi: int = 300,
    seaborn_style: str = "whitegrid",
    seaborn_context: str = "paper",
    font_scale: float = 1.2,
    metric_order=None,
    qa_class_order=None,
    save_dir: Path | str | None = None,
    file_name: str = "Fig_path1_threshold_qa_summary.png",
    fig=None,
    axes=None,
    prob_pass_low_cut: float = 0.05,
    prob_pass_high_cut: float = 0.95,
    show_title: bool = True,
    annotate_percents: bool = True,
    percent_fmt: str = "{:.0f}%",
    min_count_inside: int = 2,
    min_frac_inside: float = 0.08,
    # NOTE: keeping the name, but now means "legend above and wide"
    legend_outside: bool = True,
):
    """
    Path-1 QA overview figure.
    - DVH metric labels use mathtext (e.g., D_{2%}, V_{150%})
    - stacked bar order can put Confident pass on top
    - legend can be wide + above the plot, with QA definitions
    """
    plt.ioff()
    required_cols = {metric_col, threshold_col, qa_class_col, p_pass_col, n_trials_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    import matplotlib as mpl
    from matplotlib.patches import Patch

    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
        }
    ):
        df_plot = df.copy()

        def _norm_qa(c):
            c_str = str(c).strip().lower()
            if "confident" in c_str and "pass" in c_str:
                return "Confident pass"
            if "confident" in c_str and "fail" in c_str:
                return "Confident fail"
            if "border" in c_str:
                return "Borderline"
            return str(c)

        df_plot["_qa_class_plot"] = df_plot[qa_class_col].map(_norm_qa)

        # --- DVH math label helpers ---
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            return f"{int(thr)}" if isinstance(thr, (int, np.integer)) else f"{thr:g}"

        def _make_criterion(row) -> str:
            m = row[metric_col]
            t = _thr_str(row[threshold_col])
            mm = _metric_math(m)
            if _metric_is_percent(m):
                return rf"{mm}$\ \geq\ {t}\%$"
            else:
                return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        df_plot["criterion_pretty"] = df_plot.apply(_make_criterion, axis=1)

        if metric_order is None:
            metric_order = ["D_2% (Gy)", "D_50% (Gy)", "D_98% (Gy)", "V_150% (%)"]
            metric_order = [m for m in metric_order if m in df_plot[metric_col].unique()]

        criterion_order: list[str] = []
        for m in metric_order:
            sub = df_plot[df_plot[metric_col] == m]
            if not sub.empty:
                criterion_order.append(_make_criterion(sub.iloc[0]))
        if not criterion_order:
            criterion_order = list(df_plot["criterion_pretty"].unique())

        if qa_class_order is None:
            # keep this order for legend ordering (pass first)
            qa_class_order = ["Confident pass", "Borderline", "Confident fail"]
            qa_class_order = [c for c in qa_class_order if c in df_plot["_qa_class_plot"].unique()]
            for c in sorted(df_plot["_qa_class_plot"].unique()):
                if c not in qa_class_order:
                    qa_class_order.append(c)

        # IMPORTANT: stack order (bottom -> top). We want Confident pass on TOP.
        # So draw it last.
        stack_order = [c for c in ["Confident fail", "Borderline", "Confident pass"] if c in qa_class_order]
        # include any unexpected classes at the bottom by default
        for c in qa_class_order:
            if c not in stack_order:
                stack_order.insert(0, c)

        qa_palette = {
            "Confident pass": "#66c2a5",
            "Borderline": "#fc8d62",
            "Confident fail": "#8da0cb",
        }

        # legend labels with definitions
        hi = float(prob_pass_high_cut)
        lo = float(prob_pass_low_cut)
        legend_label = {
            "Confident pass": rf"Confident pass ($p_{{b,r}}\geq {hi:.2f}$)",
            "Borderline":     rf"Borderline ($ {lo:.2f}<p_{{b,r}}<{hi:.2f}$)",
            "Confident fail": rf"Confident fail ($p_{{b,r}}\leq {lo:.2f}$)",
        }

        with sns.axes_style(seaborn_style), sns.plotting_context(seaborn_context, font_scale=font_scale):
            if fig is None or axes is None:
                fig, axes = plt.subplots(
                    2, 1, figsize=figsize, dpi=dpi, sharex=False,
                    gridspec_kw={"height_ratios": [1, 1.2]},
                )
            ax_bar, ax_box = axes

            counts = (
                df_plot.groupby(["criterion_pretty", "_qa_class_plot"])
                .size()
                .unstack("_qa_class_plot")
                .reindex(index=criterion_order)
                .fillna(0)
            )

            # ensure all columns exist
            for c in qa_class_order:
                if c not in counts.columns:
                    counts[c] = 0
            counts = counts[qa_class_order]
            totals = counts.sum(axis=1).to_numpy()

            default_palette = sns.color_palette("Set2", n_colors=len(qa_class_order))
            class_colors = {cls: qa_palette.get(cls, default_palette[i]) for i, cls in enumerate(qa_class_order)}

            bottoms = np.zeros(len(counts))
            x = np.arange(len(counts.index))

            # ---- stacked bar (using stack_order) ----
            for cls in stack_order:
                vals = counts[cls].values
                bars = ax_bar.bar(
                    x, vals, bottom=bottoms,
                    color=class_colors.get(cls, "0.7"),
                    edgecolor="black",
                    linewidth=0.8,
                    label=cls,  # (we will build our own legend handles)
                )

                if annotate_percents:
                    for xi, (rect, v, btm, tot) in enumerate(zip(bars, vals, bottoms, totals)):
                        if tot <= 0 or v <= 0:
                            continue
                        pct = 100.0 * float(v) / float(tot)
                        txt = percent_fmt.format(pct) + f", {v:.0f}"

                        y_mid = float(btm) + float(v) / 2.0
                        x_mid = rect.get_x() + rect.get_width() / 2.0
                        inside_ok = (int(v) >= int(min_count_inside)) and ((v / tot) >= float(min_frac_inside))

                        if inside_ok:
                            ax_bar.text(
                                x_mid, y_mid, txt,
                                ha="center", va="center",
                                fontsize="small", color="black", zorder=20
                            )
                        else:
                            # spread outside labels a bit
                            is_bottom = (cls == stack_order[0])
                            is_top = (cls == stack_order[-1])
                            dx = (-0.18 if is_bottom else (0.18 if is_top else 0.0))
                            dy = 0.6 + 0.25 * (xi % 2)
                            ax_bar.annotate(
                                txt,
                                xy=(x_mid, y_mid),
                                xytext=(x_mid + dx, float(btm) + float(v) + dy),
                                ha="center", va="bottom",
                                fontsize="small",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor="white",
                                    edgecolor="black",
                                    alpha=0.9,
                                ),
                                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
                                zorder=30,
                            )

                bottoms = bottoms + vals

            ax_bar.set_ylabel("Number of biopsies")

            # multi-line labels: criterion on line 1, n_b on line 2
            xticklabels = [
                f"{crit}\n$n_b={int(t)}$"
                for crit, t in zip(counts.index, totals)
            ]

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(xticklabels, rotation=0)



            if show_title:
                ax_bar.set_title("Path-1 per-biopsy threshold QA classification", fontsize="medium")

            ymax = max(1.0, float(totals.max()) * 1.15)
            ax_bar.set_ylim(0, ymax)

            # ---- Legend: wide + above ----
            # Build handles in desired legend order (pass, borderline, fail)
            handles = []
            labels = []
            for cls in ["Confident pass", "Borderline", "Confident fail"]:
                if cls in qa_class_order:
                    handles.append(Patch(facecolor=class_colors.get(cls, "0.7"), edgecolor="black"))
                    labels.append(legend_label.get(cls, cls))
            # add any unexpected classes
            for cls in qa_class_order:
                if cls not in ("Confident pass", "Borderline", "Confident fail"):
                    handles.append(Patch(facecolor=class_colors.get(cls, "0.7"), edgecolor="black"))
                    labels.append(cls)

            if legend_outside and handles:
                leg = fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.99),   # <-- was 1.02
                    ncol=min(len(labels), 3),
                    frameon=True,
                    framealpha=0.95,
                    fontsize="small",
                    borderaxespad=0.2,
                )

                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("black")

            # ---- Bottom panel: box + jitter ----
            df_plot["criterion_pretty"] = pd.Categorical(
                df_plot["criterion_pretty"], categories=criterion_order, ordered=True
            )

            sns.boxplot(
                data=df_plot, x="criterion_pretty", y=p_pass_col, ax=ax_box,
                whis=(5, 95), width=0.6, fliersize=0,
                color="white", linewidth=0.8,
            )

            sns.stripplot(
                data=df_plot, x="criterion_pretty", y=p_pass_col,
                hue="_qa_class_plot",
                hue_order=[c for c in ["Confident pass", "Borderline", "Confident fail"] if c in qa_class_order],
                dodge=False, jitter=True, size=4, alpha=0.8,
                ax=ax_box, palette=class_colors, zorder=10,
            )

            ax_box.axhline(prob_pass_high_cut, linestyle="--", linewidth=0.8, color="black")
            ax_box.axhline(prob_pass_low_cut, linestyle="--", linewidth=0.8, color="black")

            ax_box.set_ylabel(r"$p_{b,r}$ (Monte Carlo pass probability)")
            ax_box.set_xlabel("Threshold criterion")
            ax_box.set_ylim(-0.05, 1.05)
            ax_box.tick_params(axis="x", labelrotation=0)

            # Use the same multi-line labels (criterion + n_b) on the boxplot
            ax_box.set_xticks(np.arange(len(xticklabels)))
            ax_box.set_xticklabels(xticklabels, rotation=0)



            if ax_box.get_legend() is not None:
                ax_box.get_legend().remove()

            ax_bar.text(-0.05, 1.02, "A", transform=ax_bar.transAxes,
                        fontsize="large", fontweight="bold", va="bottom")
            ax_box.text(-0.05, 1.02, "B", transform=ax_box.transAxes,
                        fontsize="large", fontweight="bold", va="bottom")

            # layout: leave room for top legend
            if legend_outside:
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
            else:
                fig.tight_layout()

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / file_name, dpi=dpi, bbox_inches="tight")

        return fig, (ax_bar, ax_box)



def production_plot_path1_threshold_qa_summary_v2(
    df: pd.DataFrame,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    qa_class_col: str = "qa_class",
    p_pass_col: str = "p_pass",
    n_trials_col: str = "n_trials",
    figsize=(8, 8),
    dpi: int = 300,
    seaborn_style: str = "ticks",   # STYLE: slightly more minimal than whitegrid
    seaborn_context: str = "paper",
    font_scale: float = 1.2,
    metric_order=None,
    qa_class_order=None,
    save_dir: Path | str | None = None,
    file_name: str = "Fig_path1_threshold_qa_summary.png",
    fig=None,
    axes=None,
    prob_pass_low_cut: float = 0.05,
    prob_pass_high_cut: float = 0.95,
    show_title: bool = True,
    annotate_percents: bool = True,
    percent_fmt: str = "{:.0f}%",
    min_count_inside: int = 2,
    min_frac_inside: float = 0.08,
    legend_outside: bool = True,
):
    """
    Path-1 QA overview figure with a more minimalist, paper-style aesthetic.
    """
    plt.ioff()
    required_cols = {metric_col, threshold_col, qa_class_col, p_pass_col, n_trials_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    import matplotlib as mpl
    from matplotlib.patches import Patch

    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
        }
    ):
        df_plot = df.copy()

        def _norm_qa(c):
            c_str = str(c).strip().lower()
            if "confident" in c_str and "pass" in c_str:
                return "Confident pass"
            if "confident" in c_str and "fail" in c_str:
                return "Confident fail"
            if "border" in c_str:
                return "Borderline"
            return str(c)

        df_plot["_qa_class_plot"] = df_plot[qa_class_col].map(_norm_qa)

        # --- DVH math label helpers ---
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            return f"{int(thr)}" if isinstance(thr, (int, np.integer)) else f"{thr:g}"

        def _make_criterion(row) -> str:
            m = row[metric_col]
            t = _thr_str(row[threshold_col])
            mm = _metric_math(m)
            if _metric_is_percent(m):
                return rf"{mm}$\ \geq\ {t}\%$"
            else:
                return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        df_plot["criterion_pretty"] = df_plot.apply(_make_criterion, axis=1)

        if metric_order is None:
            metric_order = ["D_2% (Gy)", "D_50% (Gy)", "D_98% (Gy)", "V_150% (%)"]
            metric_order = [m for m in metric_order if m in df_plot[metric_col].unique()]

        criterion_order: list[str] = []
        for m in metric_order:
            sub = df_plot[df_plot[metric_col] == m]
            if not sub.empty:
                criterion_order.append(_make_criterion(sub.iloc[0]))
        if not criterion_order:
            criterion_order = list(df_plot["criterion_pretty"].unique())

        if qa_class_order is None:
            qa_class_order = ["Confident pass", "Borderline", "Confident fail"]
            qa_class_order = [c for c in qa_class_order if c in df_plot["_qa_class_plot"].unique()]
            for c in sorted(df_plot["_qa_class_plot"].unique()):
                if c not in qa_class_order:
                    qa_class_order.append(c)

        # bottom -> top
        stack_order = [c for c in ["Confident fail", "Borderline", "Confident pass"] if c in qa_class_order]
        for c in qa_class_order:
            if c not in stack_order:
                stack_order.insert(0, c)

        # STYLE: slightly more muted palette (still 3 distinct hues)
        qa_palette = {
            "Confident pass": "#4f9685",   # muted teal
            "Borderline":     "#d29168",   # muted orange
            "Confident fail": "#6a78a5",   # muted blue
        }

        hi = float(prob_pass_high_cut)
        lo = float(prob_pass_low_cut)
        legend_label = {
            "Confident pass": rf"Confident pass ($p_{{b,r}}\geq {hi:.2f}$)",
            "Borderline":     rf"Borderline ($ {lo:.2f}<p_{{b,r}}<{hi:.2f}$)",
            "Confident fail": rf"Confident fail ($p_{{b,r}}\leq {lo:.2f}$)",
        }

        with sns.axes_style(seaborn_style), sns.plotting_context(seaborn_context, font_scale=font_scale):
            if fig is None or axes is None:
                fig, axes = plt.subplots(
                    2, 1, figsize=figsize, dpi=dpi, sharex=False,
                    gridspec_kw={"height_ratios": [1, 1.2]},
                )
            ax_bar, ax_box = axes

            counts = (
                df_plot.groupby(["criterion_pretty", "_qa_class_plot"])
                .size()
                .unstack("_qa_class_plot")
                .reindex(index=criterion_order)
                .fillna(0)
            )

            for c in qa_class_order:
                if c not in counts.columns:
                    counts[c] = 0
            counts = counts[qa_class_order]
            totals = counts.sum(axis=1).to_numpy()

            default_palette = sns.color_palette("Set2", n_colors=len(qa_class_order))
            class_colors = {cls: qa_palette.get(cls, default_palette[i]) for i, cls in enumerate(qa_class_order)}

            bottoms = np.zeros(len(counts))
            x = np.arange(len(counts.index))

            # STYLE: slightly narrower bars, light edges
            bar_width = 0.6

            # subtle grid
            ax_bar.set_axisbelow(True)
            #ax_bar.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8")
            ax_bar.xaxis.grid(False)

            # ---- stacked bar ----
            for cls in stack_order:
                vals = counts[cls].values
                bars = ax_bar.bar(
                    x,
                    vals,
                    width=bar_width,
                    bottom=bottoms,
                    color=class_colors.get(cls, "0.7"),
                    edgecolor="0.7",         # STYLE: light grey, not black
                    linewidth=0.5,
                    label=cls,
                )

                if annotate_percents:
                    for xi, (rect, v, btm, tot) in enumerate(zip(bars, vals, bottoms, totals)):
                        if tot <= 0 or v <= 0:
                            continue
                        pct = 100.0 * float(v) / float(tot)
                        txt = percent_fmt.format(pct) + f", {v:.0f}"

                        y_mid = float(btm) + float(v) / 2.0
                        x_mid = rect.get_x() + rect.get_width() / 2.0
                        inside_ok = (int(v) >= int(min_count_inside)) and ((v / tot) >= float(min_frac_inside))

                        if inside_ok:
                            ax_bar.text(
                                x_mid,
                                y_mid,
                                txt,
                                ha="center",
                                va="center",
                                fontsize="x-small",   # STYLE
                                color="black",
                                zorder=20,
                            )
                        else:
                            # STYLE: lighter, cleaner annotation box
                            is_bottom = (cls == stack_order[0])
                            is_top = (cls == stack_order[-1])
                            dx = (-0.18 if is_bottom else (0.18 if is_top else 0.0))
                            dy = 0.4 + 0.2 * (xi % 2)
                            ax_bar.annotate(
                                txt,
                                xy=(x_mid, y_mid),
                                xytext=(x_mid + dx, float(btm) + float(v) + dy),
                                ha="center",
                                va="bottom",
                                fontsize="x-small",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor="white",
                                    edgecolor="0.7",
                                    linewidth=0.5,
                                    alpha=0.9,
                                ),
                                arrowprops=dict(
                                    arrowstyle="-",
                                    lw=0.5,
                                    color="0.5",
                                ),
                                zorder=30,
                            )

                bottoms = bottoms + vals

            ax_bar.set_ylabel("Number of biopsies")

            xticklabels = [
                f"{crit}\n$n_b={int(t)}$"
                for crit, t in zip(counts.index, totals)
            ]

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(xticklabels, rotation=0)

            # STYLE: slightly smaller title and less bold
            if show_title:
                ax_bar.set_title(
                    "Per-biopsy threshold QA classification",
                    fontsize="medium",
                    fontweight="normal",
                )

            ymax = max(1.0, float(totals.max()) * 1.15)
            ax_bar.set_ylim(0, ymax)

            # clean spines
            for spine in ["top", "right"]:
                ax_bar.spines[spine].set_visible(False)

            # ---- Legend ----
            handles = []
            labels = []
            for cls in ["Confident pass", "Borderline", "Confident fail"]:
                if cls in qa_class_order:
                    handles.append(Patch(facecolor=class_colors.get(cls, "0.7"), edgecolor="0.5", linewidth=0.5))
                    labels.append(legend_label.get(cls, cls))
            for cls in qa_class_order:
                if cls not in ("Confident pass", "Borderline", "Confident fail"):
                    handles.append(Patch(facecolor=class_colors.get(cls, "0.7"), edgecolor="0.5", linewidth=0.5))
                    labels.append(cls)

            if legend_outside and handles:
                leg = fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.99),
                    ncol=min(len(labels), 3),
                    frameon=False,          # STYLE: frameless legend
                    fontsize="small",
                    borderaxespad=0.2,
                )

            # STYLE: subtle grid
            ax_bar.yaxis.grid(False)


            # ---- Bottom panel: box + jitter ----
            df_plot["criterion_pretty"] = pd.Categorical(
                df_plot["criterion_pretty"], categories=criterion_order, ordered=True
            )

            sns.boxplot(
                data=df_plot,
                x="criterion_pretty",
                y=p_pass_col,
                ax=ax_box,
                whis=(5, 95),
                width=0.6,
                fliersize=0,
                color="white",
                linewidth=0.8,
            )

            sns.stripplot(
                data=df_plot,
                x="criterion_pretty",
                y=p_pass_col,
                hue="_qa_class_plot",
                hue_order=[c for c in ["Confident pass", "Borderline", "Confident fail"] if c in qa_class_order],
                dodge=False,
                jitter=True,
                size=4,
                alpha=0.8,
                ax=ax_box,
                palette=class_colors,
                zorder=10,
            )

            ax_box.axhline(prob_pass_high_cut, linestyle="--", linewidth=0.8, color="black")
            ax_box.axhline(prob_pass_low_cut, linestyle="--", linewidth=0.8, color="black")

            ax_box.set_ylabel(r"$p_{b,r}$ (Monte Carlo pass probability)")
            ax_box.set_xlabel("Threshold criterion")
            ax_box.set_ylim(-0.05, 1.05)
            ax_box.tick_params(axis="x", labelrotation=0)

            ax_box.set_xticks(np.arange(len(xticklabels)))
            ax_box.set_xticklabels(xticklabels, rotation=0)

            # keep, but make subtle on box panel
            ax_box.set_axisbelow(True)
            ax_box.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.9")

            if ax_box.get_legend() is not None:
                ax_box.get_legend().remove()

            for spine in ["top", "right"]:
                ax_box.spines[spine].set_visible(False)

            ax_bar.text(-0.05, 1.02, "A", transform=ax_bar.transAxes,
                        fontsize="large", fontweight="bold", va="bottom")
            ax_box.text(-0.05, 1.02, "B", transform=ax_box.transAxes,
                        fontsize="large", fontweight="bold", va="bottom")

            if legend_outside:
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
            else:
                fig.tight_layout()

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / file_name, dpi=dpi, bbox_inches="tight")

        return fig, (ax_bar, ax_box)


def production_plot_path1_p_pass_vs_margin_by_metric(
    df: pd.DataFrame,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    margin_col: str = "distance_from_threshold_nominal",
    p_pass_col: str = "p_pass",
    n_trials_col: str = "n_trials",
    qa_class_col: str = "qa_class",
    figsize=(10, 8),
    dpi: int = 300,
    seaborn_style: str = "whitegrid",
    seaborn_context: str = "paper",
    font_scale: float = 1.1,
    metric_order=None,
    save_dir: Path | str | None = None,
    file_name: str = "Fig_path1_p_pass_vs_margin_by_metric.png",
    add_logistic_fit: bool = True,
    logistic_min_points: int = 6,
    fig=None,
    axes=None,
    prob_pass_low_cut: float = 0.05,
    prob_pass_high_cut: float = 0.95,
    show_panel_titles: bool = True,
    include_rule_in_xlabel: bool = True,   # <--- NEW
    annotate_fit_stats: bool = False,
    # supports: "n", "mcfadden_r2", "aic", "wrmse", "brier"
    fit_stats: tuple[str, ...] = ("n", "mcfadden_r2"),
    fit_stats_loc: tuple[float, float] = (0.70, 0.45),  # fallback if auto fails
    show_required_margin_line: bool = False,
    required_prob: float = 0.95,
    required_line_kwargs: dict | None = None,
    # NEW: precomputed logistic coefficients / scores
    coef_df: pd.DataFrame | None = None,
    coef_metric_col: str | None = None,
    coef_threshold_col: str | None = None,
):
    """
    Path-1 QA figure: p_{b,r} versus nominal margin δ_{b,r} per metric.

    If `coef_df` is provided, the logistic curve, GOF metrics, and required
    margin δ̂_p are all read from that table (no re-fitting in this function).

    Otherwise, the function falls back to fitting a 1D logistic model
    internally (previous behaviour).
    """
    plt.ioff()
    required_cols = {metric_col, threshold_col, margin_col, p_pass_col, n_trials_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    import matplotlib as mpl
    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
        }
    ):
        df_plot = df.copy()

        def _norm_qa(c):
            c_str = str(c).strip().lower()
            if "confident" in c_str and "pass" in c_str:
                return "Confident pass"
            if "confident" in c_str and "fail" in c_str:
                return "Confident fail"
            if "border" in c_str:
                return "Borderline"
            return str(c)

        if qa_class_col in df_plot.columns:
            df_plot["_qa_class_plot"] = df_plot[qa_class_col].map(_norm_qa)
        else:
            df_plot["_qa_class_plot"] = "NA"

        # --- DVH label / axis helpers ---
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            return f"{int(thr)}" if isinstance(thr, (int, np.integer)) else f"{thr:g}"

        def _make_criterion(row) -> str:
            m = row[metric_col]
            t = _thr_str(row[threshold_col])
            mm = _metric_math(m)
            if _metric_is_percent(m):
                return rf"{mm}$\ \geq\ {t}\%$"
            return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        """
        def _make_x_label(row) -> str:
            m = row[metric_col]
            t = _thr_str(row[threshold_col])
            mm = _metric_math(m)
            if _metric_is_percent(m):
                return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t}%)"
            return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t} Gy)"
        """

        def _make_x_label(row, rule_index: int, include_rule: bool = True) -> str:
            """
            Build x-axis label for a given metric/threshold row.

            Example (with include_rule=True, rule_index=1):
            "Nominal margin δ_{b,1} (r = 1 : D_{2%} ≥ 32 Gy)"
            """
            # main symbol δ_{b,1}
            rule_sym = rf"\delta_{{b,{rule_index}}}"
            label = rf"Nominal margin ${rule_sym}$"

            if include_rule:
                # existing criterion, e.g. "$D_{2\%}$\ \geq\ 32\,\mathrm{Gy}"
                crit = _make_criterion(row)

                # strip outer $ so we can embed it into one math block
                crit_inner = crit.strip("$")

                # one math environment: r = 1 : <criterion>
                label += rf" ($r = {rule_index}: {crit_inner}$)"

            return label



        def _x_units_suffix(metric_name: str) -> str:
            return "%" if _metric_is_percent(metric_name) else " Gy"

        if metric_order is None:
            metric_order = ["D_2% (Gy)", "D_50% (Gy)", "D_98% (Gy)", "V_150% (%)"]
            metric_order = [m for m in metric_order if m in df_plot[metric_col].unique()]

        n_metrics = len(metric_order)
        nrows = 2 if n_metrics > 2 else 1
        ncols = 2 if n_metrics > 1 else 1

        qa_palette = {
            "Confident pass": "#66c2a5",
            "Borderline": "#fc8d62",
            "Confident fail": "#8da0cb",
        }

        # --- optional statsmodels fallback (only if coef_df is None) ---
        try:
            import statsmodels.api as sm  # type: ignore
        except Exception:
            sm = None
            add_logistic_fit_local = coef_df is not None
        else:
            add_logistic_fit_local = add_logistic_fit or (coef_df is not None)

        def _fit_logit_1d(sub: pd.DataFrame):
            if coef_df is not None:
                return None, False
            if sm is None or len(sub) < logistic_min_points:
                return None, False
            try:
                X = sub[[margin_col]].to_numpy()
                y = np.clip(sub[p_pass_col].to_numpy(), 1e-4, 1 - 1e-4)
                w = sub[n_trials_col].to_numpy()
                X_design = sm.add_constant(X, has_constant="add")
                model = sm.GLM(y, X_design, family=sm.families.Binomial(), freq_weights=w)
                res = model.fit()

                X0 = np.ones((len(y), 1))
                model0 = sm.GLM(y, X0, family=sm.families.Binomial(), freq_weights=w)
                res0 = model0.fit()
                return (res, res0), True
            except Exception:
                return None, False

        def _required_margin_from_fit(res, p: float) -> float | None:
            try:
                p = float(p)
                if not (0.0 < p < 1.0):
                    return None
                b0 = float(res.params[0])
                b1 = float(res.params[1])
                if abs(b1) < 1e-12:
                    return None
                logit = np.log(p / (1.0 - p))
                return (logit - b0) / b1
            except Exception:
                return None

        def _weighted_scores_from_fit(sub: pd.DataFrame, res):
            try:
                x = sub[margin_col].to_numpy(dtype=float)
                y = sub[p_pass_col].to_numpy(dtype=float)
                w = sub[n_trials_col].to_numpy(dtype=float)

                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
                x, y, w = x[ok], y[ok], w[ok]
                if x.size == 0:
                    return None, None

                Xd = sm.add_constant(x, has_constant="add")
                yhat = np.asarray(res.predict(Xd), dtype=float)

                e2 = (y - yhat) ** 2
                wsum = float(np.sum(w))
                if wsum <= 0:
                    return None, None

                wbrier = float(np.sum(w * e2) / wsum)
                wrmse = float(np.sqrt(wbrier))
                return wrmse, wbrier
            except Exception:
                return None, None

        # --- annotation text builders ---
        def _format_panel_box_from_coef(row_coef: pd.Series, metric_name: str, x_req: float | None) -> str:
            parts: list[str] = []
            for k in fit_stats:
                kk = k.lower()

                if kk in ("n", "n_pts", "npoints", "n_biopsies"):
                    n_b = row_coef.get("n_biopsies", np.nan)
                    if np.isfinite(n_b):
                        parts.append(f"N={int(n_b):d}")

                elif kk == "aic":
                    aic = row_coef.get("aic", np.nan)
                    if np.isfinite(aic):
                        parts.append(f"AIC={aic:.1f}")

                elif kk in ("mcfadden_r2", "mcfadden", "pseudo_r2", "r2"):
                    r2 = row_coef.get("mcfadden_r2", np.nan)
                    if np.isfinite(r2):
                        parts.append(rf"$R^2_{{McF}}$={r2:.2f}")

                elif kk in ("wrmse", "w_rmse", "rmse_w"):
                    wrmse = row_coef.get("rmse_prob_w", np.nan)
                    if np.isfinite(wrmse):
                        parts.append("wRMSE=" + f"{wrmse:.2f}")

                elif kk in ("brier", "wbrier", "w_brier"):
                    wbrier = row_coef.get("brier_w", np.nan)
                    if np.isfinite(wbrier):
                        parts.append("wBrier=" + f"{wbrier:.2f}")

            if show_required_margin_line and (x_req is not None) and np.isfinite(x_req):
                suffix = _x_units_suffix(metric_name)
                parts.append(rf"$\hat{{\delta}}_{{{required_prob:.2f}}}$={x_req:.1f}{suffix}")

            return "\n".join(parts)

        def _format_panel_box_from_fit(
            *, sub: pd.DataFrame, res, res0, metric_name: str, x_req: float | None
        ) -> str:
            parts: list[str] = []

            for k in fit_stats:
                kk = k.lower()

                if kk in ("n", "n_pts", "npoints"):
                    parts.append(f"N={len(sub):d}")

                elif kk == "aic":
                    try:
                        parts.append(f"AIC={res.aic:.1f}")
                    except Exception:
                        pass

                elif kk in ("mcfadden_r2", "mcfadden", "pseudo_r2", "r2"):
                    try:
                        llf = float(res.llf)
                        llnull = float(res0.llf)
                        r2 = np.nan if llnull == 0 else (1.0 - (llf / llnull))
                        parts.append(rf"$R^2_{{McF}}$={r2:.2f}")
                    except Exception:
                        pass

                elif kk in ("wrmse", "w_rmse", "rmse_w"):
                    wrmse, _ = _weighted_scores_from_fit(sub, res)
                    if wrmse is not None and np.isfinite(wrmse):
                        parts.append("wRMSE=" + f"{wrmse:.2f}")

                elif kk in ("brier", "wbrier", "w_brier"):
                    _, wbrier = _weighted_scores_from_fit(sub, res)
                    if wbrier is not None and np.isfinite(wbrier):
                        parts.append("wBrier=" + f"{wbrier:.2f}")

            if show_required_margin_line and (x_req is not None) and np.isfinite(x_req):
                suffix = _x_units_suffix(metric_name)
                parts.append(rf"$\hat{{\delta}}_{{{required_prob:.2f}}}$={x_req:.1f}{suffix}")

            return "\n".join(parts)

        def _choose_fitbox_loc(
            xvals: np.ndarray,
            yvals: np.ndarray,
            txt: str,
            avoid_x: Sequence[float] | None = None,
        ) -> tuple[float, float]:
            try:
                xvals = np.asarray(xvals, float)
                yvals = np.asarray(yvals, float)
                ok = np.isfinite(xvals) & np.isfinite(yvals)
                xvals, yvals = xvals[ok], yvals[ok]
                if xvals.size == 0:
                    return fit_stats_loc

                xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
                if xmax <= xmin:
                    return fit_stats_loc

                # fixed y range for normalization
                ymin, ymax = -0.05, 1.05

                xN = (xvals - xmin) / (xmax - xmin)
                yN = (yvals - ymin) / (ymax - ymin)
                yN = np.clip(yN, 0.0, 1.0)

                # NEW: penalize specific vertical lines (e.g. required margin)
                if avoid_x:
                    avoid_arr = np.array(
                        [ax for ax in avoid_x if np.isfinite(ax)], dtype=float
                    )
                    if avoid_arr.size > 0:
                        # keep only within data x-span, otherwise ignore
                        avoid_arr = avoid_arr[(avoid_arr >= xmin) & (avoid_arr <= xmax)]
                        if avoid_arr.size > 0:
                            avoid_xN = (avoid_arr - xmin) / (xmax - xmin)
                            # tile across y to simulate a vertical "wall" of points
                            n_y_fake = 25
                            fake_yN = np.linspace(0.0, 1.0, n_y_fake)
                            fake_xN = np.repeat(avoid_xN, n_y_fake)
                            fake_yN = np.tile(fake_yN, avoid_arr.size)

                            xN = np.concatenate([xN, fake_xN])
                            yN = np.concatenate([yN, fake_yN])

                n_lines = txt.count("\n") + 1
                box_w = 0.34
                box_h = min(0.12 + 0.05 * n_lines, 0.42)

                candidates = [
                    (0.68, 0.45),
                    (0.68, 0.08),
                    (0.68, 0.70),
                    (0.02, 0.70),
                    (0.02, 0.45),
                    (0.02, 0.08),
                ]

                best = None
                best_score = 1e18
                for xa, ya in candidates:
                    xb = min(xa + box_w, 0.99)
                    yb = min(ya + box_h, 0.99)
                    inside = (xN >= xa) & (xN <= xb) & (yN >= ya) & (yN <= yb)
                    score = float(np.sum(inside))
                    tie = -xa
                    if (score < best_score) or (
                        score == best_score and (best is None or tie < -best[0])
                    ):
                        best = (xa, ya)
                        best_score = score

                return best if best is not None else fit_stats_loc
            except Exception:
                return fit_stats_loc


        # map coef_df column names for metric / threshold if supplied
        if coef_df is not None:
            coef_df = coef_df.copy()
            if coef_metric_col is None:
                coef_metric_col = metric_col
            if coef_threshold_col is None:
                coef_threshold_col = threshold_col

        with sns.axes_style(seaborn_style), sns.plotting_context(seaborn_context, font_scale=font_scale):
            if fig is None or axes is None:
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=figsize, dpi=dpi,
                    sharex=False, sharey=True,
                )

            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

            qa_classes = sorted(df_plot["_qa_class_plot"].unique())
            if len(qa_classes) == 1 and qa_classes[0] == "NA":
                qa_classes = []
                class_colors = {}
            else:
                preferred = ["Confident pass", "Borderline", "Confident fail"]
                qa_classes = [c for c in preferred if c in qa_classes] + [
                    c for c in qa_classes if c not in preferred
                ]
                default_palette = sns.color_palette("Set2", n_colors=len(qa_classes))
                class_colors = {
                    cls: qa_palette.get(cls, default_palette[i]) for i, cls in enumerate(qa_classes)
                }

            letters = "ABCD"
            for i, metric_name in enumerate(metric_order):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]

                sub = df_plot[df_plot[metric_col] == metric_name].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                # scatter
                if qa_classes:
                    for cls in ["Confident pass", "Borderline", "Confident fail"]:
                        sub_cls = sub[sub["_qa_class_plot"] == cls]
                        if sub_cls.empty:
                            continue
                        ax.scatter(
                            sub_cls[margin_col], sub_cls[p_pass_col],
                            label=cls,
                            s=45,
                            alpha=0.90 if cls != "Borderline" else 0.75,
                            edgecolor="black",
                            linewidth=0.5,
                            color=class_colors.get(cls, "0.6"),
                            zorder=5,
                        )
                else:
                    ax.scatter(
                        sub[margin_col], sub[p_pass_col],
                        s=45, alpha=0.8,
                        edgecolor="black", linewidth=0.5,
                        color="0.6", zorder=5
                    )

                ax.axhline(prob_pass_high_cut, linestyle="--", linewidth=0.9, color="gray", zorder=1)
                ax.axhline(prob_pass_low_cut,  linestyle="--", linewidth=0.9, color="gray", zorder=1)
                ax.axvline(0.0, linestyle=":", linewidth=0.9, color="black", zorder=1)

                x_req = None

                # --- logistic curve + annotation from precomputed coef_df (preferred) ---
                if coef_df is not None:
                    thr_val = sub[threshold_col].iloc[0]
                    mask = coef_df[coef_metric_col] == metric_name
                    if coef_threshold_col in coef_df.columns:
                        mask &= coef_df[coef_threshold_col] == thr_val
                    row_matches = coef_df[mask]
                    if not row_matches.empty:
                        row_coef = row_matches.iloc[0]

                        x_min, x_max = sub[margin_col].min(), sub[margin_col].max()
                        x_grid = np.linspace(x_min - 1.0, x_max + 1.0, 250)

                        b0 = float(row_coef.get("b_const", np.nan))
                        b1 = float(row_coef.get(f"b_{margin_col}", np.nan))
                        if np.isfinite(b0) and np.isfinite(b1):
                            eta = b0 + b1 * x_grid
                            y_pred = 1.0 / (1.0 + np.exp(-eta))
                            ax.plot(
                                x_grid, y_pred,
                                linewidth=1.6, color="black",
                                label="Logistic fit", zorder=10,
                            )

                        if show_required_margin_line:
                            col_req = f"margin_at_p{int(round(required_prob * 100))}"
                            if col_req in row_coef and np.isfinite(row_coef[col_req]):
                                x_req = float(row_coef[col_req])
                                kwargs = dict(color="black", linestyle="--", linewidth=1.2, alpha=0.9)
                                if required_line_kwargs:
                                    kwargs.update(required_line_kwargs)
                                ax.axvline(x_req, **kwargs)

                        if annotate_fit_stats or (show_required_margin_line and x_req is not None):
                            txt = _format_panel_box_from_coef(row_coef=row_coef,
                                                              metric_name=metric_name,
                                                              x_req=x_req)
                            if txt:
                                xa, ya = _choose_fitbox_loc(
                                    sub[margin_col].to_numpy(),
                                    sub[p_pass_col].to_numpy(),
                                    txt,
                                    avoid_x=[x_req] if x_req is not None else None,
                                )
                                ax.text(
                                    xa, ya, txt,
                                    transform=ax.transAxes,
                                    ha="left", va="bottom",
                                    fontsize="small",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        facecolor="white",
                                        edgecolor="black",
                                        alpha=0.9,
                                    ),
                                    zorder=20,
                                )

                # --- fallback: refit logistic inside plotting function (legacy mode) ---
                elif add_logistic_fit_local:
                    fit_pair, ok = _fit_logit_1d(sub)
                    if ok and fit_pair is not None:
                        res, res0 = fit_pair

                        x_grid = np.linspace(sub[margin_col].min() - 1.0,
                                             sub[margin_col].max() + 1.0, 250)
                        X_grid = sm.add_constant(x_grid, has_constant="add")
                        y_pred = res.predict(X_grid)
                        ax.plot(
                            x_grid, y_pred,
                            linewidth=1.6, color="black",
                            label="Logistic fit", zorder=10,
                        )

                        if show_required_margin_line:
                            x_req = _required_margin_from_fit(res, required_prob)
                            if x_req is not None and np.isfinite(x_req):
                                kwargs = dict(color="black", linestyle="--",
                                              linewidth=1.2, alpha=0.9)
                                if required_line_kwargs:
                                    kwargs.update(required_line_kwargs)
                                ax.axvline(x_req, **kwargs)

                        if annotate_fit_stats or (show_required_margin_line and x_req is not None):
                            txt = _format_panel_box_from_fit(
                                sub=sub, res=res, res0=res0,
                                metric_name=metric_name, x_req=x_req,
                            )
                            if txt:
                                xa, ya = _choose_fitbox_loc(
                                    sub[margin_col].to_numpy(),
                                    sub[p_pass_col].to_numpy(),
                                    txt,
                                    avoid_x=[x_req] if x_req is not None else None,
                                )
                                ax.text(
                                    xa, ya, txt,
                                    transform=ax.transAxes,
                                    ha="left", va="bottom",
                                    fontsize="small",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        facecolor="white",
                                        edgecolor="black",
                                        alpha=0.9,
                                    ),
                                    zorder=20,
                                )

                if show_panel_titles:
                    ax.set_title(_make_criterion(sub.iloc[0]), fontsize="medium")

                # i goes 0,1,2,... so rule index is i+1
                ax.set_xlabel(
                    _make_x_label(
                        sub.iloc[0],
                        rule_index=i + 1,
                        include_rule=include_rule_in_xlabel,
                    )
                )

                ax.tick_params(axis="x", labelbottom=True)

                if i < len(letters):
                    ax.text(
                        0.02, 0.98, letters[i],
                        transform=ax.transAxes,
                        fontsize="large", fontweight="bold",
                        va="top", ha="left",
                    )

                if (i % ncols) == 0:
                    ax.set_ylabel(r"$p_{b,r}$ (Monte Carlo pass probability)")
                else:
                    ax.set_ylabel("")

                ax.set_ylim(-0.05, 1.05)

            # global legend
            handles, labels = [], []
            for ax in axes_flat:
                if ax.get_visible():
                    h, lab = ax.get_legend_handles_labels()
                    if h:
                        handles, labels = h, lab
                        break
            if handles:
                leg = fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=min(len(labels), 4),
                    frameon=True, framealpha=0.95,
                    fontsize="small",
                )
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("black")

            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / file_name, dpi=dpi, bbox_inches="tight")

        return fig, axes





def production_plot_path1_logit_margin_plus_grad_families(
    pred_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    label_col: str = "label",
    margin_col: str = "distance_from_threshold_nominal",
    grad_col: str = "nominal_core_mean_grad_gy_per_mm",
    p_pass_col: str = "p_pass",
    qa_class_col: str = "qa_class",
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 300,
    seaborn_style: str = "whitegrid",
    seaborn_context: str = "paper",
    font_scale: float = 1.1,
    metric_order: Sequence[str] | None = None,
    save_dir: Path | str | None = None,
    file_name: str = "Fig_Path1_logit_margin_plus_grad_families.png",
    n_grad_levels: int = 4,
    grad_levels: Sequence[float] | None = None,
    grad_quantiles: Sequence[float] | None = None,
    prob_pass_cutoffs: tuple[float, float] = (0.05, 0.95),
    n_margin_grid: int = 200,
    scatter_size: float = 45.0,
    scatter_alpha: float = 0.9,
    panel_letter_start: str = "A",
    # panel fit / comparison stats
    annotate_fit_stats: bool = False,
    fit_stats: Sequence[str] = (
        "n",
        "mcfadden_r2",
        "aic",
        "wrmse",
        "brier",
        "delta_aic",
        "delta_rmse",
        "lr_pvalue",
    ),
    fit_stats_loc: tuple[float, float] = (0.02, 0.05),
    comparison_df: pd.DataFrame | None = None,
    comparison_label_col: str = "label",
    # NEW: overlay margin-only (1D) model
    overlay_1d_model: bool = False,
    coef1_df: pd.DataFrame | None = None,
    coef1_label_col: str = "label",
    overlay_1d_kwargs: dict | None = None,
    
):
    """
    Path-1 QA figure: 2-predictor logit fit (margin + gradient) for each threshold.

    For each threshold label, we:
      * show scatter of (margin, p_{b,r}) coloured by QA class
      * overlay a family of logistic curves for several fixed nominal gradients
      * optionally overlay the 1D margin-only logistic curve
      * optionally annotate a small box with model / comparative fit metrics.

    Notes
    -----
    - All GOF / scoring metrics are for the SINGLE 2-predictor model per threshold.
      They do not depend on gradient slice.
    - If `comparison_df` is provided (1D vs 2D comparison), delta-metrics and the
      LR p-value can be included in the annotation box by listing them in fit_stats.
    - If `overlay_1d_model=True` and `coef1_df` is supplied, the margin-only fit
      (with coefficients from coef1_df) is drawn as a single line in each panel.
    """

    plt.ioff()

    required_pred = {metric_col, threshold_col, label_col, margin_col, grad_col, p_pass_col}
    missing_pred = required_pred - set(pred_df.columns)
    if missing_pred:
        raise ValueError(f"pred_df is missing required columns: {sorted(missing_pred)}")

    required_coef = {label_col, "b_const", f"b_{margin_col}", f"b_{grad_col}"}
    missing_coef = required_coef - set(coef_df.columns)
    if missing_coef:
        raise ValueError(f"coef_df is missing required columns: {sorted(missing_coef)}")

    import matplotlib as mpl
    from matplotlib.lines import Line2D

    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
        }
    ):
        df_plot = pred_df.copy()
        coef_plot = coef_df.copy()

        # --- QA class normalisation ---
        def _norm_qa(c):
            c_str = str(c).strip().lower()
            if "confident" in c_str and "pass" in c_str:
                return "Confident pass"
            if "confident" in c_str and "fail" in c_str:
                return "Confident fail"
            if "border" in c_str:
                return "Borderline"
            return str(c)

        if qa_class_col in df_plot.columns:
            df_plot["_qa_class_plot"] = df_plot[qa_class_col].map(_norm_qa)
        else:
            df_plot["_qa_class_plot"] = "NA"

        qa_class_order = ["Confident pass", "Borderline", "Confident fail"]
        qa_palette = {
            "Confident pass": "#66c2a5",
            "Borderline": "#fc8d62",
            "Confident fail": "#8da0cb",
            "NA": "0.6",
        }

        # --- DVH label helpers ---
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            if isinstance(thr, (int, np.integer)):
                return f"{int(thr)}"
            if isinstance(thr, float):
                return f"{thr:g}"
            try:
                return f"{float(thr):g}"
            except Exception:
                return str(thr)

        def _make_title(metric_name: str, thr) -> str:
            mm = _metric_math(metric_name)
            t = _thr_str(thr)
            if _metric_is_percent(metric_name):
                return rf"{mm}$\ \geq\ {t}\%$"
            return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        def _make_xlabel(metric_name: str, thr) -> str:
            """
            X-axis: nominal margin for this DVH metric vs its threshold.
            Keep math only for δ and the DVH metric; write units as plain text.
            """
            mm = _metric_math(metric_name)   # e.g. "$D_{2\\%}$"
            t = _thr_str(thr)
            if _metric_is_percent(metric_name):
                # '%' in normal text is fine because text.usetex=False
                return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t}%)"
            # dose threshold in Gy as plain text
            return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t} Gy)"


        def _choose_fitbox_loc(xvals: np.ndarray, yvals: np.ndarray, txt: str) -> tuple[float, float]:
            """Pick a box location (axes coords) that roughly avoids points."""
            try:
                xvals = np.asarray(xvals, float)
                yvals = np.asarray(yvals, float)
                m = np.isfinite(xvals) & np.isfinite(yvals)
                xvals = xvals[m]
                yvals = yvals[m]
                if xvals.size == 0 or yvals.size == 0:
                    return fit_stats_loc

                xmin, xmax = float(xvals.min()), float(xvals.max())
                ymin, ymax = -0.05, 1.05  # fixed y-limits in this figure
                if xmax <= xmin:
                    return fit_stats_loc

                xN = (xvals - xmin) / (xmax - xmin)
                yN = (yvals - ymin) / (ymax - ymin)
                yN = np.clip(yN, 0.0, 1.0)

                n_lines = txt.count("\n") + 1
                box_w = 0.32
                box_h = min(0.13 + 0.05 * n_lines, 0.40)

                candidates = [
                    (0.66, 0.55),
                    (0.66, 0.10),
                    (0.02, 0.55),
                    (0.02, 0.10),
                ]

                best = fit_stats_loc
                best_score = 1e9
                for xa, ya in candidates:
                    xb = min(xa + box_w, 0.99)
                    yb = min(ya + box_h, 0.99)
                    inside = (xN >= xa) & (xN <= xb) & (yN >= ya) & (yN <= yb)
                    score = int(inside.sum())
                    if score < best_score:
                        best_score = score
                        best = (xa, ya)
                return best
            except Exception:
                return fit_stats_loc

        def _format_panel_box_text(lbl: str) -> str:
            """Build text for the fit-metrics box for a given label."""
            parts: list[str] = []

            row2 = coef_plot.loc[coef_plot[label_col] == lbl]
            row2 = row2.iloc[0] if not row2.empty else None

            row_cmp = None
            if comparison_df is not None:
                sub = comparison_df.loc[comparison_df[comparison_label_col] == lbl]
                if not sub.empty:
                    row_cmp = sub.iloc[0]

            for key in fit_stats:
                kk = str(key).lower()
                if kk in ("n", "n_biopsies", "npts", "n_points"):
                    if row2 is not None and "n_biopsies" in row2:
                        try:
                            parts.append(f"N={int(row2['n_biopsies'])}")
                        except Exception:
                            pass
                elif kk == "aic":
                    if row2 is not None and "aic" in row2:
                        try:
                            parts.append(f"AIC={float(row2['aic']):.1f}")
                        except Exception:
                            pass
                elif kk in ("mcfadden_r2", "mcfadden", "pseudo_r2", "r2"):
                    if row2 is not None and "mcfadden_r2" in row2:
                        try:
                            parts.append(r"$R^2_{McF}$=" + f"{float(row2['mcfadden_r2']):.2f}")
                        except Exception:
                            pass
                elif kk in ("wrmse", "rmse_prob_w", "rmse"):
                    if row2 is not None and "rmse_prob_w" in row2:
                        try:
                            parts.append(f"wRMSE={float(row2['rmse_prob_w']):.3f}")
                        except Exception:
                            pass
                elif kk in ("brier", "brier_w", "wbrier"):
                    if row2 is not None and "brier_w" in row2:
                        try:
                            parts.append(f"wBrier={float(row2['brier_w']):.3f}")
                        except Exception:
                            pass
                elif kk == "delta_aic":
                    if row_cmp is not None and "delta_aic" in row_cmp:
                        try:
                            parts.append(r"$\Delta\mathrm{AIC}$=" + f"{float(row_cmp['delta_aic']):+.1f}")
                        except Exception:
                            pass
                elif kk in ("delta_mcfadden_r2", "delta_r2"):
                    if row_cmp is not None and "delta_mcfadden_r2" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta R^2_{McF}$=" + f"{float(row_cmp['delta_mcfadden_r2']):+.2f}"
                            )
                        except Exception:
                            pass
                elif kk in ("delta_rmse", "delta_rmse_prob_w"):
                    if row_cmp is not None and "delta_rmse_prob_w" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta$wRMSE=" + f"{float(row_cmp['delta_rmse_prob_w']):+.3f}"
                            )
                        except Exception:
                            pass
                elif kk in ("delta_brier", "delta_brier_w"):
                    if row_cmp is not None and "delta_brier_w" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta$wBrier=" + f"{float(row_cmp['delta_brier_w']):+.3f}"
                            )
                        except Exception:
                            pass
                elif kk in ("lr", "lr_stat", "lrchi2"):
                    if row_cmp is not None and "lr_stat" in row_cmp:
                        try:
                            parts.append(r"LR $\chi^2$=" + f"{float(row_cmp['lr_stat']):.1f}")
                        except Exception:
                            pass
                elif kk in ("lr_p", "lr_pvalue", "p_lr"):
                    if row_cmp is not None and "lr_pvalue" in row_cmp:
                        try:
                            parts.append(f"LR p={float(row_cmp['lr_pvalue']):.3f}")
                        except Exception:
                            pass

            return "\n".join(parts)

        # Panel ordering by label; optionally enforce metric order
        label_table = (
            df_plot[[metric_col, threshold_col, label_col]]
            .drop_duplicates()
            .sort_values(by=[metric_col, threshold_col])
        )
        label_order = label_table[label_col].tolist()

        if metric_order is not None:
            metric_rank = {m: i for i, m in enumerate(metric_order)}
            label_table["_metric_rank"] = label_table[metric_col].map(
                lambda m: metric_rank.get(m, len(metric_rank))
            )
            label_table = label_table.sort_values(by=["_metric_rank", threshold_col])
            label_order = label_table[label_col].tolist()

        low_cut, high_cut = prob_pass_cutoffs

        # global gradient levels (so legend is consistent)
        if grad_levels is None:
            g_all = df_plot[grad_col].dropna().to_numpy()
            if g_all.size == 0:
                grad_levels = [0.0]
            else:
                if grad_quantiles is None:
                    grad_quantiles = np.linspace(0.15, 0.85, n_grad_levels)
                grad_levels = np.quantile(g_all, np.asarray(grad_quantiles, dtype=float)).tolist()
        grad_levels = [float(g) for g in np.asarray(grad_levels, dtype=float)]
        grad_colors = sns.color_palette("viridis", n_colors=len(grad_levels))

        n_panels = len(label_order)
        if n_panels <= 2:
            n_rows, n_cols = 1, n_panels
        elif n_panels <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_cols = 3
            n_rows = int(np.ceil(n_panels / n_cols))

        with sns.axes_style(seaborn_style), sns.plotting_context(
            seaborn_context, font_scale=font_scale
        ):
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=figsize,
                dpi=dpi,
                sharex=False,
                sharey=True,
            )
            axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

            panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            letter_offset = ord(panel_letter_start.upper()) - ord("A")

            for idx, lbl in enumerate(label_order):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]

                sub = df_plot[df_plot[label_col] == lbl].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                metric_name = sub[metric_col].iloc[0]
                thr = sub[threshold_col].iloc[0]

                m_min = float(sub[margin_col].min())
                m_max = float(sub[margin_col].max())
                if not np.isfinite(m_min) or not np.isfinite(m_max):
                    x_grid = np.linspace(-2.0, 2.0, n_margin_grid)
                else:
                    span = max(m_max - m_min, 1e-3)
                    pad = 0.05 * span
                    x_grid = np.linspace(m_min - pad, m_max + pad, n_margin_grid)

                c_row = coef_plot.loc[coef_plot[label_col] == lbl]
                if c_row.empty:
                    raise ValueError(f"No coefficient row found for label={lbl!r}")
                c_row = c_row.iloc[0]
                b0 = float(c_row["b_const"])
                b_delta = float(c_row[f"b_{margin_col}"])
                b_grad = float(c_row[f"b_{grad_col}"])

                # scatter
                for cls in qa_class_order:
                    sub_cls = sub[sub["_qa_class_plot"] == cls]
                    if sub_cls.empty:
                        continue
                    ax.scatter(
                        sub_cls[margin_col],
                        sub_cls[p_pass_col],
                        s=scatter_size,
                        alpha=scatter_alpha if cls != "Borderline" else scatter_alpha * 0.8,
                        edgecolor="black",
                        linewidth=0.4,
                        color=qa_palette.get(cls, "0.6"),
                        zorder=3,
                    )

                # logistic families (fixed gradients)
                for g_val, color in zip(grad_levels, grad_colors):
                    eta = b0 + b_delta * x_grid + b_grad * float(g_val)
                    p_grid = 1.0 / (1.0 + np.exp(-eta))
                    ax.plot(x_grid, p_grid, color=color, linewidth=2.0, alpha=0.95, zorder=4)

                # optional overlay of margin-only fit (1D model)
                if overlay_1d_model and coef1_df is not None:
                    c1 = coef1_df.loc[coef1_df[coef1_label_col] == lbl]
                    if (not c1.empty) and ("b_const" in c1.columns) and (f"b_{margin_col}" in c1.columns):
                        c1 = c1.iloc[0]
                        b0_1d = float(c1["b_const"])
                        b_delta_1d = float(c1[f"b_{margin_col}"])
                        eta_1d = b0_1d + b_delta_1d * x_grid
                        p_1d = 1.0 / (1.0 + np.exp(-eta_1d))

                        line_kwargs = {
                            "color": "black",
                            "linestyle": "--",
                            "linewidth": 2.0,
                            "alpha": 0.9,
                            "zorder": 5,
                        }
                        if overlay_1d_kwargs is not None:
                            line_kwargs.update(overlay_1d_kwargs)
                        ax.plot(x_grid, p_1d, **line_kwargs)

                ax.axhline(high_cut, linestyle="--", linewidth=0.9, color="black", zorder=1)
                ax.axhline(low_cut, linestyle="--", linewidth=0.9, color="black", zorder=1)
                ax.axvline(0.0, linestyle=":", linewidth=0.9, color="black", zorder=1)

                ax.set_title(_make_title(metric_name, thr), fontsize="medium")
                ax.set_xlabel(_make_xlabel(metric_name, thr))
                if (idx % n_cols) == 0:
                    ax.set_ylabel(r"$p_{b,r}$ (Monte Carlo pass probability)")
                else:
                    ax.set_ylabel("")

                ax.set_ylim(-0.05, 1.05)

                # panel letter
                letter_idx = idx + letter_offset
                if 0 <= letter_idx < len(panel_letters):
                    ax.text(
                        0.02,
                        0.98,
                        panel_letters[letter_idx],
                        transform=ax.transAxes,
                        fontsize="large",
                        fontweight="bold",
                        va="top",
                        ha="left",
                    )

                # optional fit-metrics box
                if annotate_fit_stats:
                    txt = _format_panel_box_text(lbl)
                    if txt:
                        xa, ya = _choose_fitbox_loc(
                            sub[margin_col].to_numpy(), sub[p_pass_col].to_numpy(), txt
                        )
                        ax.text(
                            xa,
                            ya,
                            txt,
                            transform=ax.transAxes,
                            ha="left",
                            va="bottom",
                            fontsize="small",
                            bbox=dict(
                                boxstyle="round,pad=0.25",
                                facecolor="white",
                                edgecolor="black",
                                alpha=0.9,
                            ),
                            zorder=15,
                        )

            # hide unused axes
            for j in range(n_panels, len(axes_flat)):
                axes_flat[j].set_visible(False)

            # legends: QA classes (markers) + gradient families (lines) + optional 1D overlay
            qa_handles = []
            for cls in qa_class_order:
                if cls not in df_plot["_qa_class_plot"].unique():
                    continue
                qa_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=7,
                        markerfacecolor=qa_palette.get(cls, "0.6"),
                        markeredgecolor="black",
                        label=cls,
                    )
                )

            grad_handles = [
                Line2D(
                    [0],
                    [0],
                    color=col,
                    lw=2.0,
                    label=rf"$\bar{{g}} = {g:.2f}\ \mathrm{{Gy/mm}}$",
                )
                for g, col in zip(grad_levels, grad_colors)
            ]

            overlay_handle = None
            if overlay_1d_model:
                line_kwargs = {"color": "black", "linestyle": "--", "linewidth": 2.0}
                if overlay_1d_kwargs is not None:
                    line_kwargs.update(overlay_1d_kwargs)
                overlay_handle = Line2D(
                    [0],
                    [0],
                    color=line_kwargs.get("color", "black"),
                    lw=line_kwargs.get("linewidth", 2.0),
                    linestyle=line_kwargs.get("linestyle", "--"),
                    label="Margin-only logit fit",
                )

            handles = qa_handles + grad_handles
            if overlay_handle is not None:
                handles.append(overlay_handle)

            if handles:
                leg = fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=min(len(handles), 4),
                    frameon=True,
                    framealpha=0.95,
                    fontsize="small",
                    title=r"QA class, gradient families, and margin-only fit",
                    title_fontsize="small",
                )
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("black")

            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(save_dir) / file_name, dpi=dpi, bbox_inches="tight")

        return fig, axes


def production_plot_path1_logit_margin_plus_grad_families_generalized(
    pred_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    label_col: str = "label",
    margin_col: str = "distance_from_threshold_nominal",
    grad_col: str = "nominal_core_mean_grad_gy_per_mm",
    p_pass_col: str = "p_pass",
    qa_class_col: str = "qa_class",
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 300,
    seaborn_style: str = "whitegrid",
    seaborn_context: str = "paper",
    font_scale: float = 1.1,
    metric_order: Sequence[str] | None = None,
    save_dir: Path | str | None = None,
    file_name: str = "Fig_Path1_logit_margin_plus_grad_families.png",
    n_grad_levels: int = 4,
    grad_levels: Sequence[float] | None = None,
    grad_quantiles: Sequence[float] | None = None,
    prob_pass_cutoffs: tuple[float, float] = (0.05, 0.95),
    n_margin_grid: int = 200,
    scatter_size: float = 45.0,
    scatter_alpha: float = 0.9,
    panel_letter_start: str = "A",
    include_rule_in_xlabel: bool = True,   # <<< NEW
    # panel fit / comparison stats
    annotate_fit_stats: bool = False,
    fit_stats: Sequence[str] = (
        "n",
        "mcfadden_r2",
        "aic",
        "wrmse",
        "brier",
        "delta_aic",
        "delta_rmse",
        "lr_pvalue",
    ),
    fit_stats_loc: tuple[float, float] = (0.02, 0.05),
    comparison_df: pd.DataFrame | None = None,
    comparison_label_col: str = "label",
    # overlay margin-only (1D) model
    overlay_1d_model: bool = False,
    coef1_df: pd.DataFrame | None = None,
    coef1_label_col: str = "label",
    overlay_1d_kwargs: dict | None = None,
    # Optional: per-panel control of which secondary predictor to use
    per_label_secondary: dict[str, str] | None = None,
    # Optional: per-panel legend formatting for secondary predictor levels
    per_label_grad_label_template: dict[str, str] | None = None,
    per_label_legend_title: dict[str, str] | None = None,
    # NEW: per-panel units & annotation for the secondary predictor
    per_label_secondary_unit: dict[str, str] | None = None,
    per_label_secondary_annotation: dict[str, str] | None = None,
    # NEW: manual override for stats-box location, per label
    # values can be: "top-left", "top-right", "bottom-left", "bottom-right"
    per_label_stats_box_corner: dict[str, str] | None = None,
    per_label_stats_box_xy: dict[str, tuple[float, float]] | None = None,
    # NEW: font sizes for per-panel secondary-legend and fit-stats box
    panel_legend_fontsize: float | str = 8,
    panel_legend_title_fontsize: float | str = 8,
    fit_stats_fontsize: float | str = 9,
    annotation_box_width_fraction: float = 0.28,

):
    """
    Generalised Path-1 QA figure: 2-predictor logit fit (margin + secondary)
    for each threshold, with *identical styling* to
    `production_plot_path1_logit_margin_plus_grad_families`, but allowing
    a different secondary predictor in each panel.

    Expected inputs
    ---------------
    pred_df :
        Concatenation of prediction tables from `fit_path1_logit_per_threshold`
        for all candidate secondary predictors. In the 2D-scan pipeline this is
        `pred2_all_df`, which includes:
          - metric_col, threshold_col, label_col
          - margin_col
          - p_pass_col, n_pass, n_trials, qa_class_col
          - one column per candidate secondary predictor, e.g.
                nominal_core_mean_grad_gy_per_mm,
                nominal_core_mean_dose_gy,
                rectum_nn_dist_mm, ...
          - a column "secondary_predictor" indicating which predictor was used
            in that particular 2D model block.

    coef_df :
        Concatenation of coefficient tables from `fit_path1_logit_per_threshold`
        for all candidate secondary predictors (`coef2_all_df`), with columns:
          - metric_col, threshold_col, label_col
          - "secondary_predictor"
          - b_const, b_<margin_col>, b_<secondary_predictor>, ...
          - goodness-of-fit metrics (aic, mcfadden_r2, brier_w, rmse_prob_w, ...)

    comparison_df :
        Optional table containing 1D vs 2D comparison metrics. If it has a
        column "secondary_predictor" (e.g. `best_per_rule`), we will use that
        to infer the best secondary predictor per label *unless* an explicit
        `per_label_secondary` mapping is supplied.
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    plt.ioff()

    # --- basic column checks on predictions ---
    required_pred = {metric_col, threshold_col, label_col, margin_col, p_pass_col}
    missing_pred = required_pred - set(pred_df.columns)
    if missing_pred:
        raise ValueError(f"pred_df is missing required columns: {sorted(missing_pred)}")

    # We only require grad_col if the user does NOT supply any per-label mapping
    if per_label_secondary is None and grad_col not in pred_df.columns:
        raise ValueError(
            "grad_col is not present in pred_df and no per_label_secondary "
            "mapping was provided."
        )

    # --- basic column checks on coefficients ---
    required_coef = {label_col, "b_const", f"b_{margin_col}"}
    missing_coef = required_coef - set(coef_df.columns)
    if missing_coef:
        raise ValueError(f"coef_df is missing required columns: {sorted(missing_coef)}")

    # --- infer per-label secondary predictor from comparison_df if available ---
    label_to_secondary: dict[str, str] = {}

    # Start from explicit user mapping, if provided
    if per_label_secondary:
        label_to_secondary.update(per_label_secondary)

    # If the caller passed a comparison_df with a secondary_predictor column,
    # use that as the default mapping for any labels not already in the dict.
    if comparison_df is not None and "secondary_predictor" in comparison_df.columns:
        best_map = (
            comparison_df[[comparison_label_col, "secondary_predictor"]]
            .dropna(subset=["secondary_predictor"])
            .drop_duplicates(subset=[comparison_label_col])
        )
        for _, row in best_map.iterrows():
            lbl = str(row[comparison_label_col])
            sec = str(row["secondary_predictor"])
            if lbl not in label_to_secondary:
                label_to_secondary[lbl] = sec

    # If we still have no mapping at all, we fall back to a single global grad_col
    # (the original behaviour: one predictor for all panels).
    use_per_label = bool(label_to_secondary)

    # convenience: presence of secondary_predictor tag columns
    has_sec_tag_pred = "secondary_predictor" in pred_df.columns
    has_sec_tag_coef = "secondary_predictor" in coef_df.columns

    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
        }
    ):
        df_plot = pred_df.copy()
        coef_plot = coef_df.copy()

        # --- QA class normalisation ---
        def _norm_qa(c):
            c_str = str(c).strip().lower()
            if "confident" in c_str and "pass" in c_str:
                return "Confident pass"
            if "confident" in c_str and "fail" in c_str:
                return "Confident fail"
            if "border" in c_str:
                return "Borderline"
            return str(c)

        if qa_class_col in df_plot.columns:
            df_plot["_qa_class_plot"] = df_plot[qa_class_col].map(_norm_qa)
        else:
            df_plot["_qa_class_plot"] = "NA"

        qa_class_order = ["Confident pass", "Borderline", "Confident fail"]
        qa_palette = {
            "Confident pass": "#66c2a5",
            "Borderline": "#fc8d62",
            "Confident fail": "#8da0cb",
            "NA": "0.6",
        }

        # --- DVH label helpers (unchanged from original) ---
        """
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            if isinstance(thr, (int, np.integer)):
                return f"{int(thr)}"
            if isinstance(thr, float):
                return f"{thr:g}"
            try:
                return f"{float(thr):g}"
            except Exception:
                return str(thr)

        def _make_title(metric_name: str, thr) -> str:
            mm = _metric_math(metric_name)
            t = _thr_str(thr)
            if _metric_is_percent(metric_name):
                return rf"{mm}$\ \geq\ {t}\%$"
            return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        def _make_xlabel(metric_name: str, thr) -> str:
            mm = _metric_math(metric_name)
            t = _thr_str(thr)
            if _metric_is_percent(metric_name):
                return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t}%)"
            return rf"Nominal margin $\delta_{{b,r}}$ ({mm} nominal margin vs. {t} Gy)"
        """

                # --- DVH label helpers ---
        def _metric_math(m: str) -> str:
            s = str(m)
            if s.startswith("D_") and "%" in s:
                sub = s.split("D_")[1].split("%")[0].strip()
                return rf"$D_{{{sub}\%}}$"
            if s.startswith("V_") and "%" in s:
                sub = s.split("V_")[1].split("%")[0].strip()
                return rf"$V_{{{sub}\%}}$"
            return rf"$\mathrm{{{s}}}$"

        def _metric_is_percent(m: str) -> bool:
            s = str(m)
            return s.startswith("V_") and "%" in s

        def _thr_str(thr) -> str:
            if isinstance(thr, (int, np.integer)):
                return f"{int(thr)}"
            if isinstance(thr, float):
                return f"{thr:g}"
            try:
                return f"{float(thr):g}"
            except Exception:
                return str(thr)

        def _make_criterion(metric_name: str, thr) -> str:
            """
            Criterion text, e.g. '$D_{2\\%}$\\ \\geq\\ 32\\,\\mathrm{Gy}$'
            """
            mm = _metric_math(metric_name)
            t = _thr_str(thr)
            if _metric_is_percent(metric_name):
                return rf"{mm}$\ \geq\ {t}\%$"
            return rf"{mm}$\ \geq\ {t}\,\mathrm{{Gy}}$"

        def _make_title(metric_name: str, thr) -> str:
            # Panel title is just the criterion
            return _make_criterion(metric_name, thr)

        def _make_xlabel(
            metric_name: str,
            thr,
            rule_index: int,
            include_rule: bool = True,
            is_percent: bool | None = None,
        ) -> str:
            """
            X label, e.g.
              'Nominal dose margin δ_{b,1} (Gy; r = 1 : D_{2%} ≥ 32 Gy)'
            """
            # Decide units for the margin
            if is_percent is None:
                is_percent = _metric_is_percent(metric_name)

            unit_text = "%" if is_percent else "Gy"

            # main symbol δ_{b,1}
            rule_sym = rf"\delta_{{b,{rule_index}}}"
            # keep δ in math, units as plain text
            label = rf"Nominal dose margin ${rule_sym}$ ({unit_text})"

            if include_rule:
                # Use the same criterion text as the title
                crit = _make_criterion(metric_name, thr)
                # strip outer $ so we embed it in one math block
                crit_inner = crit.strip("$")
                label += rf" ($r = {rule_index}: {crit_inner}$)"

            return label



        # --- helper to place fit-metrics box (unchanged) ---
        def _choose_fitbox_loc(xvals: np.ndarray, yvals: np.ndarray, txt: str) -> tuple[float, float]:
            try:
                xvals = np.asarray(xvals, float)
                yvals = np.asarray(yvals, float)
                m = np.isfinite(xvals) & np.isfinite(yvals)
                xvals = xvals[m]
                yvals = yvals[m]
                if xvals.size == 0 or yvals.size == 0:
                    return fit_stats_loc

                xmin, xmax = float(xvals.min()), float(xvals.max())
                ymin, ymax = -0.05, 1.05
                if xmax <= xmin:
                    return fit_stats_loc

                xN = (xvals - xmin) / (xmax - xmin)
                yN = (yvals - ymin) / (ymax - ymin)
                yN = np.clip(yN, 0.0, 1.0)

                n_lines = txt.count("\n") + 1
                box_w = 0.32
                box_h = min(0.13 + 0.05 * n_lines, 0.40)

                candidates = [
                    (0.66, 0.55),
                    (0.66, 0.10),
                    (0.02, 0.55),
                    (0.02, 0.10),
                ]

                best = fit_stats_loc
                best_score = 1e9
                for xa, ya in candidates:
                    xb = min(xa + box_w, 0.99)
                    yb = min(ya + box_h, 0.99)
                    inside = (xN >= xa) & (xN <= xb) & (yN >= ya) & (yN <= yb)
                    score = int(inside.sum())
                    if score < best_score:
                        best_score = score
                        best = (xa, ya)
                return best
            except Exception:
                return fit_stats_loc

        # --- fit-metrics text box (now respects chosen secondary predictor) ---
        def _format_panel_box_text(lbl: str) -> str:
            parts: list[str] = []

            sec = label_to_secondary.get(lbl) if use_per_label else None

            # Coefficients row (from coef_plot)
            row2 = coef_plot.loc[coef_plot[label_col] == lbl]
            if sec is not None and has_sec_tag_coef and not row2.empty:
                row2 = row2.loc[row2["secondary_predictor"] == sec]
            row2 = row2.iloc[0] if not row2.empty else None

            # Comparison metrics row (from comparison_df)
            row_cmp = None
            if comparison_df is not None:
                sub = comparison_df.loc[comparison_df[comparison_label_col] == lbl]
                if sec is not None and "secondary_predictor" in comparison_df.columns and not sub.empty:
                    sub = sub.loc[sub["secondary_predictor"] == sec]
                if not sub.empty:
                    row_cmp = sub.iloc[0]

            for key in fit_stats:
                kk = str(key).lower()
                if kk in ("n", "n_biopsies", "npts", "n_points"):
                    if row2 is not None and "n_biopsies" in row2:
                        try:
                            parts.append(f"N={int(row2['n_biopsies'])}")
                        except Exception:
                            pass
                elif kk == "aic":
                    if row2 is not None and "aic" in row2:
                        try:
                            parts.append(f"AIC={float(row2['aic']):.1f}")
                        except Exception:
                            pass
                elif kk in ("mcfadden_r2", "mcfadden", "pseudo_r2", "r2"):
                    if row2 is not None and "mcfadden_r2" in row2:
                        try:
                            parts.append(r"$R^2_{McF}$=" + f"{float(row2['mcfadden_r2']):.2f}")
                        except Exception:
                            pass
                elif kk in ("wrmse", "rmse_prob_w", "rmse"):
                    if row2 is not None and "rmse_prob_w" in row2:
                        try:
                            parts.append(f"wRMSE={float(row2['rmse_prob_w']):.2f}")
                        except Exception:
                            pass
                elif kk in ("brier", "brier_w", "wbrier"):
                    if row2 is not None and "brier_w" in row2:
                        try:
                            parts.append(f"wBrier={float(row2['brier_w']):.2f}")
                        except Exception:
                            pass
                elif kk == "delta_aic":
                    if row_cmp is not None and "delta_aic" in row_cmp:
                        try:
                            parts.append(r"$\Delta\mathrm{AIC}$=" + f"{float(row_cmp['delta_aic']):+.1f}")
                        except Exception:
                            pass
                elif kk in ("delta_mcfadden_r2", "delta_r2"):
                    if row_cmp is not None and "delta_mcfadden_r2" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta R^2_{McF}$=" + f"{float(row_cmp['delta_mcfadden_r2']):+.2f}"
                            )
                        except Exception:
                            pass
                elif kk in ("delta_rmse", "delta_rmse_prob_w"):
                    if row_cmp is not None and "delta_rmse_prob_w" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta$wRMSE=" + f"{float(row_cmp['delta_rmse_prob_w']):+.2f}"
                            )
                        except Exception:
                            pass
                elif kk in ("delta_brier", "delta_brier_w"):
                    if row_cmp is not None and "delta_brier_w" in row_cmp:
                        try:
                            parts.append(
                                r"$\Delta$wBrier=" + f"{float(row_cmp['delta_brier_w']):+.2f}"
                            )
                        except Exception:
                            pass
                elif kk in ("lr", "lr_stat", "lrchi2"):
                    if row_cmp is not None and "lr_stat" in row_cmp:
                        try:
                            parts.append(r"LR $\chi^2$=" + f"{float(row_cmp['lr_stat']):.1f}")
                        except Exception:
                            pass
                elif kk in ("lr_p", "lr_pvalue", "p_lr"):
                    if row_cmp is not None and "lr_pvalue" in row_cmp:
                        try:
                            parts.append(f"LR p={float(row_cmp['lr_pvalue']):.2f}")
                        except Exception:
                            pass

            return "\n".join(parts)

        # --- Panel ordering by label; optionally enforce metric order (unchanged) ---
        label_table = (
            df_plot[[metric_col, threshold_col, label_col]]
            .drop_duplicates()
            .sort_values(by=[metric_col, threshold_col])
        )
        label_order = label_table[label_col].tolist()

        if metric_order is not None:
            metric_rank = {m: i for i, m in enumerate(metric_order)}
            label_table["_metric_rank"] = label_table[metric_col].map(
                lambda m: metric_rank.get(m, len(metric_rank))
            )
            label_table = label_table.sort_values(by=["_metric_rank", threshold_col])
            label_order = label_table[label_col].tolist()

        low_cut, high_cut = prob_pass_cutoffs

        # --- colours for secondary predictor families ---
        # We only use grad_levels to determine how many families (and thus colours)
        # if the user supplied them explicitly.
        if grad_levels is not None:
            n_levels = len(grad_levels)
        else:
            n_levels = n_grad_levels

        grad_colors = sns.color_palette("viridis", n_colors=n_levels)

        n_panels = len(label_order)
        if n_panels <= 2:
            n_rows, n_cols = 1, n_panels
        elif n_panels <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_cols = 3
            n_rows = int(np.ceil(n_panels / n_cols))

        with sns.axes_style(seaborn_style), sns.plotting_context(
            seaborn_context, font_scale=font_scale
        ):
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=figsize,
                dpi=dpi,
                sharex=False,
                sharey=True,
            )
            axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

            panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            letter_offset = ord(panel_letter_start.upper()) - ord("A")

            # Global legend handles: QA class markers (+ optional 1D overlay).
            qa_handles_for_global: list[Line2D] = []
            qa_present_global = set(df_plot["_qa_class_plot"].unique())
            for cls in qa_class_order:
                if cls not in qa_present_global:
                    continue
                qa_handles_for_global.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=7,
                        markerfacecolor=qa_palette.get(cls, "0.6"),
                        markeredgecolor="black",
                        label=cls,
                    )
                )

            overlay_handle_global = None
            if overlay_1d_model:
                line_kwargs = {"color": "black", "linestyle": "--", "linewidth": 2.0}
                if overlay_1d_kwargs is not None:
                    line_kwargs.update(overlay_1d_kwargs)
                overlay_handle_global = Line2D(
                    [0],
                    [0],
                    color=line_kwargs.get("color", "black"),
                    lw=line_kwargs.get("linewidth", 2.0),
                    linestyle=line_kwargs.get("linestyle", "--"),
                    label="Margin-only logit fit",
                )

            # --- main panel loop ---
            for idx, lbl in enumerate(label_order):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]

                sub = df_plot[df_plot[label_col] == lbl].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                metric_name = sub[metric_col].iloc[0]
                thr = sub[threshold_col].iloc[0]

                # Which secondary predictor does this panel use?
                if use_per_label and lbl in label_to_secondary:
                    this_grad_col = label_to_secondary[lbl]
                else:
                    this_grad_col = grad_col

                if this_grad_col not in sub.columns:
                    raise ValueError(
                        f"Secondary predictor column {this_grad_col!r} not found in pred_df for label={lbl!r}"
                    )

                # --- determine unit string for this panel ---
                if per_label_secondary_unit is not None and lbl in per_label_secondary_unit:
                    unit_str = per_label_secondary_unit[lbl]  # e.g. r"\mathrm{Gy/mm}"
                else:
                    unit_str = ""


                # x-grid in margin space
                m_min = float(sub[margin_col].min())
                m_max = float(sub[margin_col].max())
                if not np.isfinite(m_min) or not np.isfinite(m_max):
                    x_grid = np.linspace(-2.0, 2.0, n_margin_grid)
                else:
                    span = max(m_max - m_min, 1e-3)
                    pad = 0.05 * span
                    x_grid = np.linspace(m_min - pad, m_max + pad, n_margin_grid)

                # coefficients row for this panel
                c_row = coef_plot.loc[coef_plot[label_col] == lbl].copy()
                if use_per_label and has_sec_tag_coef:
                    c_row = c_row.loc[c_row["secondary_predictor"] == this_grad_col]

                if c_row.empty:
                    raise ValueError(
                        f"No coefficient row found for label={lbl!r} and secondary predictor={this_grad_col!r}"
                    )
                c_row = c_row.iloc[0]

                b0 = float(c_row["b_const"])
                b_delta = float(c_row[f"b_{margin_col}"])

                coef_grad_colname = f"b_{this_grad_col}"
                if coef_grad_colname not in c_row.index:
                    raise ValueError(
                        f"coef_df is missing coefficient column {coef_grad_colname!r} "
                        f"for label={lbl!r}. Available columns: {list(c_row.index)}"
                    )
                b_grad = float(c_row[coef_grad_colname])

                # scatter of empirical Monte Carlo pass probabilities
                for cls in qa_class_order:
                    sub_cls = sub[sub["_qa_class_plot"] == cls]
                    if sub_cls.empty:
                        continue
                    ax.scatter(
                        sub_cls[margin_col],
                        sub_cls[p_pass_col],
                        s=scatter_size,
                        alpha=scatter_alpha if cls != "Borderline" else scatter_alpha * 0.8,
                        edgecolor="black",
                        linewidth=0.4,
                        color=qa_palette.get(cls, "0.6"),
                        zorder=3,
                    )

                # panel-specific secondary predictor values (for family curves)
                sec_vals = sub[this_grad_col].dropna().to_numpy()
                if sec_vals.size == 0:
                    sec_levels = np.zeros(n_levels, dtype=float)
                else:
                    if grad_quantiles is None:
                        q = np.linspace(0.15, 0.85, n_levels)
                    else:
                        q = np.asarray(grad_quantiles, dtype=float)
                    sec_levels = np.quantile(sec_vals, q).astype(float)

                # Build panel-specific legend handles for secondary levels
                grad_handles_panel: list[Line2D] = []

                # Choose legend label template for this panel
                if per_label_grad_label_template is not None and lbl in per_label_grad_label_template:
                    # user-specified template; can use {value} and optional {unit}
                    panel_grad_template = per_label_grad_label_template[lbl]
                else:
                    # sensible defaults
                    if this_grad_col == "nominal_core_mean_grad_gy_per_mm":
                        if not unit_str:
                            unit_str = r"\mathrm{Gy/mm}"
                        panel_grad_template = r"$\bar{{g}} = {value:.2f}\ {unit}$"
                    else:
                        if unit_str:
                            panel_grad_template = f"{this_grad_col} = " + "{value:.2f} {unit}"
                        else:
                            panel_grad_template = f"{this_grad_col} = " + "{value:.2f}"


                # logistic families
                for color, sec_val in zip(grad_colors, sec_levels):
                    eta = b0 + b_delta * x_grid + b_grad * float(sec_val)
                    p_grid = 1.0 / (1.0 + np.exp(-eta))
                    ax.plot(x_grid, p_grid, color=color, linewidth=2.0, alpha=0.95, zorder=4)

                    grad_handles_panel.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            lw=2.0,
                            label=panel_grad_template.format(value=sec_val, unit=unit_str),
                        )
                    )


                # optional overlay of 1D margin-only model
                if overlay_1d_model and coef1_df is not None:
                    c1 = coef1_df.loc[coef1_df[coef1_label_col] == lbl]
                    if (not c1.empty) and ("b_const" in c1.columns) and (f"b_{margin_col}" in c1.columns):
                        c1 = c1.iloc[0]
                        b0_1d = float(c1["b_const"])
                        b_delta_1d = float(c1[f"b_{margin_col}"])
                        eta_1d = b0_1d * np.ones_like(x_grid) + b_delta_1d * x_grid
                        p_1d = 1.0 / (1.0 + np.exp(-eta_1d))

                        line_kwargs = {
                            "color": "black",
                            "linestyle": "--",
                            "linewidth": 2.0,
                            "alpha": 0.9,
                            "zorder": 5,
                        }
                        if overlay_1d_kwargs is not None:
                            line_kwargs.update(overlay_1d_kwargs)
                        ax.plot(x_grid, p_1d, **line_kwargs)

                ax.axhline(high_cut, linestyle="--", linewidth=0.9, color="black", zorder=1)
                ax.axhline(low_cut, linestyle="--", linewidth=0.9, color="black", zorder=1)
                ax.axvline(0.0, linestyle=":", linewidth=0.9, color="black", zorder=1)

                # Build 1–2 line title: rule + optional secondary predictor annotation
                title_str = _make_criterion(metric_name, thr)  # same as before, e.g. "$D_{2\%} \ge 32\,\mathrm{Gy}$"

                if per_label_secondary_annotation is not None and lbl in per_label_secondary_annotation:
                    # second line: arbitrary LaTeX / text you pass in the dict
                    sec_annot = per_label_secondary_annotation[lbl]
                    title_str = title_str + "\n" + sec_annot

                ax.set_title(title_str, fontsize="medium", pad=8)

                # Is this rule a percent-type margin? (e.g. V150 ≥ 50%)
                is_percent = ("%" in str(lbl)) or _metric_is_percent(metric_name)

                ax.set_xlabel(
                    _make_xlabel(
                        metric_name,
                        thr,
                        rule_index=idx + 1,
                        include_rule=include_rule_in_xlabel,
                        is_percent=is_percent,
                    )
                )

                if (idx % n_cols) == 0:
                    ax.set_ylabel(r"$p_{b,r}$ (Monte Carlo pass probability)")
                else:
                    ax.set_ylabel("")


                ax.set_ylim(-0.05, 1.05)

                # panel letter
                letter_idx = idx + letter_offset
                if 0 <= letter_idx < len(panel_letters):
                    ax.text(
                        0.02,
                        0.98,
                        panel_letters[letter_idx],
                        transform=ax.transAxes,
                        fontsize="large",
                        fontweight="bold",
                        va="top",
                        ha="left",
                    )

                                # per-panel secondary legend (if per_label_* used)
                leg_panel = None
                if per_label_secondary is not None or per_label_grad_label_template is not None:
                    if per_label_legend_title is not None and lbl in per_label_legend_title:
                        panel_legend_title = per_label_legend_title[lbl]
                    else:
                        panel_legend_title = "Secondary predictor levels"

                    leg_panel = ax.legend(
                        handles=grad_handles_panel,
                        loc="lower right",
                        fontsize=panel_legend_fontsize,
                        title=panel_legend_title,
                        title_fontsize=panel_legend_title_fontsize,
                        frameon=True,
                        framealpha=0.95,
                    )

                    leg_panel.get_frame().set_facecolor("white")
                    leg_panel.get_frame().set_edgecolor("black")

                if annotate_fit_stats:
                    txt = _format_panel_box_text(lbl)
                    if txt:
                        # 1) explicit per-panel coordinates win
                        if per_label_stats_box_xy is not None and lbl in per_label_stats_box_xy:
                            xa, ya = per_label_stats_box_xy[lbl]

                        # 2) otherwise, named corner override (top-left, top-right, etc.)
                        elif per_label_stats_box_corner is not None and lbl in per_label_stats_box_corner:
                            corner = per_label_stats_box_corner[lbl]
                            if corner == "top-left":
                                xa, ya = 0.02, 0.68
                            elif corner == "top-right":
                                xa, ya = 0.66, 0.68
                            elif corner == "bottom-left":
                                xa, ya = 0.02, 0.08
                            elif corner == "bottom-right":
                                xa, ya = 0.66, 0.08
                            else:
                                # unknown string → fall back to automatic
                                xa, ya = _choose_fitbox_loc(
                                    sub[margin_col].to_numpy(),
                                    sub[p_pass_col].to_numpy(),
                                    txt,
                                )

                        # 3) otherwise, existing behaviour (glued to legend if present,
                        #    else automatic placement)
                        elif leg_panel is not None:
                            try:
                                fig.canvas.draw()
                                bbox = leg_panel.get_window_extent()
                                bbox_ax = bbox.transformed(ax.transAxes.inverted())
                                box_w = annotation_box_width_fraction
                                pad = 0.02
                                xa = max(bbox_ax.x0 - box_w - pad, 0.02)
                                ya = bbox_ax.y0
                            except Exception:
                                xa, ya = _choose_fitbox_loc(
                                    sub[margin_col].to_numpy(),
                                    sub[p_pass_col].to_numpy(),
                                    txt,
                                )
                        else:
                            xa, ya = _choose_fitbox_loc(
                                sub[margin_col].to_numpy(),
                                sub[p_pass_col].to_numpy(),
                                txt,
                            )

                        ax.text(
                            xa,
                            ya,
                            txt,
                            transform=ax.transAxes,
                            ha="left",
                            va="bottom",
                            fontsize=fit_stats_fontsize,
                            bbox=dict(
                                boxstyle="round,pad=0.25",
                                facecolor="white",
                                edgecolor="black",
                                alpha=0.9,
                            ),
                            zorder=15,
                        )





            # hide unused axes
            for j in range(n_panels, len(axes_flat)):
                axes_flat[j].set_visible(False)

            # --- Global legend: QA classes (+ optional 1D overlay) ---
            handles_global = qa_handles_for_global.copy()
            if overlay_handle_global is not None:
                handles_global.append(overlay_handle_global)

            if handles_global:
                leg_global = fig.legend(
                    handles=handles_global,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=min(len(handles_global), 4),
                    frameon=True,
                    framealpha=0.95,
                    fontsize="small",
                    title=r"QA class and margin-only fit",
                    title_fontsize="small",
                )
                leg_global.get_frame().set_facecolor("white")
                leg_global.get_frame().set_edgecolor("black")

            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(save_dir) / file_name, dpi=dpi, bbox_inches="tight")

        return fig, axes







def _build_sextant_order(df: pd.DataFrame):
    """Return ordered list of sextant keys + plain labels."""
    def to_key(row):
        lr = row['Bx voxel prostate sextant (LR)']
        ap = row['Bx voxel prostate sextant (AP)']
        si = row['Bx voxel prostate sextant (SI)']
        lr_order = {'Left': 0, 'Right': 1}
        ap_order = {'Anterior': 0, 'Posterior': 1}
        si_order = {'Base (Superior)': 0, 'Mid': 1, 'Apex (Inferior)': 2}
        return (
            lr_order.get(lr, 99),
            ap_order.get(ap, 99),
            si_order.get(si, 99),
        )

    sub = (
        df[
            [
                'Bx voxel prostate sextant (LR)',
                'Bx voxel prostate sextant (AP)',
                'Bx voxel prostate sextant (SI)',
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    sub['__order_key'] = sub.apply(to_key, axis=1)
    sub = sub.sort_values('__order_key')

    keys = []
    labels = []
    for _, row in sub.iterrows():
        lr = row['Bx voxel prostate sextant (LR)']
        ap = row['Bx voxel prostate sextant (AP)']
        si = row['Bx voxel prostate sextant (SI)']

        lr_short = 'L' if lr == 'Left' else 'R'
        ap_short = 'Ant' if ap == 'Anterior' else 'Post'
        si_short = {
            'Base (Superior)': 'Base',
            'Mid': 'Mid',
            'Apex (Inferior)': 'Apex',
        }[si]

        keys.append((lr, ap, si))
        labels.append(f"{lr_short}-{ap_short}-{si_short}")

    return keys, labels


def plot_sextant_dose_bias_panels(
    sextant_mc_dose_grad: pd.DataFrame,
    sextant_mc_deltas: pd.DataFrame,
    sextant_nominal_deltas: pd.DataFrame,
    output_dir,
    fig_scale: float = 1.0,
    dpi: int = 300,
    show: bool = False,
    fig_name: str = "Fig_sextant_dose_bias_by_double_sextant",
    # font controls
    title_fontsize: float = 14,
    cell_fontsize: float = 8,
    ytick_fontsize: float = 10,
    cbar_label_fontsize: float = 11,
    cbar_tick_fontsize: float = 9,
) -> None:
    """
    6-panel heatmap summary by prostate double sextant (3×2 layout).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys, base_labels = _build_sextant_order(sextant_mc_dose_grad)

    # n_vox per sextant (if present)
    if "n_voxels" in sextant_mc_dose_grad.columns:
        nvox_map = {
            (
                row["Bx voxel prostate sextant (LR)"],
                row["Bx voxel prostate sextant (AP)"],
                row["Bx voxel prostate sextant (SI)"],
            ): int(row["n_voxels"])
            for _, row in sextant_mc_dose_grad.iterrows()
        }

        row_labels = [
            rf"{lbl} ($n_{{\mathrm{{vox}}}}={nvox_map.get(k, 0):d}$)"
            for lbl, k in zip(base_labels, keys)
        ]
    else:
        row_labels = base_labels

    def _extract(df, median_col, q05_col=None, q95_col=None):
        """Return (median, q05, q95) as (N,1) arrays in sextant order."""
        med, q05, q95 = [], [], []
        for lr, ap, si in keys:
            row = df[
                (df["Bx voxel prostate sextant (LR)"] == lr)
                & (df["Bx voxel prostate sextant (AP)"] == ap)
                & (df["Bx voxel prostate sextant (SI)"] == si)
            ]
            if row.empty:
                med.append(np.nan)
                q05.append(np.nan)
                q95.append(np.nan)
            else:
                r = row.iloc[0]
                med.append(float(r[median_col]))
                if q05_col is not None:
                    q05.append(float(r[q05_col]))
                    q95.append(float(r[q95_col]))
        med = np.asarray(med)[:, None]
        q05 = np.asarray(q05)[:, None] if q05_col is not None else None
        q95 = np.asarray(q95)[:, None] if q95_col is not None else None
        return med, q05, q95

    # --- panel data (same as before) ---
    dose_med, dose_q05, dose_q95 = _extract(
        sextant_mc_dose_grad,
        "Dose (Gy) median",
        "Dose (Gy) q05",
        "Dose (Gy) q95",
    )

    grad_med, grad_q05, grad_q95 = _extract(
        sextant_mc_dose_grad,
        "Dose grad (Gy/mm) median",
        "Dose grad (Gy/mm) q05",
        "Dose grad (Gy/mm) q95",
    )

    trial_abs_med, trial_abs_q05, trial_abs_q95 = _extract(
        sextant_mc_deltas,
        "Dose (Gy) abs deltas abs_nominal_minus_trial median",
        "Dose (Gy) abs deltas abs_nominal_minus_trial q05",
        "Dose (Gy) abs deltas abs_nominal_minus_trial q95",
    )

    bias_q50_med, bias_q50_q05, bias_q50_q95 = _extract(
        sextant_nominal_deltas,
        "Dose (Gy) deltas nominal_minus_q50 median",
        "Dose (Gy) deltas nominal_minus_q50 q05",
        "Dose (Gy) deltas nominal_minus_q50 q95",
    )

    bias_mean_med, bias_mean_q05, bias_mean_q95 = _extract(
        sextant_nominal_deltas,
        "Dose (Gy) deltas nominal_minus_mean median",
        "Dose (Gy) deltas nominal_minus_mean q05",
        "Dose (Gy) deltas nominal_minus_mean q95",
    )

    bias_mode_med, bias_mode_q05, bias_mode_q95 = _extract(
        sextant_nominal_deltas,
        "Dose (Gy) deltas nominal_minus_mode median",
        "Dose (Gy) deltas nominal_minus_mode q05",
        "Dose (Gy) deltas nominal_minus_mode q95",
    )

    data_mats = [
        dose_med,
        grad_med,
        trial_abs_med,
        bias_q50_med,
        bias_mean_med,
        bias_mode_med,
    ]

    q05_mats = [
        dose_q05,
        grad_q05,
        trial_abs_q05,
        bias_q50_q05,
        bias_mean_q05,
        bias_mode_q05,
    ]

    q95_mats = [
        dose_q95,
        grad_q95,
        trial_abs_q95,
        bias_q50_q95,
        bias_mean_q95,
        bias_mode_q95,
    ]

    titles = [
        r"$D_{b,v}^{(t)}$",
        r"$G_{b,v}^{(t)}$",
        r"$|\Delta D_{b,v,t}|$",
        r"$\Delta D^{(\mathrm{Q50})}_{b,v}$",
        r"$\Delta D^{(\mathrm{mean})}_{b,v}$",
        r"$\Delta D^{(\mathrm{mode})}_{b,v}$",
    ]

    cmaps = [
        "viridis",
        "magma",
        "cividis",
        "coolwarm",
        "coolwarm",
        "coolwarm",
    ]

    cbar_labels = [
        r"Median $D_{b,v}^{(t)}$ (Gy)",
        r"Median $G_{b,v}^{(t)}$ (Gy mm$^{-1}$)",
        r"Median $|\Delta D_{b,v,t}|$ (Gy)",
        r"Median $\Delta D^{(\mathrm{Q50})}_{b,v}$ (Gy)",
        r"Median $\Delta D^{(\mathrm{mean})}_{b,v}$ (Gy)",
        r"Median $\Delta D^{(\mathrm{mode})}_{b,v}$ (Gy)",
    ]

    norms = []
    for j, mat in enumerate(data_mats):
        finite = np.isfinite(mat)
        if not finite.any():
            if j >= 3:
                norms.append(TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
            else:
                norms.append(Normalize(vmin=0.0, vmax=1.0))
            continue

        if j >= 3:
            max_abs = float(np.nanmax(np.abs(mat[finite])))
            if max_abs == 0:
                max_abs = 1e-6
            norms.append(TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs))
        else:
            vmax = float(np.nanmax(mat[finite]))
            if vmax <= 0:
                vmax = 1e-6
            norms.append(Normalize(vmin=0.0, vmax=vmax))

    # slightly taller again → more vertical padding
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.5 * fig_scale, 14.0 * fig_scale),
        constrained_layout=True,
    )

    for j in range(6):
        row, col = divmod(j, 2)
        ax = axes[row, col]

        mat = data_mats[j]
        q05_mat = q05_mats[j]
        q95_mat = q95_mats[j]

        annot = np.empty(mat.shape, dtype=object)
        for i in range(mat.shape[0]):
            if np.isnan(mat[i, 0]):
                annot[i, 0] = ""
            else:
                annot[i, 0] = (
                    f"{mat[i, 0]:.1f}\n"
                    f"[{q05_mat[i, 0]:.1f},{q95_mat[i, 0]:.1f}]"
                )

        hm = sns.heatmap(
            mat,
            ax=ax,
            cmap=cmaps[j],
            norm=norms[j],
            cbar=True,
            cbar_kws={
                "shrink": 1,
                "label": cbar_labels[j],
                "fraction": 0.15,
                "pad": 0.04,
                "aspect": 20,
            },
            annot=annot,
            fmt="",
            annot_kws={
                "fontsize": cell_fontsize,
                "fontweight": "bold",
                "linespacing": 2.1,   # more vertical spacing in each cell
                "va": "center",
            },
            yticklabels=row_labels if col == 0 else False,
            xticklabels=False,
            linewidths=0.3,
            linecolor="black",
        )

        cbar = hm.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(cbar_label_fontsize)
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

        ax.set_title(titles[j], fontsize=title_fontsize)
        if col == 0:
            ax.tick_params(axis="y", labelsize=ytick_fontsize)
        else:
            ax.tick_params(axis="y", left=False)

    png_path = output_dir / f"{fig_name}.png"
    svg_path = output_dir / f"{fig_name}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)














def _predictor_x_label_for_log1p_delta(
    col: str,
    label_style: str = "latex",
    predictor_label_map: dict | None = None,
    idx_sub: tuple[str, str] = ("b", "v"),
) -> str:
    """
    Map a predictor column name to an x-axis label for the log1p(|Δ|) vs predictor plots.

    If predictor_label_map is provided and contains the column, its value is used verbatim.
    Otherwise, fall back to simple LaTeX / plain-text heuristics, with special-cases for
    the nominal dose and nominal dose gradient used in the paper.
    """
    if predictor_label_map is not None and col in predictor_label_map:
        return predictor_label_map[col]

    b, v = idx_sub

    # Plain-text style: just echo the column name
    if label_style != "latex":
        return col

    base = col.strip()
    base_lower = base.lower()

    # Nominal dose gradient at voxel - reuse the same helper used for Grad[...] plots
    if "grad" in base_lower:
        # This keeps the notation consistent with the other gradient figures
        return _x_label_for("Grad[nominal] (Gy/mm)", label_style, grad_stat_tex=None)

    # Nominal dose at voxel
    if base_lower.startswith("nominal dose"):
        # D^{(0)}_{b,v} (Gy)
        return rf"$D^{{(0)}}_{{{b},{v}}}\ \mathrm{{(Gy)}}$"

    # Generic fallback: wrap the column name in \mathrm{⋯}, escaping underscores/carets
    safe = re.sub(r"([_^])", r"\\\1", base)
    return rf"$\mathrm{{{safe}}}$"


def plot_log1p_median_delta_vs_predictors_pkg(
    long_df: pd.DataFrame,
    save_dir,
    file_prefix: str,
    *,
    predictor_cols: Sequence[str],
    # scatter / regression options
    scatter: bool = True,
    scatter_sample: int | None = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 10.0,
    ci: int = 95,
    annotate_stats: bool = True,
    write_stats_csv: bool = True,
    # layout / fonts
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    legend_fontsize: int = 11,
    height: float = 3.0,
    aspect: float = 1.6,
    facet_cols: int = 2,
    title: str | None = None,
    # label style
    label_style: str = "latex",
    idx_sub: tuple[str, str] = ("b", "v"),
    j_symbol: str = "Q50",
    predictor_label_map: dict | None = None,
):
    """
    Multi-panel figure: log1p(|Δ|) for the median bias vs a set of voxel-level predictors.

    Parameters
    ----------
    long_df
        Long-form delta design table with at least:
          ['Delta kind', 'log1p|Delta|', <predictor columns>].
        Typically the output of make_long_delta_design_from_delta_design(...).
    save_dir
        Directory where the SVG/PNG (and optional CSV of regression stats) will be written.
    file_prefix
        Base name for output files, e.g. "log1p_median_delta_vs_top_predictors".
    predictor_cols
        Iterable of predictor column names to facet over (e.g. nominal dose, nominal
        dose gradient, best distance-based predictor, best radiomic predictor).
        Usually you will pass four names to obtain a 2×2 layout.

    Returns
    -------
    svg_path, png_path, csv_path, stats_df
    """
    df = long_df.copy()

    # Filter to the median delta kind only
    if "Delta kind" not in df.columns:
        raise KeyError("Expected a 'Delta kind' column in long_df.")
    df["__kind__"] = df["Delta kind"].map(_kind_prefix)
    df = df[df["__kind__"] == "Δ_median"].copy()
    if df.empty:
        raise ValueError("No rows left after filtering to Δ_median.")

    y_col = "log1p|Delta|"
    if y_col not in df.columns:
        raise KeyError("Expected a 'log1p|Delta|' column in long_df.")

    # Ensure predictors are present
    if predictor_cols is None or len(predictor_cols) == 0:
        raise ValueError("predictor_cols must be a non-empty sequence of column names.")
    predictor_cols = list(predictor_cols)
    missing = [c for c in predictor_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Predictor columns not found in long_df: {missing}")

    # Layout
    n = len(predictor_cols)
    ncols = min(facet_cols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(aspect * height * ncols, height * nrows),
        squeeze=False,
    )
    sns.set(style="whitegrid")

    # y-axis uses the same notation helper as the other delta plots
    y_label = _y_label_for(y_col, label_style, j_symbol, idx_sub)

    stats_rows: list[dict] = []

    for i, col in enumerate(predictor_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        pane = df[[col, y_col]].copy()
        pane = pane.rename(columns={col: "X", y_col: "Y"}).dropna()
        if pane.empty:
            ax.set_visible(False)
            continue

        # Single "group" for reuse of the existing panel helper
        pane["Group"] = "Δ_median"

        # Scatter + OLS + CI via the shared helper
        _panel_regplot(
            ax,
            pane,
            xcol="X",
            ycol="Y",
            hue="Group",
            ci=ci,
            scatter=scatter,
            scatter_sample=scatter_sample,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
        )

        # Axis labels
        ax.set_xlabel(
            _predictor_x_label_for_log1p_delta(
                col,
                label_style=label_style,
                predictor_label_map=predictor_label_map,
                idx_sub=idx_sub,
            ),
            fontsize=axes_label_fontsize,
        )
        ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        # Legend: there is a single entry ("Δ_median"); keep it small or hide entirely
        leg = ax.legend(title=None, frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(max(legend_fontsize - 1, 8))

        # Per-panel regression stats (slope, CI, ρ, R², n)
        s = _ols_stats(pane["X"], pane["Y"])
        s_row = {"predictor_col": col, "y_col": y_col, "delta_kind": "Δ_median", **s}
        stats_rows.append(s_row)

        if annotate_stats and not np.isnan(s.get("slope", np.nan)):
            txt = (
                f"slope={s['slope']:.3f} "
                f"[{s['slope_lo']:.3f},{s['slope_hi']:.3f}]\n"
                f"ρ={s['rho']:.3f}, R²={s['r2']:.3f}, n={s['n']}"
            )
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=max(legend_fontsize - 1, 9),
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, lw=0),
            )

    # Hide any unused subplots
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=axes_label_fontsize)
        fig.subplots_adjust(top=0.90)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{file_prefix}.svg"
    png = out_dir / f"{file_prefix}.png"
    fig.tight_layout()
    fig.savefig(svg, format="svg", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)

    # Optional CSV of stats
    stats_df = pd.DataFrame(stats_rows)
    csv_path = None
    if write_stats_csv and not stats_df.empty:
        csv_path = out_dir / f"{file_prefix}__stats.csv"
        stats_df.to_csv(csv_path, index=False)

    return svg, png, csv_path, stats_df






def plot_delta_vs_predictors_pkg(
    long_df: pd.DataFrame,
    save_dir,
    file_prefix: str,
    *,
    predictor_cols: Sequence[str],
    # NEW: general y control
    y_col: str = "log1p|Delta|",
    delta_kind_label: str | None = "Δ_median",     # e.g. "Δ_median", "Δ_mean", "Δ_mode", or None for no filter
    delta_kind_col: str = "Delta kind",
    # scatter / regression options
    scatter: bool = True,
    scatter_sample: int | None = 20000,
    scatter_alpha: float = 0.15,
    scatter_size: float = 10.0,
    ci: int = 95,
    annotate_stats: bool = True,
    write_stats_csv: bool = True,
    # layout / fonts
    axes_label_fontsize: int = 13,
    tick_label_fontsize: int = 11,
    legend_fontsize: int = 11,
    height: float = 3.0,
    aspect: float = 1.6,
    facet_cols: int = 2,
    title: str | None = None,
    # label style
    label_style: str = "latex",
    idx_sub: tuple[str, str] = ("b", "v"),
    j_symbol: str = "Q50",
    predictor_label_map: dict | None = None,
):
    """
    Multi-panel figure: (general) y-column vs a set of voxel-level predictors.

    Parameters
    ----------
    long_df
        Long-form delta design table with at least:
          [delta_kind_col, y_col, <predictor columns>].
        Typically the output of make_long_delta_design_from_delta_design(...).
    save_dir
        Directory where the SVG/PNG (and optional CSV of regression stats) will be written.
    file_prefix
        Base name for output files, e.g. "median_absDelta_vs_top_predictors".
    predictor_cols
        Iterable of predictor column names to facet over.
    y_col
        Name of the column to use on the y-axis (e.g. "log1p|Delta|" or
        something like "|Delta|", "Delta (Gy) deltas nominal_minus_q50 abs", etc.).
    delta_kind_label
        Display label used by _kind_prefix (e.g. "Δ_median"). If not None,
        rows are filtered to this kind. If None, no Delta kind filtering is applied.
    delta_kind_col
        Name of the column containing the delta kind, default "Delta kind".

    Returns
    -------
    svg_path, png_path, csv_path, stats_df
    """
    df = long_df.copy()

    # Optionally filter to a specific delta kind (default: Δ_median, same as before)
    if delta_kind_label is not None:
        if delta_kind_col not in df.columns:
            raise KeyError(f"Expected a '{delta_kind_col}' column in long_df.")
        df["__kind__"] = df[delta_kind_col].map(_kind_prefix)
        df = df[df["__kind__"] == delta_kind_label].copy()
        if df.empty:
            raise ValueError(
                f"No rows left after filtering to delta kind '{delta_kind_label}'."
            )

    # y column
    if y_col not in df.columns:
        raise KeyError(f"Expected a '{y_col}' column in long_df.")
    y_label = _y_label_for(y_col, label_style, j_symbol, idx_sub)

    # Ensure predictors are present
    if not predictor_cols:
        raise ValueError("predictor_cols must be a non-empty sequence of column names.")
    predictor_cols = list(predictor_cols)
    missing = [c for c in predictor_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Predictor columns not found in long_df: {missing}")

    # Layout
    n = len(predictor_cols)
    ncols = min(facet_cols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(aspect * height * ncols, height * nrows),
        squeeze=False,
    )
    sns.set(style="whitegrid")

    stats_rows: list[dict] = []

    for i, col in enumerate(predictor_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        pane = df[[col, y_col]].copy()
        pane = pane.rename(columns={col: "X", y_col: "Y"}).dropna()
        if pane.empty:
            ax.set_visible(False)
            continue

        # Single "group" for reuse of the existing panel helper
        group_label = delta_kind_label if delta_kind_label is not None else "all"
        pane["Group"] = group_label

        # Scatter + OLS + CI via the shared helper
        _panel_regplot(
            ax,
            pane,
            xcol="X",
            ycol="Y",
            hue="Group",
            ci=ci,
            scatter=scatter,
            scatter_sample=scatter_sample,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
        )

        # Axis labels
        ax.set_xlabel(
            _predictor_x_label_for_log1p_delta(  # still OK: only depends on predictor
                col,
                label_style=label_style,
                predictor_label_map=predictor_label_map,
                idx_sub=idx_sub,
            ),
            fontsize=axes_label_fontsize,
        )
        ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        # Legend: single entry; keep small
        leg = ax.legend(title=None, frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(max(legend_fontsize - 1, 8))

        # Per-panel regression stats (slope, CI, ρ, R², n)
        s = _ols_stats(pane["X"], pane["Y"])
        s_row = {
            "predictor_col": col,
            "y_col": y_col,
            "delta_kind": group_label,
            **s,
        }
        stats_rows.append(s_row)

        if annotate_stats and not np.isnan(s.get("slope", np.nan)):
            txt = (
                f"slope={s['slope']:.3f} "
                f"[{s['slope_lo']:.3f},{s['slope_hi']:.3f}]\n"
                f"ρ={s['rho']:.3f}, R²={s['r2']:.3f}, n={s['n']}"
            )
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=max(legend_fontsize - 1, 9),
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, lw=0),
            )

    # Hide any unused subplots
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=axes_label_fontsize)
        fig.subplots_adjust(top=0.90)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{file_prefix}.svg"
    png = out_dir / f"{file_prefix}.png"
    fig.tight_layout()
    fig.savefig(svg, format="svg", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)

    # Optional CSV of stats
    stats_df = pd.DataFrame(stats_rows)
    csv_path = None
    if write_stats_csv and not stats_df.empty:
        csv_path = out_dir / f"{file_prefix}__stats.csv"
        stats_df.to_csv(csv_path, index=False)

    return svg, png, csv_path, stats_df



# this one is generalized to allow multiple delta kinds in one plot
def plot_delta_vs_predictors_pkg_generalized(
        long_df: pd.DataFrame,
        save_dir,
        file_prefix: str,
        *,
        predictor_cols: Sequence[str],
        y_col: str = "log1p|Delta|",
        # can be "Δ_median" or ["Δ_median", "Δ_mean", "Δ_mode"]
        delta_kind_label: str | Sequence[str] | None = "Δ_median",
        delta_kind_col: str = "Delta kind",
        # scatter / regression options
        scatter: bool = True,
        scatter_sample: int | None = 20000,
        scatter_alpha: float = 0.15,
        scatter_size: float = 10.0,
        ci: int = 95,
        annotate_stats: bool = True,
        write_stats_csv: bool = True,
        # layout / fonts
        axes_label_fontsize: int = 13,
        tick_label_fontsize: int = 11,
        legend_fontsize: int = 11,
        height: float = 3.0,
        aspect: float = 1.6,
        facet_cols: int = 2,
        title: str | None = None,
        # label style
        label_style: str = "latex",
        idx_sub: tuple[str, str] = ("b", "v"),
        j_symbol: str = "Q50",
        predictor_label_map: dict | None = None,
        delta_kind_label_map = None,
        zero_x_predictors: Sequence[str] = (
            "Nominal dose (Gy)",
            "Nominal dose grad (Gy/mm)",
            # maybe distance, maybe not
        ),
    ):
    df = long_df.copy()

    # ---------- handle delta kinds ----------
    if delta_kind_label is not None:
        if delta_kind_col not in df.columns:
            raise KeyError(f"Expected a '{delta_kind_col}' column in long_df.")

        # map raw kind -> pretty label (e.g. "median" -> "Δ_median")
        df["__kind__"] = df[delta_kind_col].map(_kind_prefix)

        if isinstance(delta_kind_label, str):
            wanted = [delta_kind_label]
        else:
            wanted = list(delta_kind_label)

        df = df[df["__kind__"].isin(wanted)].copy()
        if df.empty:
            raise ValueError(
                f"No rows left after filtering to delta kinds {wanted!r}."
            )
    else:
        # no filtering, but still build a label column if possible
        if delta_kind_col in df.columns:
            df["__kind__"] = df[delta_kind_col].map(_kind_prefix)
        else:
            df["__kind__"] = "all"

    # ---------- y axis ----------
    if y_col not in df.columns:
        raise KeyError(f"Expected a '{y_col}' column in long_df.")
    y_label = _y_label_for(y_col, label_style, j_symbol, idx_sub)

    # ---------- predictor checks ----------
    if not predictor_cols:
        raise ValueError("predictor_cols must be a non-empty sequence of column names.")
    predictor_cols = list(predictor_cols)
    missing = [c for c in predictor_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Predictor columns not found in long_df: {missing}")

    # ---------- layout ----------
    n = len(predictor_cols)
    ncols = min(facet_cols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(aspect * height * ncols, height * nrows),
        squeeze=False,
    )
    sns.set(style="whitegrid")

    stats_rows: list[dict] = []

    # before the plotting loop
    same_y_range = True
    if same_y_range:
        y_max = float(df[y_col].max())
        # round up to a nice number, e.g. nearest 10
        y_max = np.ceil(y_max / 10) * 10


    for i, col in enumerate(predictor_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        # keep predictor, y, and kind
        pane = df[[col, y_col, "__kind__"]].copy()
        pane = pane.rename(
            columns={col: "X", y_col: "Y", "__kind__": "Group"}
        ).dropna()

        if pane.empty:
            ax.set_visible(False)
            continue

        # draw all groups in one call
        _panel_regplot(
            ax,
            pane,
            xcol="X",
            ycol="Y",
            hue="Group",  # <-- multiple bias types distinguished here
            ci=ci,
            scatter=scatter,
            scatter_sample=scatter_sample,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
        )

        if same_y_range:
            ax.set_ylim(0, y_max)
        else:
            ax.set_ylim(bottom=0)  # at least anchor at 0

        if col in zero_x_predictors:
            ax.set_xlim(left=0)

        # labels
        ax.set_xlabel(
            _predictor_x_label_for_log1p_delta(
                col,
                label_style=label_style,
                predictor_label_map=predictor_label_map,
                idx_sub=idx_sub,
            ),
            fontsize=axes_label_fontsize,
        )
        ax.set_ylabel(y_label, fontsize=axes_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        # legend: now has one entry per bias type
        leg = ax.legend(title=None, frameon=True)
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(max(legend_fontsize - 1, 8))
                ## map the legend text (each entry) to custom labels
                if delta_kind_label_map is not None:
                    txt_str = txt.get_text()
                    if txt_str in delta_kind_label_map:
                        txt.set_text(delta_kind_label_map[txt_str])

        # ---------- per-group stats ----------
        group_stats: list[tuple[str, dict]] = []
        for gname, sub in pane.groupby("Group", observed=True, sort=False):
            s = _ols_stats(sub["X"], sub["Y"])
            group_stats.append((gname, s))
            stats_rows.append(
                {
                    "predictor_col": col,
                    "y_col": y_col,
                    "delta_kind": gname,
                    **s,
                }
            )

        if annotate_stats:
            y0 = 0.98
            dy = 0.14  # move text down a bit for each group
            for jg, (gname, s) in enumerate(group_stats):
                if np.isnan(s.get("slope", np.nan)):
                    continue
                txt = (
                    f"{gname}: slope={s['slope']:.3f} "
                    f"[{s['slope_lo']:.3f},{s['slope_hi']:.3f}]\n"
                    f"ρ={s['rho']:.3f}, R²={s['r2']:.3f}, n={s['n']}"
                )
                ax.text(
                    0.02,
                    y0 - jg * dy,
                    txt,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=max(legend_fontsize - 1, 9),
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, lw=0),
                )

    # hide unused axes
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=axes_label_fontsize)
        fig.subplots_adjust(top=0.90)

    ## add major gridlines
    for ax in axes.flat:
        ax.grid(visible=True, which="major", linestyle="-", linewidth=0.5)
    ## add minor gridlines
        ax.minorticks_on()
        ax.grid(visible=True, which="minor", linestyle="--", linewidth=0.25, alpha=0.7)

    """
    #set x limit to start at 0, and y limit to for from 0 to max y range + 10% for all plots
    for ax in axes.flat:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    """

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg = out_dir / f"{file_prefix}.svg"
    png = out_dir / f"{file_prefix}.png"
    pdf = out_dir / f"{file_prefix}.pdf"      # <--- NEW

    fig.tight_layout()

    # high-quality exports
    for path, fmt, dpi in [
        (svg, "svg", 300),
        (png, "png", 300),
        (pdf, "pdf", 600),   # 600 dpi for raster bits in the PDF
    ]:
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.close(fig)

    stats_df = pd.DataFrame(stats_rows)
    csv_path = None
    if write_stats_csv and not stats_df.empty:
        csv_path = out_dir / f"{file_prefix}__stats.csv"
        stats_df.to_csv(csv_path, index=False)

    return svg, png, pdf, csv_path, stats_df   # <--- include pdf in return
