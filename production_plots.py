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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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



def plot_eff_size_heatmaps(eff_size_df, 
                           patient_id_col, 
                           bx_index_col, 
                           bx_id_col, 
                           eff_size_col, 
                           eff_size_type, 
                           save_dir=None,
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
            base_file_name = (f"EffSize_{eff_size_type}_Heatmap_{patient_id_col}_{patient_id}_"
                              f"{bx_index_col}_{bx_index}_{bx_id_col}_{bx_id}")
            png_path = os.path.join(save_dir, f"{base_file_name}.png")
            svg_path = os.path.join(save_dir, f"{base_file_name}.svg")
            
            plt.savefig(png_path)
            plt.savefig(svg_path)
            print(f"Saved heatmap as PNG: {png_path}")
            print(f"Saved heatmap as SVG: {svg_path}")
        
        # Close the plot to avoid displaying in non-interactive environments
        plt.close()





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
        p = save_path_base.joinpath(f"{eff_size_type}_boxed_counts_abs-{aggregate_abs}")
        plt.savefig(p.with_suffix(".png"), bbox_inches="tight")
        plt.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        print(f"Saved boxed-counts heatmap to {p}.png/.svg")
    else:
        plt.show()
    plt.close()





def dvh_boxplot(cohort_bx_dvh_metrics_df, save_path=None, custom_name=None):
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
        plt.title(f'Boxplot of (MC) Mean Values for {metric_type} Metrics', fontsize=14)
        plt.suptitle('')
        plt.xlabel('DVH Metric')
        if 'V' in metric_type:
            plt.ylabel('Percent volume')
        elif 'D' in metric_type:
            plt.ylabel('Dose (Gy)')
        plt.xticks(rotation=45)

        # After plotting
        xtick_labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]
        custom_labels = [convert_metric_label(label) for label in xtick_labels]
        plt.gca().set_xticklabels(custom_labels, rotation=45)

        # remove vertical grid lines
        plt.gca().yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().xaxis.grid(False)  # Disable vertical grid lines

        # start y axis range at 0 if metric is dose
        if 'D' in metric_type:
            # Set y-axis limits to start at 0 for dose metrics
            plt.gca().set_ylim(bottom=0)
        else:
            # For volume metrics, set y-axis limits to always range from 0 toi 100
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


