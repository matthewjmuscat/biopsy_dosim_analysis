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
                                                                                 y_axis_label):
    # plotting function
    def plot_quantile_regression_and_more_corrected(df, df_voxelized, sp_patient_all_structure_shifts_pandas_data_frame, patientUID, bx_id, bx_struct_ind, bx_ref):
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Placeholder dictionaries for regression results
        y_regressions = {}

        # Function to perform and plot kernel regression
        def perform_and_plot_kernel_regression(x, y, x_range, label, color, annotation_text = None, target_offset=0):
            kr = KernelReg(endog=y, exog=x, var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            plt.plot(x_range, y_kr, label=label, color=color, linewidth=2)
            
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

        # Perform kernel regression for each quantile and store the y-values
        for quantile in [0.05, 0.25, 0.75, 0.95]:
            q_df = df.groupby('Z (Bx frame)')[value_col_key].quantile(quantile).reset_index()
            kr = KernelReg(endog=q_df[value_col_key], exog=q_df['Z (Bx frame)'], var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            y_regressions[quantile] = y_kr

        # Filling the space between the quantile lines
        plt.fill_between(x_range, y_regressions[0.05], y_regressions[0.25], color='springgreen', alpha=1)
        plt.fill_between(x_range, y_regressions[0.25], y_regressions[0.75], color='dodgerblue', alpha=1)
        plt.fill_between(x_range, y_regressions[0.75], y_regressions[0.95], color='springgreen', alpha=1)
        
        # Additional plot enhancements
        # Plot line for MC trial num = 0
        # Kernel regression for MC trial num = 0 subset
        
        mc_trial_0 = df[df['MC trial num'] == 0]
        perform_and_plot_kernel_regression(mc_trial_0['Z (Bx frame)'], mc_trial_0[value_col_key], x_range, 'Nominal', 'red')
        

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
        
        perform_and_plot_kernel_regression(z_vals, kde_max_doses, x_range, 'KDE Max Density Dose', 'magenta')
        perform_and_plot_kernel_regression(z_vals, mean_doses, x_range, 'Mean Dose', 'orange')

        

        # Line plot for each trial
        """
        num_mc_trials_plus_nom = df_voxelized['MC trial num'].nunique()
        for trial in range(1,num_mc_trials_plus_nom):
            df_sp_trial = df[df["MC trial num"] == trial].sort_values(by='Z (Bx frame)') # sorting is to make sure that the lines are drawn properly
            df_z_simple = df_sp_trial.drop_duplicates(subset=['Z (Bx frame)'], keep='first') # remove points that have the same z value so that the line plots look better
            #plt.plot(df_z_simple['Z (Bx frame)'], df_z_simple[value_col_key], color='grey', alpha=0.1, linewidth=1, zorder = 0.9)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!
            plt.scatter(
                df_z_simple['Z (Bx frame)'], 
                df_z_simple[value_col_key], 
                color='grey', 
                alpha=0.1, 
                s=10,  # Size of dots, adjust as needed
                zorder=0.9
            )
        """

        ## Instead want to show regressions of random trials so that we can appreciate structure
        annotation_offset_index = 0
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

            annotation_text_for_trial = f"({x_dist:.1f},{y_dist:.1f},{z_dist:.1f}), d = {d_tot:.1f}"
            
            perform_and_plot_kernel_regression(mc_trial['Z (Bx frame)'], mc_trial[value_col_key], x_range, f"Trial: {trial}", 'gray', annotation_text = annotation_text_for_trial, target_offset=annotation_offset_index)
            
            plt.scatter(
                mc_trial_voxelized['Z (Bx frame)'], 
                mc_trial_voxelized[value_col_key], 
                color='grey', 
                alpha=0.1, 
                s=10,  # Size of dots, adjust as needed
                zorder=1.1
            )
            annotation_offset_index += 1
        """
        for trial in range(1,num_rand_trials_to_show):
            df_sp_trial = df_voxelized[df_voxelized["MC trial num"] == trial]
            plt.plot(df_sp_trial['Z (Bx frame)'], df_sp_trial[value_col_key], color='grey', alpha=0.1, linewidth=1, zorder = 1.1)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!
        """

        ax = plt.gca() 

        # Final plot adjustments
        ax.set_title(f'{patientUID} - {bx_id} - Dosimetry Regression',
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


        ax.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal', 'Max density dose', 'Mean dose'],loc='best', facecolor='white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes

        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()


        
        
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
            