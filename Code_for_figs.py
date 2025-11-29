# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:37:49 2025

@author: kimma
"""


import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

p_dir = "C:/Users/kimma/SynologyDrive/Code/Python/scLENS-py/Source_data/"
# %% Figure 2

file_path = p_dir + "all_info.csv"
t_df = pd.read_csv(file_path)

si = t_df['D_multiK_new'].argsort()[::-1]
t_df_sorted = t_df.iloc[si] 

multiK_cols = ['multiK_wscLENS', 'multiK', 'multiK_wscanpy', 'multiK_wscVI', 'multiK_welbow']
chooseR_cols = ['chooseR_wscLENS', 'chooseR', 'chooseR_wscanpy', 'chooseR_wscVI', 'chooseR_welbow']

multiK_score_hc = t_df_sorted[multiK_cols].values
chooseR_score_hc = t_df_sorted[chooseR_cols].values
std_multiK_score = np.std(multiK_score_hc, axis=0)
mean_multiK_score = np.mean(multiK_score_hc, axis=0)
std_chooseR_score = np.std(chooseR_score_hc, axis=0)
mean_chooseR_score = np.mean(chooseR_score_hc, axis=0)

print("=== All Data % Difference (vs Original) ===")
print(((mean_chooseR_score - mean_chooseR_score[1]) / mean_chooseR_score[1]) * 100)
print(((mean_multiK_score - mean_multiK_score[1]) / mean_multiK_score[1]) * 100)

groups = list(t_df.groupby('cat'))
if len(groups) > 2:
    tmp_df = groups[2][1]
    
    sub_multiK_score_hc = tmp_df[multiK_cols].values
    sub_chooseR_score_hc = tmp_df[chooseR_cols].values
    
    sub_std_multiK_score = np.std(sub_multiK_score_hc, axis=0)
    sub_mean_multiK_score = np.mean(sub_multiK_score_hc, axis=0)
    
    sub_std_chooseR_score = np.std(sub_chooseR_score_hc, axis=0)
    sub_mean_chooseR_score = np.mean(sub_chooseR_score_hc, axis=0)

    print("\n=== Subgroup Data % Difference (vs Original) ===")
    print(((sub_mean_chooseR_score - sub_mean_chooseR_score[1]) / sub_mean_chooseR_score[1]) * 100)
    print(((sub_mean_multiK_score - sub_mean_multiK_score[1]) / sub_mean_multiK_score[1]) * 100)

wong_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
labels_list = ["With scLENS", "Original", "PC50", "with scVI", "Elbow"]
dataset_names = t_df_sorted['name'].values

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[8, 1.5, 0.5], wspace=0.1, hspace=0.3)

def plot_row(row_idx, score_matrix, mean_score, std_score, title, ax_scatter, ax_bar):
    n_datasets, n_methods = score_matrix.shape
    
    y_positions = np.arange(n_methods)[::-1] 
    
    for i in range(n_methods):
        x_coords = np.arange(n_datasets)
        y_coords = np.full(n_datasets, y_positions[i])
        sizes = score_matrix[:, i] * 100 
        
        ax_scatter.scatter(x_coords, y_coords, s=sizes, color=wong_colors[i], alpha=0.8, edgecolors='none')

    ax_scatter.set_title(title, fontsize=18, fontweight='bold', loc='left')
    ax_scatter.set_xlim(-1, n_datasets)
    ax_scatter.set_ylim(-0.5, n_methods - 0.5)
    
    ax_scatter.set_xticks(np.arange(n_datasets))
    if row_idx == 1: 
        ax_scatter.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=10)
    else:
        ax_scatter.set_xticklabels([]) 
    
    ax_scatter.set_yticks(y_positions)
    ax_scatter.set_yticklabels(labels_list, fontsize=12)
    ax_scatter.grid(True, axis='x', linestyle='--', alpha=0.3)

    x_pos = np.arange(n_methods)
    ax_bar.bar(x_pos, mean_score, yerr=std_score, 
               color=[wong_colors[i] for i in range(n_methods)], 
               capsize=5, width=0.6)
    
    ax_bar.set_ylim(0, 1.1)
    
    ax_bar.set_xticks([]) 
    
    ax_bar.set_ylabel("Avg Score", fontsize=12)
    ax_bar.yaxis.set_label_position("right") 
    ax_bar.yaxis.tick_right() 
    
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
plot_row(0, chooseR_score_hc, mean_chooseR_score, std_chooseR_score, "ChooseR", ax1, ax2)

ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
plot_row(1, multiK_score_hc, mean_multiK_score, std_multiK_score, "MultiK", ax3, ax4)

ax_leg = fig.add_subplot(gs[:, 2])
ax_leg.axis('off')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=10)
    for label, color in zip(labels_list, wong_colors[:5])
]

ax_leg.legend(handles=legend_elements, title="Methods", loc='center left', fontsize=12)

# plt.tight_layout()
# output_path = "C:/Users/kimma/Downloads/figure2_b_python.svg"
# plt.savefig(output_path, format='svg', bbox_inches='tight')
# print(f"Figure saved to {output_path}")
plt.show()

# %% Figure 3
file_path = p_dir + "all_info.csv"
t_df = pd.read_csv(file_path)

test_df = t_df.copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.rcParams.update({'font.size': 14}) 

raw_size = np.abs(test_df['Number_of_sigs'] - 30) * 0.5 + 20
marker_sizes = raw_size ** 2 * 0.8  

scatter_kwargs = {
    's': marker_sizes,
    'cmap': 'RdBu_r',  
    'alpha': 0.6,
    'edgecolors': 'black', 
    'linewidths': 1.0      
}

ax1 = axes[0]
sc1 = ax1.scatter(test_df['sparsity'], test_df['CV_tgc'],
                  c=test_df['pure_score'],
                  vmin=-0.1, vmax=0.1,  
                  **scatter_kwargs)

ax1.set_title("Embedding quality improvement", fontsize=20, pad=15)
ax1.set_xlabel("Sparsity", fontsize=16)
ax1.set_ylabel("CV(TGC)", fontsize=16)

cbar1 = fig.colorbar(sc1, ax=ax1)
cbar1.set_label("NP", fontsize=16)
cbar1.ax.tick_params(labelsize=12)

ax2 = axes[1]
sc2 = ax2.scatter(test_df['sparsity'], test_df['CV_tgc'],
                  c=test_df['D_multiK'],
                  vmin=-0.2, vmax=0.2,  
                  **scatter_kwargs)

ax2.set_title("MultiK's Performance gap", fontsize=20, pad=15)
ax2.set_xlabel("Sparsity", fontsize=16)
ax2.set_ylabel("CV(TGC)", fontsize=16)

cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label("ΔECS", fontsize=16)
cbar2.ax.tick_params(labelsize=12)

plt.tight_layout()
# output_path = "C:/Users/kimma/Downloads/embedding_quality_multik_gap.svg"
# plt.savefig(output_path, format='svg', bbox_inches='tight')
# print(f"Figure saved to {output_path}")
plt.show()
# %% Supplementary Fig. 1
test_dir = os.path.join(p_dir, "final_label")
tmp_files = glob(os.path.join(test_dir, "*.csv"))
ecs_file_path = p_dir+"ecs_score.csv"


print(f"Found {len(tmp_files)} label files in {test_dir}")
name_mapping = {
    'scLENS_chooseR': 'ChooseR + scLENS',
    'scLENS_multiK': 'MultiK + scLENS',
    'scVI_chooseR': 'ChooseR + scVI',
    'scVI_multiK': 'MultiK + scVI',
    'elbow_chooseR': 'ChooseR + elbow',
    'elbow_multiK': 'MultiK + elbow',
    'Scanpy_chooseR': 'ChooseR + Scanpy',
    'Scanpy_multiK': 'MultiK + Scanpy',
    'chooseR': 'ChooseR',
    'multiK': 'MultiK'
}

raw_results = []

for f in tmp_files:
    try:
        df = pd.read_csv(f)
        if 'cell_type' not in df.columns: continue
            
        ground_truth = df['cell_type']
        method_cols = [c for c in df.columns if c != 'cell_type']
        
        for method in method_cols:
            pred_labels = df[method]
            valid_idx = ground_truth.notna() & pred_labels.notna()
            if not valid_idx.any(): continue
                
            gt = ground_truth[valid_idx]
            pred = pred_labels[valid_idx]

            metrics = {
                'ARI': adjusted_rand_score(gt, pred),
                'NMI': normalized_mutual_info_score(gt, pred),
                'AMI': adjusted_mutual_info_score(gt, pred)
            }
            
            mapped_name = name_mapping.get(method, method)
            
            for m_name, score in metrics.items():
                raw_results.append({
                    'Method': mapped_name,
                    'Metric': m_name,
                    'Score': score
                })
            
    except Exception as e:
        print(f"Error processing {f}: {e}")

df_raw = pd.DataFrame(raw_results)
if not df_raw.empty:
    df_summary_others = df_raw.groupby(['Method', 'Metric'])['Score'].agg(['mean', 'std']).reset_index()
else:
    df_summary_others = pd.DataFrame(columns=['Method', 'Metric', 'mean', 'std'])

df_ecs_summary = pd.DataFrame()

if os.path.exists(ecs_file_path):
    try:
        df_ecs = pd.read_csv(ecs_file_path)
        
        if 'packages' in df_ecs.columns:
            df_ecs = df_ecs.rename(columns={'packages': 'Method'})
        
        df_ecs['Metric'] = 'ECS'
        
        cols_to_keep = ['Method', 'Metric', 'mean', 'std']
        df_ecs_summary = df_ecs[[c for c in cols_to_keep if c in df_ecs.columns]]
        
    except Exception as e:
        print(f"Error loading ECS: {e}")
else:
    print(f"Warning: ECS file not found at {ecs_file_path}")

df_final = pd.concat([df_summary_others, df_ecs_summary], ignore_index=True)

if not df_final.empty:
    sort_metric = 'ECS' if 'ECS' in df_final['Metric'].unique() else 'ARI'
    
    order_df = df_final[df_final['Metric'] == sort_metric].sort_values(by='mean', ascending=False)
    method_order = order_df['Method'].tolist()
    
    metric_order = ['ARI', 'NMI', 'AMI', 'ECS']
    metric_order = [m for m in metric_order if m in df_final['Metric'].unique()]
    
    df_mean = df_final.pivot(index='Method', columns='Metric', values='mean')
    df_std = df_final.pivot(index='Method', columns='Metric', values='std')
    
    existing_order = [m for m in method_order if m in df_mean.index]
    df_mean = df_mean.reindex(existing_order)
    df_std = df_std.reindex(existing_order)
    
    df_mean = df_mean[metric_order]
    df_std = df_std[metric_order]
    
    plt.figure(figsize=(12, 6))
    
    ax = df_mean.plot(kind='bar', yerr=df_std, capsize=4, figsize=(13, 6), width=0.8, rot=45, colormap='viridis')
    
    plt.title('Comparison of Clustering Performance Metrics', fontsize=13)
    plt.xlabel('', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    
    plt.legend(title='Metric', loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=12)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    highlight_labels = {"MultiK + scLENS", "ChooseR + scLENS"}
    for label in ax.get_xticklabels():
        if label.get_text() in highlight_labels:
            label.set_fontweight('bold')
            label.set_color('#333333') 

    plt.tight_layout()
    plt.show()
    
    print("\n=== Final Summary Table ===")

else:
    print("No data available to plot.")
    
# %% Supplementary Fig. 2
file_path = p_dir + "marker_o.csv"
final_df = pd.read_csv(file_path,index_col=0)

id_col = 'data'  
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

comparisons = [
    ('MultiK Comparison', ['MultiK', 'MultiK + scLENS'], axes[0]),
    ('ChooseR Comparison', ['ChooseR', 'ChooseR + scLENS'], axes[1])
]

box_palette = {
    'MultiK'            : 'tab:blue',
    'MultiK + scLENS'   : 'tab:red',
    'ChooseR'           : 'tab:blue',
    'ChooseR + scLENS'  : 'tab:red'
}

for title, order, ax in comparisons:
    subset = final_df[final_df['Method'].isin(order)].copy()
    
    sns.boxplot(
        data=subset,
        x='Method',
        y='Average_Jaccard_Index',
        order=order,
        ax=ax,
        showfliers=False,
        palette=box_palette   
    )

    wide = subset.pivot_table(
        index=id_col,
        columns='Method',
        values='Average_Jaccard_Index'
    )

    x_positions = np.arange(len(order))

    for _, row in wide.iterrows():
        if row[order].isna().any():
            continue
        ys = row[order].values
        ax.plot(
            x_positions,
            ys,
            marker='o',
            linewidth=0.8,
            alpha=0.6,
            color='black'
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axes[0].set_ylabel('Marker Gene Recovery Score', fontsize=13)
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

