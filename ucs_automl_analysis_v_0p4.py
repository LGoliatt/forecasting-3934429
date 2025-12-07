"""
Model Evaluation Pipeline
=========================

This script processes JSON files from ML experiments. It evaluates model performance,
computes a performance index, conducts statistical analysis (ANOVA, Tukey HSD),
and visualizes uncertainty, parameter tuning, and Taylor diagrams.
"""
import os,sys
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import skill_metrics as sm
from collections import defaultdict, Counter
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_squared_error
from permetrics.regression import RegressionMetric
#%%

sns.set_context("paper")
sns.set_style(style="white", rc={
    #"font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.set_context("paper", font_scale=1.8, 
        rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            'xtick.labelsize':16,'ytick.labelsize':16,
            'font.family':"Times New Roman", }
        ) 
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})

#%%
# --- CONFIGURATION ---
REFERENCE_METRIC = 'RMSE'
N_RUNS = 50
METRICS = ['R2', 'R', 'RMSE', 'MAE', 'MAPE', ]
FOLDER_FIG='./img'

# --- UTILITIES ---
def calculate_taylor_metrics(y_true, y_pred):
    std_pred = np.std(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rms = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return std_pred, corr, rms

# --- 1. LOAD JSON FILES ---
def load_json_data(folder_path):
    results, uncertainty = [], []
    for filepath in glob.glob(os.path.join(folder_path, '*.json')):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)[0]
                y_true, y_pred = data.get("y_test", []), data.get("y_pred", [])
                model_name = data.get("estimator", "unknown")
                print(model_name)
                metrics = {}
                if y_true and y_pred:
                    metric_obj = RegressionMetric(y_true, y_pred)
                    metrics = metric_obj.get_metrics_by_list_names(METRICS)
                    metrics['Model'] = model_name
                    results.append(metrics)
                uncertainty.append({
                    'Model': model_name,
                    'MAD': data.get('uncertainty_mad',-1),
                    'Uncertainty': data.get('uncertainty_measure',-1),
                    REFERENCE_METRIC: metrics.get(REFERENCE_METRIC, None)
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return pd.DataFrame(results), pd.DataFrame(uncertainty)


def filter_models(df, models_to_remove):
    """
    Remove linhas de um DataFrame cujos modelos estão na lista de exclusão.
    A função também remove modelos com o sufixo '-FS'.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com coluna 'Model'
    - models_to_remove (list): Lista de nomes de modelos a remover (ex: ['RF', 'ANN'])

    Retorna:
    - pd.DataFrame filtrado
    """
    # Inclui versões com sufixo '-FS'
    full_models_to_remove = models_to_remove + [m + '-FS' for m in models_to_remove]
    
    # Remove modelos com ou sem sufixo
    return df[~df['Model'].str.split('-').str[0].isin(full_models_to_remove)]


# --- 2. MODEL METRICS PLOT ---
def plot_model_metrics(refname, df, save_fig=False, output_dir='./img'):
    
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_metrics = len(METRICS)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)

    #fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 8))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9,8))
    axes = axes.flatten()  # Para acesso direto aos subplots mesmo se for uma única linha

    for ax, metric in zip(axes, METRICS):
        sorted_df = df.sort_values(by='Model')  # organiza alfabeticamente os modelos
        sns.barplot(x='Model', y=metric, data=sorted_df, ax=ax, palette="viridis")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'{metric} Comparison')
        #ax.grid(linestyle='-')

    # Remove subplots extras se existirem
    for i in range(len(METRICS), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=0.5)

    if save_fig:
        os.system(f"mkdir -p {FOLDER_FIG}")
        filename = f"{BASENAME}_cmp_metrics.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()
    
# --- 3. UNCERTAINTY SCATTER ---
def plot_uncertainty(refname, df, save_fig=False, output_dir='./img'):
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x='Uncertainty', y=REFERENCE_METRIC,
        hue='Model', style='Model', s=200,
        palette='tab10', markers=True, legend=False  # Disable legend
    )
    plt.title(f"{REFERENCE_METRIC} vs Uncertainty")
    plt.grid(True)
    plt.tight_layout()

    model_names= list(df['Model'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

    for i in range(df.shape[0]):
        model = df['Model'].iloc[i]
        x = df['Uncertainty'].iloc[i]
        y = df[REFERENCE_METRIC].iloc[i]
        
        # Randomly choose ha and va
        ha = np.random.choice(['left', 'right'])
        va = np.random.choice(['top', 'bottom'])
        color = color_map[model]
        plt.text(x, y,
                 model,
                 fontsize=12,
                 ha=ha,
                 va=va,
                 alpha=1,
                 color='black',
                 )

    if save_fig:        
        filename = f"{BASENAME}_cmp_uncertainty.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()
    

# --- 4. PERFORMANCE INDEX (PI) ---
def compute_performance_index(refname, df_results, save_fig=False, output_dir='./img'):
    """Compute a weighted Performance Index (PI) and rank models."""
    weight_array = np.array([1/len(METRICS)] * len(METRICS))
    df_normalized = df_results[METRICS].copy()

    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize and invert error metrics so higher is better
    for col in df_normalized.columns:
        df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / \
                             (df_normalized[col].max() - df_normalized[col].min())

    error_metrics = ['RMSE', 'MAE', 'MAPE']
    df_normalized[error_metrics] = 1 - df_normalized[error_metrics]

    # Compute PI
    df_results['PI'] = (df_normalized * weight_array).sum(axis=1)
    df_ranked = df_results.sort_values(by='PI', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = df_ranked.index + 1
    print("\n📊 Ranked Models by Performance Index (PI):")
    print(df_ranked[['Model', 'PI', 'Rank'] + METRICS])
        
    # Group by model and calculate mean PI
    model_stats = df_ranked.groupby('Model')['PI'].agg(['mean', 'std']).reset_index()
    model_stats = model_stats.sort_values(by='mean', ascending=False).reset_index(drop=True)
    model_stats['Rank'] = model_stats.index + 1
    print("\n📊 Mean PI Scores Across Runs:")
    print(model_stats[['Rank', 'Model', 'mean', 'std']])

    pi_scores_df = model_stats[['Rank', 'Model', 'mean', 'std']]
    # Format mean ± std as a single column
    pi_scores_df["PI"] = pi_scores_df.apply(
        lambda row: f"{row['mean']:.3f} (± {row['std']:.3f})", axis=1
    )
    
    # Select only desired columns
    latex_df = pi_scores_df[["Rank", "Model", "PI"]]
    
    # Generate LaTeX table
    latex_pi_path = f"./pi_scores_table_{refname}.tex"
    latex_pi = latex_df.to_latex(
        index=False,
        escape=False,
        caption="Mean PI scores across model runs with standard deviation.",
        label="tab:pi_scores",
        column_format="ccl"
    )
    
    with open(latex_pi_path, "w") as f:
        f.write(latex_pi)
        
    # Plot PI scores
    plt.figure(figsize=(4, 6))
    x = range(len(model_stats))
    y = model_stats['mean']
    yerr = model_stats['std']

    plt.bar(x, y, yerr=yerr, capsize=5, edgecolor='black', alpha=0.8)
    plt.xticks(x, model_stats['Model'], rotation=90)
    #plt.title("Average Performance Index (PI) with Std Dev Across Models")
    plt.title(BASENAME.replace('_','').upper())
    plt.ylabel("Mean PI Score")
    plt.xlabel("Model")
    plt.tight_layout()
    
    
    if save_fig:
        filename = f"{BASENAME}_pi.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()


    return df_ranked


# --- 5. FEATURE SELECTION FREQUENCY ---
def analyze_feature_selection(models_to_remove, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    freq = defaultdict(lambda: defaultdict(int))
    all_features = set()
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('estimator', '')
            if model not in models_to_remove:
                if model.endswith('-FS'):
                    for feat in data.get('selected_features', []):
                        freq[model][feat] += 1
                        all_features.add(feat)
    
    df = pd.DataFrame(freq).fillna(0).reindex(index=sorted(all_features))
    df_pct = df.apply(lambda col: (col / N_RUNS) * 100, axis=0).round(1)
    df_long = df_pct.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Percentage')
    df_long['Feature'] = df_long['Feature'].replace(rename_dict)
    
    # plt.figure(figsize=(7, 4))
    # sns.barplot(data=df_long, x='index', y='Percentage', hue='Feature', palette='tab10')
    # #plt.title("Feature Selection Frequency (\%)")
    # plt.ylabel("Feature Selection Frequency (\%)")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.xlabel(None)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #       ncol=5, 
    #       fancybox=True, shadow=True,
    #       )    

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df_long, x='index', y='Percentage', hue='Feature', palette='tab10')

    # --- Add percentage values on top of bars ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='edge', fontsize=10, padding=3, )
    
    plt.ylabel("Feature Selection Frequency (\%)")
    plt.xticks(rotation=0)
    plt.ylim([0,110])
    plt.xlabel('')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, )
    plt.tight_layout()
    
    if save_fig:
        filename = f"{BASENAME}_fs.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()


# --- 5B. FEATURE PARETO PER MODEL ---
def plot_feature_pareto(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_features = defaultdict(Counter)
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('model_name', '')
            if model.endswith('-FS'):
                short_model = model.split('-')[0].upper()
                for feat in data.get('selected_features', []):
                    model_features[short_model][feat] += 1

    for model, feat_counter in model_features.items():
        df_aux = pd.DataFrame.from_dict(feat_counter, orient='index', columns=['Frequency'])
        df_aux.sort_values(by='Frequency', ascending=False, inplace=True)
        df_aux['Cumulative %'] = df_aux['Frequency'].cumsum() / df_aux['Frequency'].sum() * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df.index, y='Frequency', data=df_aux, ax=ax1, color='skyblue')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Features')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # ax2 = ax1.twinx()
        # ax2.plot(df.index, df['Cumulative %'], color='orange', marker='o', linestyle='--')
        # ax2.set_ylabel('Cumulative %')
        # ax2.set_ylim(0, 105)
        # ax2.axhline(80, linestyle='--', color='gray')

        plt.title(f"Feature Selection Pareto - {model}")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_fig:
            filename = f"{BASENAME}_freq_features.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
        else:
            plt.show()


# --- 6. ANOVA & TUKEY ---
def run_anova_and_tukey(df, metrics=METRICS):
    for metric in METRICS:
        groups = df.groupby('Model')[metric].apply(list).values
        f, p = stats.f_oneway(*groups)
        print(f"\nANOVA {metric}: F={f:.2f}, p={p:.4f}")
        tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
        print(tukey.summary())

    # Prepare ANOVA summary table
    anova_results = []
    
    # Calculate ANOVA F and p-values for each metric
    for metric in metrics:
        groups = df.groupby('Model')[metric].apply(list).values
        f, p = stats.f_oneway(*groups)
        anova_results.append({'Metric': metric, 'F-value': round(f, 3), 
                              #'p-value': round(p, 4),
                              'p-value': p #if p>1e-4 else '$<10^{-4}$',
                              })

    # Convert to DataFrame and pivot to get metrics as columns
    anova_df = pd.DataFrame(anova_results).set_index('Metric').T

    # Reformat p-values to scientific (exponential) notation
    anova_df.loc['p-value'] = anova_df.loc['p-value'].apply(lambda x: f"{x:.3e}")
    anova_df.loc['F-value'] = anova_df.loc['F-value'].apply(lambda x: f"{x:.3f}")

    # Save ANOVA summary table to LaTeX format
    anova_latex_path = "./anova_summary_table.tex"
    
    # Format the DataFrame as LaTeX with caption and label
    latex_anova = anova_df.to_latex(
        caption="ANOVA F and p-values for each performance metric across models.",
        label="tab:anova_metrics",
        escape=False,
        index=True,
        column_format="l" + "c" * len(anova_df.columns)
    )
    
    # Save to .tex file
    with open(anova_latex_path, "w") as f:
        f.write(latex_anova)
        
        
# --- 7. TAYLOR DIAGRAM ---
def plot_taylor_diagram(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(folder_path, '*.json'))
    results = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get("estimator", "Unknown").split("-")[0].upper()
            y_true, y_pred = np.array(data['y_test']), np.array(data['y_pred'])
            std, corr, rms = calculate_taylor_metrics(y_true, y_pred)
            results.append({'model': model, 'std': std, 'corr': corr, 'y_pred':y_pred, 'ref':y_true})

    model_names = sorted(set(r['model'] for r in results))
    cmap = plt.get_cmap('tab20')
    model_colors = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

    # ref=results[0]['ref']
    # taylor_stats=[{'sdev':np.std(ref), 'crmsd':0, 'ccoef':1,  
    #                         'label':'Observation', 'bias':1, 'rmsd':0}]
            
    # for i in range(len(results)):
    #     pred=results[i]['y_pred']
    #     ref=results[i]['ref']
    #     e = results[i]['model']
    #     ts=sm.taylor_statistics(pred,ref,'data')
    #     taylor_stats.append({'sdev':ts['sdev'][1], 'crmsd':ts['crmsd'][1], 
    #                           'ccoef':ts['ccoef'][1],  'label':e, 
    #                           'bias':sm.bias(pred, ref),
    #                           'rmsd':sm.rmsd(pred, ref),                                 
    #                           })

    # taylor_stats = pd.DataFrame(taylor_stats)

    # sm.taylor_diagram(taylor_stats['sdev'].values, 
    #                           taylor_stats['crmsd'].values, 
    #                           taylor_stats['ccoef'].values,
    #                           #markercolor =model_colors[k], 
    #                           alpha = 0.00,
    #                           markerSize = 20, rmsLabelFormat='0:.2f',
    #                           colSTD='k', colRMS='k', colCOR='k',
    #                           #overlay = overlay, 
    #                           #markerLabel = label
    #                           )
    


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.linspace(0, math.pi/2, 100), [1]*100, 'k--', alpha=0.5)
    for r in results:
        angle, radius = np.arccos(r['corr']), r['std']
        ax.plot(angle, radius, 'o', color=model_colors[r['model']], label=r['model'])

    handles, labels = ax.get_legend_handles_labels()
    used = dict()
    for h, l in zip(handles, labels):
        if l not in used:
            used[l] = h
    ax.legend(used.values(), used.keys(), bbox_to_anchor=(1.4, 1.05))
    ax.text(math.pi/4, max(r['std'] for r in results)*1.15, 'Correlation →', ha='center', fontsize=10)
    ax.text(-0.25, max(r['std'] for r in results)*0.25, 'Standard Deviation', ha='center', va='center', fontsize=10, rotation=90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xlim(0, math.pi/2)
    plt.title("Taylor Diagram (0°–90° Sector)")
    plt.tight_layout()
    
    if save_fig:
        filename = f"{BASENAME}_taylor.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()


# --- 7. FEATURE IMPORTANCE BARPLOT ---
def plot_feature_importance(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    model_feature_importance = defaultdict(lambda: defaultdict(list))
    all_features = set()

    # Coleta os dados
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('estimator', 'unknown')
            
            data['feature_importance'] = dict(zip(data.get('feature_names'), data.get('feature_importances')))
            feat_imp = data.get('feature_importance', {})
            #all_features.update(feat_imp.keys())
            #renamed_feat_imp = {rename_dict.get(k, k): v for k, v in feat_imp.items()}
            renamed_feat_imp = feat_imp
            all_features.update(renamed_feat_imp.keys())
            for feat, val in renamed_feat_imp.items():
                model_feature_importance[model][feat].append(val)

    all_features = sorted(list(all_features))

    for model, feat_dict in model_feature_importance.items():
        # Cria lista de registros por execução
        exec_data = []
        max_len = max(len(v) for v in feat_dict.values())

        for i in range(max_len):
            row = {feat: feat_dict[feat][i] if i < len(feat_dict[feat]) else 0.0 for feat in all_features}
            exec_data.append(row)

        df = pd.DataFrame(exec_data)
        
        plt.figure(figsize=(3, 4))    
        g = sns.catplot(
            data=df,
            kind='box',
            #height=5,
            #aspect=1.5,
            palette='viridis',
            linewidth=1.2,
            showfliers=False     # Hide outliers for cleaner look
        )
        
        stripplot = sns.stripplot(
            data=df,
            color=".25",
            jitter=True,
            size=6,
            alpha=0.7
        )
        g.set_axis_labels("Input variable", "Feature Importance")  # Y-axis label only
        #plt.xticks([])  # Remove x-axis ticks and labels (for cleaner look)

        if save_fig:
            filename = f"{BASENAME}_fi_bxp_{model}.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
        else:
            plt.show()

        df_mean = df.mean()
        df_std = df.std()

        # Gráfico
        x = np.arange(len(all_features))
        plt.figure(figsize=(4, 4))
        plt.bar(x, df_mean.values, #yerr=df_std.values, 
                capsize=5, color='skyblue', edgecolor='black')
        plt.xticks(x, all_features, rotation=45, ha='right')
        plt.ylabel("Mean Importance ± Std")
        plt.title(f"Feature Importance - {model}")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_fig:
            filename = f"{BASENAME}_fi_{model}.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
        else:
            plt.show()
            

# --- 7. PARAMETRIC ANALYSIS ---
def plot_model_params_distribution(refname, folder_path='./json-files', save_fig=False, output_dir='./img'):
    """
    Gera gráficos para os parâmetros dos modelos salvos em arquivos JSON.
    Gera boxplots para variáveis contínuas e barplots para variáveis discretas com até 7 valores únicos.

    Parâmetros:
    - folder_path (str): caminho da pasta com os arquivos JSON.
    - save_fig (bool): se True, salva os gráficos em vez de exibir.
    - output_dir (str): pasta onde salvar os gráficos (usado se save_fig=True).
    """

  

    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    param_data_2 = defaultdict(lambda: defaultdict(list))

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                data = json.load(f)[0]
                model_name = data.get('model_name', 'unknown')
                params = data.get('model_params', {}).get(model_name.lower(), {})
                for param, value in params.items():
                    if isinstance(value, (int, float)):
                        param_data_2[model_name][param].append(value)

    # Helper to decide plot type
    def is_discrete_with_cutoff(values):
        if all(isinstance(v, int) or isinstance(v, bool) for v in values):
            return 'boxplot' if len(set(values)) > 5 else 'barplot'
        return 'boxplot'

    # Plot per model/param
    for model, params in param_data_2.items():
        for param, values in params.items():
            values_series = pd.Series(values)
            plot_type = is_discrete_with_cutoff(values)
            nv = values_series.unique().shape[0]
            
            plt.figure(figsize=(7, 5) if plot_type == 'barplot' and nv>=3 else (2, 5) )
            if plot_type == 'barplot':
                sns.countplot(x=values_series)
                plt.xlabel(param)
                plt.ylabel("Count")
                plt.xlabel(None)
            else:
                sns.boxplot(y=values_series)
                plt.ylabel(param)
                plt.xlabel(None)

            plt.title(f"{model} \n {param} ({'Boxplot' if plot_type == 'boxplot' else 'Barplot'})")
            plt.title(f"{model} \n {param}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            if save_fig:
                filename = f"{BASENAME}_{model}_{param}_{plot_type}.png".replace(" ", "_").replace("/", "_")
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
                plt.close()
            else:
                plt.show()


def format_df_table(df, ref_colum, columns):
        
    # Recreate the formatted summary using string format mean (± std)
    formatted_summary = pd.DataFrame()
    
    # Format each metric as "mean (± std)"
    for metric in columns:
        formatted_summary[metric] = df.groupby(ref_colum)[metric].agg(['mean', 'std']).apply(
            lambda row: f"{row['mean']:.3f} (± {row['std']:.3f})" if pd.notnull(row['std']) else f"{row['mean']:.3f}",
            axis=1
        )

    # Convert to LaTeX table with caption and label
    latex_table = formatted_summary.to_latex(
        caption="Performance metrics by model (mean ± std).",
        label="tab:model_metrics_summary",
        escape=False,
        index=True,
        column_format="l" + "c" * len(formatted_summary.columns),    )
    
    # Save to file
    latex_file_path = "./model_metrics_summary.tex"
    with open(latex_file_path, "w") as f:
        f.write(latex_table)
    
    return formatted_summary


import os

def generate_latex_figures_from_folder(compute_performance_index, folder_path="./img", output_tex_path="insert_all_figures_from_folder.tex"):
    """
    Gera comandos LaTeX para incluir todas as imagens PNG em uma pasta.
    
    Args:
        folder_path (str): Caminho para a pasta contendo arquivos .png.
        output_tex_path (str): Caminho do arquivo .tex a ser salvo.
        
    Returns:
        str: Caminho do arquivo gerado.
    """
    # Lista todos os arquivos .png
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    # Monta os comandos LaTeX
    latex_figures = ""
    for filename in png_files:
        base = os.path.splitext(filename)[0].replace("an__", "")
        caption = base.replace("_", " ").title()
        label = base.lower().replace("_", "-")
        latex_figures += f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{folder_path}/{filename}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}

"""

    # Salva em arquivo .tex
    with open(output_tex_path, "w") as f:
        f.write(latex_figures)

    return output_tex_path



def plot_best_run_per_model(refname, models_to_remove, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    all_results = []
    for filepath in glob.glob(os.path.join(folder_path, '*.json')):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)[0]
                y_true, y_pred = data.get("y_test", []), data.get("y_pred", [])
                model_name = data.get("estimator", "unknown")
                print(model_name)
                metrics = {}
                if y_true and y_pred:
                    metric_obj = RegressionMetric(y_true, y_pred)
                    metrics = metric_obj.get_metrics_by_list_names(METRICS)
                    metrics['Model'] = model_name
                    metrics['y_true']=y_true
                    metrics['y_pred']=y_pred
                    all_results.append(metrics)                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                


    # --- Step 2: Group by model and select best (min RMSE) per model ---
    models_seen = {}
    for res in all_results:
        model = res['Model']
        if model not in models_to_remove:
            if model not in models_seen or res['R2'] > models_seen[model]['R2']:
                models_seen[model] = res

    sorted_models = dict(sorted(models_seen.items()))
    best_results = list(sorted_models.values())

    # --- Step 3: Plot best run of each model ---
    cmap = plt.get_cmap('tab10')

    for i, result in enumerate(best_results):
        model_name = result['Model']
        y_true = result['y_true']
        y_pred = result['y_pred']
        r2 = result['R2']
        mape = result['MAPE']
        rmse = result['RMSE']
        

        plt.figure(figsize=(5, 5))

        # Scatter plot
        plt.scatter(y_true, y_pred,
                    alpha=0.7,
                    edgecolor='k',
                    color=cmap(i),
                    label=model_name,
                    s=50)

        # Diagonal line
        lim_min = min(np.min(y_true), np.min(y_pred))
        lim_max = max(np.max(y_true), np.max(y_pred))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Ideal')

        # Labels and title
        plt.xlabel('Observed Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f"{refname} - Best {model_name}\n"
                  f"R² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=14)
        plt.title(f"{refname} - Best {model_name} - "
                  f"R² = {r2:.3f}", fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.axis('equal')  # Square axis with equal scaling
        plt.xlim(lim_min - 0.1 * (lim_max - lim_min), lim_max + 0.1 * (lim_max - lim_min))
        plt.ylim(lim_min - 0.1 * (lim_max - lim_min), lim_max + 0.1 * (lim_max - lim_min))
        plt.tight_layout()

        # Save or show
        if save_fig:
            filename = f"{model_name}_best_scatter_{refname}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved best scatter plot for {model_name} at {output_dir}")
        else:
            plt.show()


#%%

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    save_fig = False
    
    CONFIG = [(f"d{i}_", f'./json_automl_d{i}') for i in range(1,11)]
    for BASENAME, FOLDER_PATH in CONFIG:    
        
        refname = BASENAME.replace('_','').upper()
        df_results, df_uncertainty = load_json_data(FOLDER_PATH)
        models_to_remove = ['RF', 'ANN', 'LGBM']
        models_to_remove = models_to_remove + [m+'-FS' for m in models_to_remove]
        df_results = filter_models(df_results, models_to_remove)
        df_uncertainty = filter_models(df_uncertainty, models_to_remove)
    
        format_df_table(df_results, 'Model', METRICS)
        format_df_table(df_uncertainty, 'Model', ['MAD','Uncertainty', 'RMSE'])
            
        plot_model_metrics(refname, df_results, save_fig=save_fig, output_dir=FOLDER_FIG)
        run_anova_and_tukey(df_results)
        df_ranked = compute_performance_index(refname, df_results, save_fig=save_fig, output_dir=FOLDER_FIG)
        df_uncertainty_grouped = df_uncertainty.groupby('Model').mean().reset_index()
        plot_uncertainty(refname, df_uncertainty_grouped, save_fig=save_fig, output_dir=FOLDER_FIG)
        #df_feature_selection = analyze_feature_selection(models_to_remove,FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
        plot_feature_pareto(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
        plot_taylor_diagram(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
        #plot_feature_importance(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
        plot_model_params_distribution(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
        
        generate_latex_figures_from_folder(refname, FOLDER_FIG, "figures_output.tex")
    
        plot_best_run_per_model(refname, models_to_remove, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
