import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns

# Carregar arquivos e separar por tipo de alvo
def load_grouped_json_files(folder_path):
    ucs_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)            
                ucs_files.append(data)
               
    return ucs_files

# Extrair importâncias das features por grupo de arquivos
def extract_feature_importances(json_files):
    feature_importances = {}

    for file in json_files:
        if isinstance(file, list):
            for sub_file in file:
                model = sub_file.get('estimator')
                importances = sub_file.get('feature_importances', [])
                feature_names = sub_file.get('feature_names', [])

                if not model or len(importances) != len(feature_names):
                    continue

                for i, feature in enumerate(feature_names):
                    # Ignorar features chamadas "fs" ou "cs"
                    if feature in ['fs', 'cs']:
                        continue
                    if feature not in feature_importances:
                        feature_importances[feature] = {}
                    feature_importances[feature][model] = importances[i]
        elif isinstance(file, dict):
            model = file.get('estimator')
            importances = file.get('feature_importances', [])
            feature_names = file.get('feature_names', [])

            if not model or len(importances) != len(feature_names):
                continue

            for i, feature in enumerate(feature_names):
                # Ignorar features chamadas "fs" ou "cs"
                if feature in ['fs', 'cs']:
                    continue
                if feature not in feature_importances:
                    feature_importances[feature] = {}
                feature_importances[feature][model] = importances[i]

    # Converter para DataFrame por feature
    dfs = {}
    for feature, values in feature_importances.items():
        dfs[feature] = pd.Series(values).sort_index()
    return dfs

#%%

def plot_pr_feature_importances(data, title):
    """
    Gera e salva um gráfico de importância de features com qualidade de publicação.

    Args:
        data (dict): Dicionário com features como chaves e Series do pandas
                     (com modelos como índice e importâncias como valores).
        title (str): Título base para o gráfico e nome do arquivo.
    """
    # --- 1. Preparação e Ordenação dos Dados ---
    features = list(data.keys())
    models = sorted({model for feature_values in data.values() for model in feature_values.index})

    mean_importances = {feature: df.mean() for feature, df in data.items()}
    sorted_features = sorted(mean_importances, key=mean_importances.get, reverse=False)

    # --- 2. Configuração do Gráfico ---
    plt.style.use('seaborn-v0_8-paper')
    colors = sns.color_palette('viridis', n_colors=len(models))
    fig, ax = plt.subplots(figsize=(6, 8))

    n_models = len(models)
    bar_height = 0.8 / n_models
    y_pos = np.arange(len(sorted_features))

    # --- 3. Plotagem e Anotação ---
    for i, model in enumerate(models):
        importances = [data[feature].get(model, 0) for feature in sorted_features]
        offset = i * bar_height
        bars = ax.barh(y_pos + offset, importances, height=bar_height, label=model, color=colors[i])

        # --- NOVA LINHA: Adiciona os valores na frente das barras ---
        ax.bar_label(bars, padding=3, fontsize=9, fmt='%.2f')

    # --- 4. Ajustes Finos e Estética ---
    ax.set_yticks(y_pos + bar_height * (n_models - 1) / 2)
    ax.set_yticklabels(sorted_features, fontsize=12)
    ax.set_xlabel('Averaged Feature Importance', fontsize=14, )
    ax.set_ylabel('Features', fontsize=14,)
    ax.set_title(f'Averaged Feature Importance for {title}', fontsize=16, )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # --- NOVA LINHA: Ajusta o limite do eixo x para caber os rótulos ---
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15, left=ax.get_xlim()[0] * 3) # Aumenta o limite direito em 15%

    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    fig.tight_layout()

    # --- 5. Salvando em Alta Qualidade ---
    output_filename_base = f'./img/Feature_Importance_{title.replace(" ", "_")}'
    print(f"Salvando gráficos em {output_filename_base}.png e .pdf")
    plt.savefig(f'{output_filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_filename_base}.pdf', bbox_inches='tight')
    
    plt.show()

# --- Loop Principal para Execução ---
folder_paths = glob.glob("json_v0p2_automl_*")
folder_paths.sort()
for folder_path in folder_paths:
    print(f"Processando pasta: {folder_path}")
    ucs_json = load_grouped_json_files(folder_path)

    # Extrai o nome do dataset dos dados JSON
    datasets = [entry['dataset'] for sublist in ucs_json for entry in sublist if 'dataset' in entry]
    if not datasets:
        print(f"Aviso: Nenhum dataset encontrado em {folder_path}. Pulando.")
        continue
    unique_dataset = list(set(datasets))[0].split('-')[0]

    # Extração dos dados de importância
    ucs_data = extract_feature_importances(ucs_json)

    # Remove as features 'fs' e 'cs', se necessário
    ucs_data = {key: value for key, value in ucs_data.items() if key not in ['fs', 'cs']}

    # Gera o gráfico com a nova função profissional
    plot_pr_feature_importances(ucs_data, unique_dataset)
    
    
#%%
