from sklearn import svm
from sklearn_rvm import EMRVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.randn(300, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

kernels = ["linear", "poly", "rbf", "sigmoid"]
results = {}

for kernel_name in kernels:
    # --- Avaliando SVM ---
    start_time = time.time()
    if kernel_name == "linear":
        svm_model = svm.SVC(kernel=kernel_name, random_state=42)
    else:
        # Usar gamma=2 para consistência com o exemplo anterior de SVM
        svm_model = svm.SVC(kernel=kernel_name, gamma=2, random_state=42)
    svm_model.fit(X, y)
    svm_time = time.time() - start_time
    y_pred_svm = svm_model.predict(X)

    accuracy_svm = accuracy_score(y, y_pred_svm)
    precision_svm = precision_score(y, y_pred_svm, average='weighted', zero_division=0)
    recall_svm = recall_score(y, y_pred_svm, average='weighted', zero_division=0)
    f1_svm = f1_score(y, y_pred_svm, average='weighted', zero_division=0)

    results[f"SVM - {kernel_name}"] = {
        "Acurácia": accuracy_svm,
        "Precisão": precision_svm,
        "Recall": recall_svm,
        "F1-Score": f1_svm,
        "Tempo (s)": svm_time,
    }

    # --- Avaliando RVM ---
    start_time = time.time()
    if kernel_name == "linear":
        # Removido random_state, pois EMRVC não aceita este parâmetro
        rvm_model = EMRVC(kernel=kernel_name)
    else:
        # Removido random_state, pois EMRVC não aceita este parâmetro
        rvm_model = EMRVC(kernel=kernel_name, gamma=2)
    rvm_model.fit(X, y)
    rvm_time = time.time() - start_time
    y_pred_rvm = rvm_model.predict(X)

    accuracy_rvm = accuracy_score(y, y_pred_rvm)
    precision_rvm = precision_score(y, y_pred_rvm, average='weighted', zero_division=0)
    recall_rvm = recall_score(y, y_pred_rvm, average='weighted', zero_division=0)
    f1_rvm = f1_score(y, y_pred_rvm, average='weighted', zero_division=0)

    results[f"RVM - {kernel_name}"] = {
        "Acurácia": accuracy_rvm,
        "Precisão": precision_rvm,
        "Recall": recall_rvm,
        "F1-Score": f1_rvm,
        "Tempo (s)": rvm_time,
    }

# Cria um DataFrame para exibir os resultados
df_results = pd.DataFrame.from_dict(results, orient="index")

print("\nMétricas de Avaliação dos Modelos SVM e RVM (Dataset XOR):\n")
print(df_results.round(4))

metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

fig, axes = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=(10, 5 * len(metrics_to_plot)))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    df_results[metric].plot(kind='bar', ax=axes[i], title=f'{metric} por Modelo e Kernel')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--')

plt.tight_layout()
plt.show()
