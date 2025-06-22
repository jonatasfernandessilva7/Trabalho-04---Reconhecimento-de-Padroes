import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

resultados = {}

inicio_rvm = time.time()
y_pred_rvm = model.predict(X)
fim_rvm = time.time()

resultados['RVM'] = {
    'Acurácia': accuracy_score(y, y_pred_rvm),
    'Precisão': precision_score(y, y_pred_rvm, average='macro'),
    'Recall': recall_score(y, y_pred_rvm, average='macro'),
    'F1-Score': f1_score(y, y_pred_rvm, average='macro'),
    'Tempo (s)': fim_rvm - inicio_rvm
}

df_resultados = pd.DataFrame(resultados).T
display(df_resultados)

df_resultados.drop(columns='Tempo (s)').plot(kind='bar', figsize=(10, 6))
plt.title("Comparação de Métricas - SVM vs RVM")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(True)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(6, 4))
df_resultados['Tempo (s)'].plot(kind='bar', color=['steelblue', 'orange'])
plt.title("Tempo de Execução")
plt.ylabel("Segundos")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()
