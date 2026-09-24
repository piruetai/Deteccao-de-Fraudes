#Principais bibliotecas utilizadas na formulação e treino do modelo
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import lightgbm as lgb #Modelo baseado em Árvore mais eficiente que o XGBOOST
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


#Dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

#Realizando a leitura do dataset, e atribuindo a uma variável
df = pd.read_csv(url)

#Vizualizando a proporção de fraudes/não fraudes em porcentagem
#print(df["Class"].value_counts(normalize=True))


#Feature Engineering, padronização de valores informativos para melhor aprendizado do modelo

#Transformação logarítimica para compreensão de valores, auxiliando para uma melhor comparação de valores maiores e menores
df["Amount_log"] = np.log1p(df["Amount"])
#print(df["Amount_log"])
#print(df["Amount"])

os.system('cls')

#Transforma a média e desvio padrão da coluna amount em um único valor mensurável 
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

#Separando os dados do modelo em conjuntos,  70% para treino 30% para teste
x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = (
train_test_split(x, y, stratify = y, test_size=0.3, random_state=42))


#Regressão Logística, considerada o próprio modelo em si
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


#Utilizando métricas de avaliação do modelo: precision, recall  e f1-score
print(classification_report(y_test, y_pred))
input("Pressione para Continuar...")
os.system('cls')


#Avaliação Gráfica do Modelo
#Curva ROC, para medição de falsos positivos e positivos reais
y_probs = model.predict_proba(x_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("Curva ROC")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Positivos Reais")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))

#Curva precision-recall, que mede quantos casos reconhecidos como fraude, são verdadeiramente fraudes
y_probs = model.predict_proba(x_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Balanceamento de Dados, balanceando a quantidades das classes, para melhor treinamento

#Undersampling, equalizando a quantidade de fraudes e não fraudes, ao diminuir a classe majoritária
"""
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(n=len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])
"""
#Oversampling, criando novos dados, aumentando a classe minoritária
"""
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)
"""

#Conjuntos de Árvores de Decisão, onde cada uma aprende padrões independentes, culminando num modelo mais preciso
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    #balanceia as classes automaticamente
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("Conjunto de Árvores de Decisão Independentes")
print(classification_report(y_test, y_pred_rf))
input("Pressione para Continuar...")
os.system('cls')

#Pipeline, utilizado para encadear os fluxos de processamento em modelos
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train, y_train)

y_pred_pipeline = pipeline.predict(x_test)

#Threshold é responsável por definir a sensibilidade do modelo, quanto a detecção de fraudes
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
print("Modelo do Pipeline (mesmo que o primeiro)")
print(classification_report(y_test, y_pred_custom))
input("Pressione para Continuar...")
os.system('cls')


#XGBOOST, é um algoritimo de treino complexo baseado em boosting, 
# onde cada modelo é treinado em sequência e cada iteração corrige os erros anteriores
xgb = XGBClassifier(
    scale_pos_weight=10, 
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print("Modelo XGBOOST")
print(classification_report(y_test, y_pred_xgb))
input("Pressione para Continuar...") 
os.system('cls')


# Criando o modelo
# scale_pos_weight ajuda a compensar o desbalanceamento entre fraude/não fraude
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.02,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42
)

# Treinando
model.fit(x_train, y_train)

# Prevendo
y_pred = model.predict(x_test)

# Avaliando
print("Modelo LightGBM")


# Threshold ideal usando o modelo LightGBM
y_probs_lgb = model.predict_proba(x_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_probs_lgb)

# F1-score para cada threshold, pra achar o ponto de equilíbrio
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
melhor_idx = np.argmax(f1_scores)
melhor_threshold = thresholds[melhor_idx]

print(f"Melhor threshold encontrado: {melhor_threshold:.3f}")
print(f"F1-score nesse ponto: {f1_scores[melhor_idx]:.3f}")

y_pred_custom = (y_probs_lgb > melhor_threshold).astype(int)
print("Modelo LightGBM com threshold ajustado")
print(classification_report(y_test, y_pred_custom))

input("Pressione para Continuar...") 
os.system('cls')


#Explicabilidade ou SHAP, elabora como a decisão do modelo é influenciada por cada variável
print("Explicabilidade do Modelo XGB")
explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)
input("Pressione para Continuar...") 
os.system('cls')


print("Explicabilidade do Modelo LightGBM")
explainer = shap.Explainer(model)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)
input("Pressione para Encerrar...") 
os.system('cls')

