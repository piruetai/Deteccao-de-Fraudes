# Detecção de Fraude em Transações de Cartão de Crédito

Projeto de Machine Learning aplicado à detecção de fraudes em transações financeiras, utilizando Python e diferentes algoritmos de classificação supervisionada.

## 📋 Sobre o Projeto

O objetivo é identificar transações fraudulentas em um dataset real e altamente desbalanceado (fraudes representam menos de 1% dos dados). O projeto percorre todo o ciclo de um problema de ML: preparação dos dados, treinamento de múltiplos modelos, tratamento de desbalanceamento, avaliação com métricas adequadas, ajuste fino e explicabilidade.

**Dataset:** Transações de cartão de crédito, com variáveis anonimizadas (V1–V28, resultado de uma transformação PCA), além de `Time` e `Amount`.

## 🛠️ Bibliotecas Utilizadas

| Biblioteca | Finalidade |
|---|---|
| `pandas`, `numpy` | Manipulação e transformação dos dados |
| `scikit-learn` | Pré-processamento, divisão treino/teste, métricas, Regressão Logística, Random Forest |
| `lightgbm` | Modelo de boosting em árvore |
| `xgboost` | Modelo de boosting em árvore |
| `imbalanced-learn` (SMOTE) | Técnica de balanceamento por oversampling |
| `shap` | Explicabilidade dos modelos |
| `matplotlib` | Visualização gráfica (curvas ROC e Precision-Recall) |

## 🧠 Modelos Treinados e Comparados

### 1. Regressão Logística
Modelo linear usado como baseline. Simples e interpretável, mas com menor capacidade de capturar relações não-lineares nos dados. Serviu de ponto de partida para comparação com os demais.

### 2. Random Forest
Conjunto (ensemble) de árvores de decisão treinadas de forma independente, cujas previsões são combinadas. Uso do parâmetro `class_weight="balanced"` para compensar automaticamente o desbalanceamento das classes, sem precisar alterar os dados diretamente.

### 3. XGBoost
Algoritmo de boosting, onde os modelos são treinados sequencialmente e cada novo modelo corrige os erros dos anteriores. Uso do `scale_pos_weight` para dar mais peso à classe minoritária (fraude).

### 4. LightGBM
Também baseado em boosting, mas com uma estratégia de crescimento de árvore diferente (leaf-wise), tornando-o geralmente mais rápido e eficiente que o XGBoost em datasets grandes, mantendo desempenho equivalente ou superior.

**Resultado do LightGBM na classe de fraude (Classe 1):**

| Métrica | Valor |
|---|---|
| Precision | 0.69 |
| Recall | 0.79 |
| F1-score | 0.74 |

## 📊 Principais Aprendizados

### 1. Acurácia é uma métrica enganosa em dados desbalanceados
Como fraudes são raras, um modelo que sempre prevê "não fraude" alcançaria mais de 99% de acurácia — e seria completamente inútil. Por isso, o foco da avaliação foi em **precision, recall e F1-score**, que revelam o desempenho real na classe minoritária.

- **Precision** responde: *das vezes que o modelo disse "fraude", quantas realmente eram?*
- **Recall** responde: *das fraudes reais, quantas o modelo conseguiu identificar?*
- **F1-score** equilibra as duas métricas anteriores.

### 2. Dados desbalanceados exigem tratamento específico
Foram exploradas diferentes abordagens para lidar com o desbalanceamento:
- `class_weight="balanced"` (Random Forest)
- `scale_pos_weight` (XGBoost e LightGBM)
- Oversampling com **SMOTE** (técnica alternativa, testável para comparação)

### 3. O threshold de decisão pode ser ajustado
Por padrão, os modelos classificam como fraude quando a probabilidade prevista ultrapassa 0.5. Esse limiar foi ajustado a partir da curva Precision-Recall, buscando o ponto de melhor F1-score — permitindo priorizar recall (pegar mais fraudes) ou precision (reduzir falsos alarmes), conforme a necessidade do negócio.

### 4. Modelos baseados em árvore não exigem padronização dos dados
Diferente da Regressão Logística, algoritmos como Random Forest, XGBoost e LightGBM não são sensíveis à escala das variáveis, simplificando o pré-processamento.

### 5. Explicabilidade é essencial em contextos financeiros
Usando **SHAP**, foi possível entender quais variáveis mais influenciaram cada decisão do modelo — algo fundamental em aplicações financeiras, onde auditoria e transparência das decisões automatizadas são exigidas.

### 6. Boas práticas de avaliação
- Separação treino/teste com `stratify=y`, garantindo que a proporção de fraudes seja mantida em ambos os conjuntos.
- Uso de curvas **ROC** e **Precision-Recall** para visualizar o comportamento do modelo além de um único ponto de corte.

## 🚀 Possíveis Melhorias Futuras

- Validação cruzada estratificada (`StratifiedKFold`) para resultados mais robustos
- Tuning de hiperparâmetros (`GridSearchCV`, `RandomizedSearchCV` ou Optuna)
- Testar combinações de oversampling (SMOTE) com undersampling
- Ensemble entre LightGBM, XGBoost e Random Forest
- Engenharia de features adicionais a partir de `Time`

## ▶️ Como Executar

```bash
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn shap matplotlib
python nome_do_arquivo.py
```

O script exibe os resultados de cada modelo em sequência no terminal, pausando entre eles (pressione Enter para continuar), além de abrir gráficos de curva ROC, Precision-Recall e importância de features via SHAP.
