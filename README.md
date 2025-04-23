# X-Means Clustering com K-Means

Este projeto implementa uma variação do algoritmo **X-Means** usando **K-Means** como base, com aplicação adicional em alocação de portfólios baseada em **paridade de risco com agrupamento (Clustering Risk Parity)**.

---

## 🧠 O que é X-Means?

**X-Means** é uma extensão do algoritmo K-Means que descobre automaticamente o número ideal de clusters dentro de um intervalo pré-definido.  
Em vez de fixar um valor de `k`, o X-Means começa com um número mínimo de clusters (`k_min`) e, com base no critério de informação bayesiano (**BIC**), divide iterativamente os clusters até atingir um número máximo (`k_max`) ou até que nenhuma divisão adicional melhore o modelo.

---

## 🔍 Funcionamento do Algoritmo

### 1. Inicialização
- Define um número inicial de clusters `k_min` (mínimo 2).
- Executa o algoritmo **K-Means** para separar os dados em `k_min` grupos.

### 2. Expansão dos Clusters
Para cada cluster:
- Realiza uma nova divisão em 2 subclusters usando **K-Means** novamente.
- Compara a qualidade da divisão usando o critério **BIC**:
  - Calcula a verossimilhança (log-likelihood) dos dados assumindo uma distribuição normal multivariada.
  - Avalia o BIC do cluster original e dos subclusters.

### 3. Decisão com BIC
- Se a soma dos BICs dos subclusters for **melhor (menor)** que o BIC do cluster original, a divisão é aceita.
- Caso contrário, o cluster **não** é dividido.
- Utiliza uma pilha (**stack**) para explorar divisões alternativas, garantindo uma abordagem semelhante à busca em profundidade (**DFS**).

### 4. Junção dos Resultados
- Agrupa os subclusters gerados em um único vetor de clusters finais.
- Reordena as identificações dos clusters para garantir unicidade.

---

## 📈 Aplicação em Portfólios

Além do agrupamento, este projeto utiliza os clusters obtidos para alocação de ativos com base na **Clustering Risk Parity (CRP)**.

### Etapas:
1. Agrupamento dos ativos com **X-Means**.
2. Cálculo da **matriz de covariância** dos ativos.
3. Otimização dos **pesos de alocação** minimizando a diferença entre contribuições marginais de risco dos ativos dentro dos clusters.

---

## 🧩 Funções Principais

- `fit(data, seed)`  
  Executa o algoritmo X-Means nos dados fornecidos.

- `bic`, `bic_linha`  
  Calcula o critério de informação bayesiano para o modelo com e sem divisão do cluster.

- `peso(cluster, cov)`  
  Calcula os pesos otimizados para um portfólio baseado em **Clustering Risk Parity**.

---

## 📦 Dependências

- `numpy`  
- `scipy`  
- `pandas`  
- `scikit-learn`  
- `yfinance` *(para aplicações com dados de mercado)*

---

## ✍️ Autor

Implementado por **Richard** como parte de um projeto de pesquisa em **Finanças Quantitativas** e **Machine Learning**.


Desenvolvido com foco em aplicações quantitativas para alocação de portfólios financeiros.
