## Uso do K-Means++ na Implementação

A implementação do X-Means neste repositório utiliza o **K-Means++** para a inicialização dos centroides. Essa abordagem combina as vantagens do K-Means++ (melhor inicialização) com a capacidade do X-Means de determinar automaticamente o número de clusters. Essa combinação garante que os clusters iniciais sejam bem posicionados, resultando em uma convergência mais rápida e em melhores resultados finais.

Além disso, este projeto tem como objetivo aplicar o algoritmo X-Means na **criação de portfólios de investimento**. Essa aplicação é especialmente útil na identificação de grupos de ativos com características semelhantes, facilitando a diversificação e a análise de risco-retorno em estratégias financeiras.

## X-Means: Uma Extensão Poderosa do K-Means

Bem-vindo ao repositório de implementação do algoritmo **X-Means**! Este documento tem como objetivo explicar como o X-Means funciona, destacar suas vantagens em relação ao K-Means e ao K-Means++.



## O que é o X-Means?
O **X-Means** é uma extensão aprimorada do algoritmo K-Means, projetada para lidar com uma limitação comum do K-Means: a necessidade de definir previamente o número de clusters (*k*). Enquanto o K-Means tradicional exige que o usuário forneça o valor de *k*, o X-Means utiliza um critério automático para determinar o número ótimo de clusters.

O X-Means realiza isso iterativamente, dividindo clusters existentes e avaliando a qualidade da divisão utilizando métricas estatísticas como o **Critério de Informação de Akaike (AIC)** ou o **Critério de Informação Bayesiano (BIC)**.



## Principais Vantagens do X-Means

1. **Determinação Automática do Número de Clusters:**
   - Não é necessário informar o número de clusters inicialmente, reduzindo o trabalho manual e o risco de escolhas subótimas.

2. **Eficiência Computacional:**
   - O X-Means é mais eficiente do que técnicas de busca exaustiva de *k*, como o Método do Cotovelo.

3. **Melhor Segmentação dos Dados:**
   - A divisão iterativa melhora a qualidade dos clusters em relação ao K-Means tradicional.

4. **Robustez ao Ruído:**
   - Ao utilizar critérios como o BIC, o X-Means evita overfitting, que pode ocorrer ao escolher muitos clusters.

5. **Superioridade ao K-Means++:**
   - Embora o K-Means++ melhore a inicialização dos centroides em relação ao K-Means, ele ainda depende de uma definição manual do número de clusters. O X-Means supera essa limitação ao determinar automaticamente o *k* ótimo, além de fornecer clusters de maior qualidade ao integrar avaliações estatísticas no processo.

## Comparativo entre X-Means, K-Means e K-Means++

| Característica                  | K-Means                      | K-Means++                  | X-Means                     |
|---------------------------------|------------------------------|----------------------------|-----------------------------|
| Inicialização dos Centroides  | Aleatória                   | Melhorada                  | Melhorada                   |
| Escolha do Número de Clusters | Definido manualmente        | Definido manualmente      | Determinação automática   |
| Eficiência Computacional      | Alta                         | Alta                       | Moderada                    |
| Precisão dos Clusters         | Moderada                     | Alta                       | Muito Alta                  |



## Como Funciona o Algoritmo X-Means?

O algoritmo X-Means funciona em três etapas principais:

1. **Inicialização:**
   - Realiza o agrupamento inicial utilizando o K-Means tradicional.

2. **Divisão dos Clusters:**
   - Divide cada cluster candidato em dois subclusters e recalcula os centroides.

3. **Avaliação e Seleção:**
   - Utiliza critérios estatísticos como o AIC ou o BIC para avaliar se a divisão dos clusters melhora a segmentação dos dados. Apenas divisões que aumentam a qualidade são mantidas.


## 📚 Referências
- Nakagawa, K., Kawahara, T., Ito, A. (2020). Asset Allocation Strategy with Non-Hierarchical Clustering Risk Parity Portfolio. Journal of Mathematical Finance, 10, 513-524.
- Ishioka, T. (2006) An Expansion of X-Means: Progressive Iteration of K-Means and Merging of the Clusters. Journal of Computational Statistics, 18, 3-13.



