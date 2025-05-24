# README - Classificação de Reviews B2W com Keras/TensorFlow

Este documento descreve o fluxo de trabalho e o código contido no notebook `B2W_Review_Classification_Corrected_v7.1.ipynb`. O objetivo principal é treinar um modelo de rede neural para classificar reviews de produtos da B2W como "Recomenda" ou "Não Recomenda" com base no texto da review.

## Estrutura do Notebook

O notebook está organizado nas seguintes seções:

### 1. Instalação de Dependências

*   **Objetivo:** Garantir que todas as bibliotecas necessárias (TensorFlow, Pandas, Scikit-learn, etc.) estejam instaladas no ambiente de execução (como o Google Colab).
*   **Código:** Contém um comando `!pip install ...` comentado. Descomente e execute esta célula se estiver em um ambiente onde essas bibliotecas não estão pré-instaladas.

### 2. Importar Bibliotecas

*   **Objetivo:** Carregar todas as funções e classes necessárias das bibliotecas que serão usadas ao longo do notebook.
*   **Código:** Inclui imports para `pandas` (manipulação de dados), `numpy` (cálculos numéricos), `tensorflow` e `keras` (construção e treinamento do modelo), `sklearn` (divisão de dados, métricas), `seaborn` e `matplotlib` (visualização), `re` e `string` (processamento de texto).

### 3. Carregar e Preparar os Dados

*   **Objetivo:** Ler o arquivo `B2W-Reviews01.csv`, selecionar as colunas relevantes (`review_text`, `recommend_to_a_friend`), remover linhas com dados ausentes e transformar a coluna `recommend_to_a_friend` ('Yes'/'No') em uma coluna numérica binária (`target`: 1/0).
*   **Código:** Usa `pd.read_csv` para carregar os dados. Realiza a limpeza e a transformação da coluna alvo. Exibe informações sobre o dataset resultante e a distribuição das classes.

### 4. Dividir Dados em Treino e Teste (80/20)

*   **Objetivo:** Separar o dataset em um conjunto maior para treinamento (80%) e um conjunto menor para teste final (20%). A divisão é estratificada para manter a proporção original das classes em ambos os conjuntos.
*   **Código:** Utiliza a função `train_test_split` do Scikit-learn.

### 5. Pré-processamento de Texto com TextVectorization

*   **Objetivo:** Converter o texto das reviews (strings) em sequências numéricas que a rede neural possa entender. Isso envolve:
    *   **Padronização:** Limpeza do texto (converter para minúsculas, remover HTML e pontuação) usando a função `custom_standardization`.
    *   **Tokenização:** Dividir o texto em palavras (tokens).
    *   **Vetorização:** Mapear cada palavra para um índice numérico e criar sequências de tamanho fixo.
*   **Código:**
    *   Define a função `custom_standardization` e a registra com `@utils.register_keras_serializable()` para que possa ser salva e carregada com o modelo.
    *   Cria uma camada `TextVectorization`, especificando a função de padronização, o tamanho máximo do vocabulário (`max_features`) e o comprimento da sequência (`sequence_length`).
    *   Adapta (`.adapt()`) a camada **apenas** com os dados de treino (`X_train_full`) para que ela aprenda o vocabulário.

### 6. Construção do Modelo de Rede Neural

*   **Objetivo:** Definir a arquitetura da rede neural.
*   **Código:**
    *   Define uma função `create_model()` que retorna um modelo `Sequential` do Keras.
    *   O modelo inclui:
        *   A camada `TextVectorization` (já adaptada) como primeira camada.
        *   Uma camada `Embedding` para aprender representações vetoriais das palavras.
        *   Uma camada `GlobalAveragePooling1D` para reduzir a dimensionalidade.
        *   Duas camadas `Dense` (totalmente conectadas), a última com ativação `sigmoid` para classificação binária.
    *   Compila o modelo definindo o otimizador (`adam`), a função de perda (`binary_crossentropy`) e a métrica (`accuracy`).
    *   Exibe um sumário da arquitetura do modelo.

### 7. Treinamento com Validação Cruzada (Stratified K-Fold)

*   **Objetivo:** Avaliar a robustez do modelo treinando-o e validando-o em diferentes subconjuntos dos dados de treino. Isso ajuda a garantir que o desempenho não seja dependente de uma única divisão treino/validação.
*   **Código:**
    *   Usa `StratifiedKFold` para dividir os dados de treino em 5 folds (partes).
    *   Em cada fold, treina um novo modelo usando 4 partes para treino e 1 para validação.
    *   O treinamento (`.fit()`) recebe os dados como `tf.data.Dataset` contendo o texto bruto (a vetorização ocorre dentro do modelo).
    *   Avalia (`.evaluate()`) o modelo treinado no fold de validação.
    *   Calcula e exibe a acurácia média e o desvio padrão entre os folds, com explicações.

### 8. Treinamento do Modelo Final (com todos os dados de treino)

*   **Objetivo:** Treinar um modelo final usando **todos** os dados de treino (`X_train_full`, `y_train_full`) para obter a melhor versão possível do modelo.
*   **Código:**
    *   Cria um `tf.data.Dataset` com todos os dados de treino.
    *   Cria uma nova instância do modelo usando `create_model()`.
    *   Treina o modelo (`.fit()`) com o dataset completo por um número definido de épocas.

### 9. Avaliação Final no Conjunto de Teste

*   **Objetivo:** Avaliar o desempenho do modelo final em dados que ele nunca viu durante o treinamento (o conjunto de teste separado na etapa 4). Isso fornece uma estimativa realista de como o modelo generaliza para novas reviews.
*   **Código:**
    *   Avalia (`.evaluate()`) o modelo final usando `X_test.values` e `y_test`.
    *   Exibe a `Loss` e a `Acurácia` no conjunto de teste, com explicações.
    *   Faz predições (`.predict()`) no conjunto de teste.
    *   Gera e exibe um `Relatório de Classificação` detalhado (precisão, recall, F1-score) com explicações.
    *   Gera e exibe uma `Matriz de Confusão` visual usando `seaborn`/`matplotlib`, mostrando os acertos e erros do modelo, com explicações.

### 10. Salvar o Modelo Treinado
*   **Observação** Não consegui usar pickle pois estava com dificuldade para usar a biblioteca então usei o keras
*   **Objetivo:** Salvar o modelo final treinado em um arquivo `.keras`. Isso permite reutilizar o modelo posteriormente sem a necessidade de retreiná-lo.
*   **Código:** Usa `final_model.save()` para salvar o modelo. Como a função `custom_standardization` foi registrada, ela é salva junto com o modelo.

### 11. Criar Função de Predição para Novos Dados

*   **Objetivo:** Definir uma função reutilizável (`predict_new_review`) que carrega o modelo salvo e classifica uma nova review fornecida como texto.
*   **Código:**
    *   Define a função `custom_standardization` novamente (com o decorador) para que esteja no escopo ao carregar o modelo.
    *   Define a função `predict_new_review` que:
        *   Recebe o texto da review e o caminho do arquivo do modelo.
        *   Carrega o modelo usando `keras.models.load_model()`, passando a função customizada em `custom_objects` para garantir que o Keras a encontre.
        *   Faz a predição (`.predict()`) passando o texto como um `tf.constant`.
        *   Retorna a classe predita (0 ou 1) e a probabilidade.
        *   Inclui tratamento de erro (`try`/`except`).

### 12. Exemplo de Uso da Função de Predição

*   **Objetivo:** Demonstrar como usar a função `predict_new_review` para classificar exemplos de reviews positivas e negativas.
*   **Código:**
    *   Define o nome do arquivo do modelo a ser carregado.
    *   Define uma função auxiliar `run_prediction_example` para simplificar a exibição.
    *   Define strings de exemplo para reviews positiva e negativa.
    *   Chama `run_prediction_example` para cada exemplo, que por sua vez chama `predict_new_review` e exibe a review e a recomendação resultante.
    *   **Importante:** Inclui um aviso para garantir que a célula #10 (salvar modelo) foi executada antes desta.

---
