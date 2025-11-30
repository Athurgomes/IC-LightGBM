# Classificação espectral de bactérias multirresistentes com Gradient Boosting aplicado a imagens hiperespectrais SWIR

O presente repositório contém a infraestrutura de código (pipeline Python) desenvolvida para a **preparação e limpeza de dados hiperespectrais SWIR (1000–2500 nm)**, com o objetivo de gerar *features* robustas para o modelo de classificação de bactérias multirresistentes.

## Objetivo do Projeto
O principal objetivo desta fase do projeto é **extrair apenas os pixels internos à Região de Interesse (ROI)** – o "batoque" contendo a amostra bacteriana – e aplicar técnicas de **Quimiometria** para remover ruído, variações de luz (scattering) e redundância espectral.

## Status Atual: Pipeline de Pré-processamento CONCLUÍDO

As seguintes etapas do pipeline foram implementadas, calibradas e estatisticamente validadas:

### 1. Ingestão e Calibração
- **Leitura de Dados (Bronze Level):** Implementação da rotina para carregar cubos de dados `.hdr` brutos.
- **Calibração à Refletância:** Conversão automática dos dados de Radiância (DN) para Refletância, utilizando as referências de Branco (White Reference) e Escuro (Dark Reference) para neutralizar os efeitos da fonte de luz e do sensor.

### 2. Segmentação e Extração da ROI (Região de Interesse)
- **Corte de Bandas Ruidosas:** Remoção das bandas iniciais e finais do espectro SWIR (regiões com baixa relação sinal-ruído).
- **Segmentação Automática (Transformada de Hough):** Implementação do algoritmo de **Transformada de Hough** para detecção automática e precisa do contorno circular do batoque, eliminando a necessidade de anotação manual.
- **Calibração Fina:** Inclusão de uma **margem de segurança** no raio detectado, garantindo que o ROI extraído contenha apenas o material biológico puro, excluindo reflexos e bordas do plástico.

### 3. Aplicação e Validação de Técnicas Quimiométricas
Os dados de cada ROI (pixels internos) foram estruturados em matrizes 2D (pixels x bandas) e processados individualmente:

| Técnica | Módulo Python | Objetivo |
| :--- | :--- | :--- |
| **Multiplicative Scatter Correction (MSC)** | `preprocessing.py` | Correção das variações de intensidade e espalhamento de luz (scattering) entre as amostras. |
| **Savitzky-Golay (1ª Derivada)** | `preprocessing.py` | Remoção de ruído aditivo e realce de picos de absorção químicos sutis. |
| **Z-Score Standardization** | `preprocessing.py` | Padronização dos dados para média zero e desvio padrão unitário, essencial para alguns algoritmos de Machine Learning. |

---

## Validação dos Resultados

A eficácia do pipeline foi validada por meio de análises estatísticas e visualizações, demonstrando a limpeza e a estruturação dos dados:

1.  **Gráficos de Espectros (Espaguete Plots):** Demonstração visual de como o MSC reduz a dispersão vertical entre as amostras.
2.  **Análise de Componentes Principais (PCA):** O PCA comprovou que o MSC removeu o efeito dominante da intensidade luminosa (PC1 caiu de ~94% para ~64%), permitindo que as diferenças químicas (PC2) fossem reveladas. O PCA também isolou **outliers** (amostras anômalas).
3.  **Matriz de Correlação:** Demonstração visual de como a 1ª Derivada de Savitzky-Golay quebrou a alta redundância entre as bandas espectrais (transformando a matriz de "quente" para "fria").

## Próximas Etapas

Com os dados limpos e robustos (nível Silver/Gold), o projeto avança para a fase de Modelagem:
1.  Seleção de Bandas (Feature Selection).
2.  Divisão dos dados em treino/teste.
3.  Treinamento e otimização do classificador **LightGBM**.
