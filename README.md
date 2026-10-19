# Daily reference evapotranspiration from NASA/POWER in the Cfa climate of Brazil

Código e resultados do artigo *Reanalysis-based machine learning and empirical models for daily reference
evapotranspiration* (Semina: Ciências Exatas e Tecnológicas, em submissão) e da construção dos assets
`observados_inmet` no Google Earth Engine.

**English summary.** Daily reference evapotranspiration (ETo) estimated from NASA/POWER inputs was validated
against FAO-56 Penman-Monteith (PM-FAO56) computed from observed data at 90 INMET weather stations in the humid
subtropical (Cfa) climate of Brazil (2001–2020). PM-FAO56 with three wind speed options, five empirical equations
(original and calibrated), and random forest, LightGBM, and multilayer perceptron models were compared under
spatially blocked cross-validation and a temporal test.

## Principais resultados

- O vento do NASA/POWER foi pouco confiável (KGE mediano ≤ 0,16). O PM-FAO56 com dados do POWER e vento
  constante de 2 m s⁻¹ teve o melhor desempenho entre as equações (KGE mediano de 0,906; RMSE de 0,55 mm d⁻¹).
- Os modelos de aprendizado de máquina com radiação solar foram estatisticamente equivalentes a esse PM-FAO56
  (melhor KGE mediano de 0,907) na validação cruzada espacial.
- Sem radiação solar, a acurácia do aprendizado de máquina caiu para KGE ≈ 0,77.
- A calibração corrigiu os vieses de Priestley-Taylor e Makkink; Turc foi acurado com os coeficientes originais.

## Estrutura

```
eto_cfa_nasapower/
├── codigo/                  scripts (executar a partir desta pasta)
│   ├── caminhos.py          todos os caminhos do projeto e o projeto do GEE
│   ├── metadados.py ... finalizar_assets.py    parte 1: assets observados_inmet no GEE
│   └── selecionar_estacoes.py ... figuras_resultados.py    parte 2: estudo da ETo (artigo)
├── notebooks/
│   ├── observados_inmet.ipynb   apresentação dos assets (com saídas)
│   └── notebook_fonte.py        fonte do notebook (formato percent)
├── dados/                   entradas pequenas: metadados das estações, Köppen, blocos espaciais, contorno das UFs
├── resultados/
│   ├── observados/          resumo por estação de cada variável (período, dias observados e preenchidos)
│   └── semina/              métricas, figuras e tabelas do artigo
├── nao_subir/               dados grandes e intermediários (fora do git; ver abaixo)
├── requirements.txt
└── README.md
```

## Dados de entrada

| Dado | Onde fica | Como obter |
|---|---|---|
| Séries observadas das estações (Tmax, Tmin, RH, Rs, u2, pr), em `.npz` | `nao_subir/observados_inmet/` | Arquivos de estações de Xavier et al. (2022), disponibilizados pelo autor da grade BR-DWGD ([página do projeto](https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data)) |
| Grade BR-DWGD | GEE: `projects/sat-io/open-datasets/BR-DWGD` | Lida diretamente no GEE por `extrair_xavier.py` |
| Malha municipal e lista de municípios do IBGE | `nao_subir/ibge/` | API de malhas e de localidades do IBGE |
| Mapa de Köppen de Alvares et al. (2013), 100 m | GEE (asset do IPEF no MapBiomas) | Lido por `selecionar_estacoes.py`; o resultado já está em `dados/` |
| NASA/POWER diário (API v2.10.0, comunidade AG) | `nao_subir/power/` | Baixado por `baixar_power.py` |

O vento do NASA/POWER foi solicitado também como `WSC` com `wind-elevation=10` e `wind-surface=airportgrass`
(velocidade a 10 m sobre grama) e convertido para 2 m pelo perfil logarítmico da FAO-56 (eq. 47).

## Instalação

```bash
pip install -r requirements.txt
earthengine authenticate          # necessário para as etapas que usam o GEE
```

O projeto de nuvem do GEE é lido da variável de ambiente `EE_PROJECT` (padrão em `codigo/caminhos.py`).
Os cálculos agrometeorológicos usam o pacote [agrometeorologiapy](https://github.com/fcoliveira-utfpr/agrometeorologiapy).

## Como reproduzir

Execute os scripts a partir da pasta `codigo/`.

### Parte 1. Assets `observados_inmet` no GEE

Séries diárias observadas (1961-01-01 a 2020-07-31) com as falhas preenchidas pela grade BR-DWGD e sinalizadas
no campo `preenchido` (0 = observado; 1 = Xavier). Uma feição por estação-dia, em
`projects/fcoliveira/assets/observados_inmet/{Tmax,Tmin,RH,Rs,u2,pr}`.

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `metadados.py` | Código IBGE, município e UF de cada estação → `dados/estacoes_<var>.csv` |
| 2 | `extrair_xavier.py` | Amostra a grade BR-DWGD no pixel de cada estação → `nao_subir/cache/` |
| 3 | `preencher_enviar.py` | Preenche as falhas e envia ao GEE em lotes (sem bucket) |
| 4 | `orquestrar.py` (ou `rodar.sh`) | Executa 2–3, a exportação final e as descrições dos assets |
| 5 | `finalizar_assets.py` | Aplica as descrições pendentes e apaga os lotes temporários |

Como o projeto do GCP não tem bucket, os dados sobem em lotes de cerca de 6 MB (`tmp/<var>_parteNNN`), unidos e
expandidos no servidor numa única tabela por variável. A precipitação, com 13.259 pluviômetros, gerou 115 lotes;
as demais variáveis, com cerca de 1.100 estações, 10 a 11 lotes.

As coleções do sat-io guardam os valores empacotados da BR-DWGD; o valor físico é `bruto × MULT + ADD`, com os
fatores em `preencher_enviar.py` (`ESCALA`).

### Parte 2. Estudo da ETo (artigo)

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `selecionar_estacoes.py` | Classe de Köppen e anos completos; seleciona 90 estações Cfa com ≥ 5 anos → `dados/estacoes_cfa.csv` |
| 2 | `baixar_power.py` | Baixa o NASA/POWER diário (2001–2020) nas estações |
| 3 | `montar_base.py` | Base diária com dias de cinco variáveis observadas; ETo por todos os métodos (`eto.py`) |
| 4 | `modelos.py` | Validação cruzada espacial (5 blocos por k-means), teste temporal e SHAP |
| 5 | `avaliar.py` | Métricas por estação, medianas com IC bootstrap, Wilcoxon e decomposição do erro |
| 6 | `figuras.py`, `figuras_resultados.py` | Figuras 1–6 e Tabelas do artigo → `resultados/semina/` |

A ETo de referência é o PM-FAO56 calculado com os dados da estação, apenas nos dias em que as cinco variáveis
foram observadas (`preenchido = 0`). As entradas dos modelos vêm apenas do NASA/POWER.

## Pasta `nao_subir/`

Ignorada pelo git (`.gitignore`). Contém cerca de 6 GB de dados que podem ser obtidos novamente ou regenerados pelos
scripts:

| Subpasta | Conteúdo | Gerado por |
|---|---|---|
| `observados_inmet/` | Arquivos de estações de Xavier et al. (2022) | download |
| `cache/` | Grade BR-DWGD nas estações e matrizes `.npy` | `extrair_xavier.py`, `preencher_enviar.py` |
| `compactos/` | Séries preenchidas por estação-ano enviadas ao GEE | `preencher_enviar.py` |
| `ibge/` | Malha municipal e lista de municípios | download |
| `power/` | Séries diárias do NASA/POWER | `baixar_power.py` |
| `intermediarios/` | Base diária, previsões e amostras SHAP | `montar_base.py`, `modelos.py` |
| `logs/` | Registros das execuções | — |
| `manuscrito/` | PDF do manuscrito | — |

## Como citar

Oliveira, F. C. de. Reanalysis-based machine learning and empirical models for daily reference
evapotranspiration. *Semina: Ciências Exatas e Tecnológicas* (em submissão).

Xavier, A. C.; Scanlon, B. R.; King, C. W.; Alves, A. I. New improved Brazilian daily weather gridded data
(1961–2020). *International Journal of Climatology*, v. 42, n. 16, p. 8390–8404, 2022.
https://doi.org/10.1002/joc.7731

## Autor

Fabrício Correia de Oliveira, Universidade Tecnológica Federal do Paraná (UTFPR), Câmpus Santa Helena.
fcoliveira@utfpr.edu.br
