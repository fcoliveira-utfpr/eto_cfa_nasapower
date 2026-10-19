"""Descrições (Markdown) e propriedades dos assets observados_inmet no GEE."""
import ee

from caminhos import PASTA_EE as PASTA

VARIAVEIS = {
    "Tmax": ("Temperatura máxima diária do ar", "°C", "TMAX"),
    "Tmin": ("Temperatura mínima diária do ar", "°C", "TMIN"),
    "RH": ("Umidade relativa média diária do ar", "%", "RH"),
    "Rs": ("Radiação solar global diária", "MJ m-2 dia-1", "RS"),
    "u2": ("Velocidade média diária do vento a 2 m", "m s-1", "U2"),
    "pr": ("Precipitação pluvial diária", "mm dia-1", "PR"),
}


def descricao(var, resumo):
    nome, unid, _ = VARIAVEIS[var]
    rede = ("pluviômetros (ANA, INMET e redes estaduais) usados por Xavier et al. (2022)" if var == "pr"
            else "estações meteorológicas do INMET (automáticas e convencionais) usadas por Xavier et al. (2022)")
    return f"""# Observados INMET — {nome} ({var})

Série diária observada em **{resumo['estacoes']} {rede}**, de 01/01/1961 a 31/07/2020, com as falhas
preenchidas pela grade **BR-DWGD** (Xavier et al., 2022; `projects/sat-io/open-datasets/BR-DWGD`).

**Unidade:** {unid}
**Período de cada estação:** do primeiro ao último dia com observação (limitado a 31/07/2020).
**Preenchimento:** {resumo['pct_preenchido']:.1f}% dos {resumo['dias']:,} estação-dias vieram da grade de Xavier
(pixel de 0,1° que contém a estação; estações em pixels mascarados do litoral usam a média focal de 2 pixels).

## Campos (uma feição por estação-dia)
| campo | descrição |
|---|---|
| `cod_estacao` | código da estação (INMET, p. ex. A807; ou código do pluviômetro) |
| `cod_ibge` | código IBGE (7 dígitos) do município da estação |
| `municipio`, `uf` | município e unidade da federação |
| `lat`, `lon`, `alt` | coordenadas (graus) e altitude (m) |
| `data` | data (AAAA-MM-DD); `system:time_start` também preenchido |
| `ano`, `mes` | ano e mês |
| `valor` | valor diário ({unid}) |
| `preenchido` | **0 = observado; 1 = preenchido com Xavier (BR-DWGD)** |

## Exemplo no GEE (JavaScript)
```javascript
var tab = ee.FeatureCollection('{PASTA}/{var}');
// série de uma estação, apenas dados observados
var serie = tab.filter(ee.Filter.eq('cod_estacao', 'A807'))
               .filter(ee.Filter.eq('preenchido', 0))
               .filterDate('2010-01-01', '2011-01-01');
print(ui.Chart.feature.byFeature(serie, 'system:time_start', 'valor'));
// todas as estações de um município (código IBGE)
print(tab.filter(ee.Filter.eq('cod_ibge', '4106902')).aggregate_array('cod_estacao').distinct());
// proporção de dias preenchidos por estação
print(tab.filter(ee.Filter.eq('uf', 'PR')).reduceColumns(ee.Reducer.mean().group(1, 'cod_estacao'),
                                                         ['preenchido', 'cod_estacao']));
```

## Exemplo em Python
```python
import ee
ee.Initialize(project='SEU-PROJETO')
tab = ee.FeatureCollection('{PASTA}/{var}')
serie = (tab.filter(ee.Filter.eq('cod_estacao', 'A807'))
            .filterDate('2010-01-01', '2011-01-01'))
df = ee.data.computeFeatures({{'expression': serie, 'fileFormat': 'PANDAS_DATAFRAME'}})
df[['data', 'valor', 'preenchido']].head()
```

## Citação
Xavier, A. C.; Scanlon, B. R.; King, C. W.; Alves, A. I. New improved Brazilian daily weather gridded data
(1961–2020). International Journal of Climatology, v. 42, n. 16, p. 8390-8404, 2022. https://doi.org/10.1002/joc.7731

Organizado por F. C. de Oliveira (UTFPR) em 2026.
"""


def aplicar(var, resumo):
    nome, unid, col = VARIAVEIS[var]
    ee.data.setAssetProperties(f"{PASTA}/{var}", {
        "description": descricao(var, resumo),
        "title": f"Observados INMET - {nome}",
        "variavel": var, "unidade": unid, "periodo": "1961-01-01/2020-07-31",
        "fonte_preenchimento": f"projects/sat-io/open-datasets/BR-DWGD/{col}",
        "n_estacoes": int(resumo["estacoes"]), "pct_preenchido": round(float(resumo["pct_preenchido"]), 2),
    })


def descricao_pasta():
    linhas = "\n".join(f"- `{v}`: {n} ({u})" for v, (n, u, _) in VARIAVEIS.items())
    return f"""# observados_inmet
Séries diárias observadas (1961-2020) das estações usadas na grade BR-DWGD (Xavier et al., 2022), com falhas
preenchidas pela própria grade e sinalizadas no campo `preenchido` (0 = observado, 1 = Xavier).
Cada ponto traz o código da estação e o código IBGE do município.

{linhas}
"""
