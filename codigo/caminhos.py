"""Caminhos do projeto. Tudo o que é grande ou baixável fica em nao_subir/ (ignorado pelo git)."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# versionados
DADOS = RAIZ / "dados"                                  # metadados das estações, Köppen, blocos, contornos
RES_OBS = RAIZ / "resultados" / "observados"            # resumos por estação dos assets observados_inmet
RES = RAIZ / "resultados" / "semina"                    # métricas do artigo
FIG, TAB = RES / "figuras", RES / "tabelas"

# não versionados
LOCAL = RAIZ / "nao_subir"
OBS = LOCAL / "observados_inmet"     # arquivos de estações de Xavier et al. (2022): Tmax.npz, ..., pr.npz
CACHE = LOCAL / "cache"              # grade BR-DWGD amostrada nas estações e matrizes .npy
COMPACTOS = LOCAL / "compactos"      # séries preenchidas por estação-ano, enviadas ao GEE
POWER = LOCAL / "power"              # séries diárias do NASA/POWER por estação
INTER = LOCAL / "intermediarios"     # base diária, previsões e amostras SHAP
IBGE = LOCAL / "ibge"                # malha municipal e lista de municípios do IBGE

# Google Earth Engine
PROJETO_EE = os.environ.get("EE_PROJECT", "mapbiomas-brazil")   # projeto de nuvem usado nas chamadas
PASTA_EE = "projects/fcoliveira/assets/observados_inmet"        # destino dos assets

for _p in (FIG, TAB, RES_OBS, CACHE, COMPACTOS, POWER, INTER):
    _p.mkdir(parents=True, exist_ok=True)
