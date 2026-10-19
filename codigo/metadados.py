"""Metadados das estações: código da estação, código IBGE, município e UF (junção espacial com a malha do IBGE)."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from caminhos import DADOS, IBGE, OBS
VARS = ["Tmax", "Tmin", "RH", "Rs", "u2", "pr"]


def municipios():
    mun = gpd.read_file(IBGE / "br_municipios.geojson").rename(columns={"codarea": "cod_ibge"})
    info = json.loads((IBGE / "municipios_ibge.json").read_text(encoding="utf-8"))
    nomes = pd.DataFrame([{
        "cod_ibge": str(m["id"]), "municipio": m["nome"],
        "uf": (m.get("microrregiao") or {}).get("mesorregiao", {}).get("UF", {}).get("sigla")
        or (m.get("regiao-imediata") or {}).get("regiao-intermediaria", {}).get("UF", {}).get("sigla"),
    } for m in info])
    return mun.merge(nomes, on="cod_ibge", how="left")


def estacoes(var, mun):
    z = np.load(OBS / f"{var}.npz", allow_pickle=True)
    ll = z["lat_lon_alt"]
    est = pd.DataFrame({"cod_estacao": z["ID"].astype(str), "lat": ll[:, 0].astype(float),
                        "lon": ll[:, 1].astype(float), "alt": ll[:, 2].astype(float)})
    pts = gpd.GeoDataFrame(est, geometry=gpd.points_from_xy(est.lon, est.lat), crs="EPSG:4674").to_crs(mun.crs)
    j = gpd.sjoin(pts, mun[["cod_ibge", "municipio", "uf", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    # estações fora dos polígonos (litoral, fronteira): município mais próximo
    fora = j.cod_ibge.isna()
    if fora.any():
        m5880 = mun.to_crs(5880)
        prox = gpd.sjoin_nearest(pts[fora].to_crs(5880), m5880[["cod_ibge", "municipio", "uf", "geometry"]],
                                 how="left", distance_col="dist_m")
        prox = prox[~prox.index.duplicated()]
        for c in ("cod_ibge", "municipio", "uf"):
            j.loc[fora, c] = prox[c]
        j.loc[fora, "dist_municipio_km"] = prox["dist_m"] / 1000
    j["municipio_por_proximidade"] = fora.astype(int)
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))


if __name__ == "__main__":
    mun = municipios()
    for v in VARS:
        e = estacoes(v, mun)
        e.to_csv(DADOS / f"estacoes_{v}.csv", index=False)
        print(v, len(e), "estações |", int(e.municipio_por_proximidade.sum()), "por proximidade |",
              e.cod_ibge.isna().sum(), "sem município | UFs:", e.uf.nunique())
