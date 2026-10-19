"""Seleciona as estações do domínio Cfa com dados suficientes para o estudo.

1. Classe de Köppen de cada estação com radiação solar no mapa de Alvares et al. (2013) na versão de 100 m do
   IPEF (asset do GEE); pixels sem classe (litoral) recebem a classe vizinha num raio de 500 m.
2. Número de anos completos entre 2001-01-01 e 2020-07-31, contando apenas os dias com as cinco variáveis
   (Tmax, Tmin, RH, Rs, u2) observadas na estação.
3. Estações Cfa com pelo menos 5 anos completos (1.827 dias).
4. Recorte do mapa Cfa em grade de 0,04° para a Figura 1.
Saídas em dados/: koppen_estacoes.csv, estacoes_cfa.csv e cfa_ipef_004.npz
"""
import ee
import numpy as np
import pandas as pd

from caminhos import DADOS, OBS, PROJETO_EE

KOPPEN = "projects/mapbiomas-workspace/SOLOS/COVARIAVEIS/IPEF_2013_KOPPEN_100M_2025"
INI, FIM = "2001-01-01", "2020-07-31"
ANOS_MIN = 5

ee.Initialize(project=PROJETO_EE)


def koppen_estacoes(meta):
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"id": r.cod_estacao})
                               for r in meta.itertuples()])
    im = ee.Image(f"{KOPPEN}/koppen_l3").addBands(ee.Image(f"{KOPPEN}/koppen_l2"))
    im = im.unmask(im.focalMax(500, "circle", "meters"))
    df = ee.data.computeFeatures({"expression": im.reduceRegions(fc, ee.Reducer.first(), scale=100),
                                  "fileFormat": "PANDAS_DATAFRAME"}).set_index("id")
    l3 = [c for c in df.columns if c.startswith("koppen_l3_")]
    l2 = [c for c in df.columns if c.startswith("koppen_l2_")]

    def rotulo(row, cols, pre):
        v = row[cols].astype(float)
        return v.idxmax().removeprefix(pre) if v.max() == 1 else None

    return pd.Series([rotulo(r, l3, "koppen_l3_") or rotulo(r, l2, "koppen_l2_") for _, r in df.iterrows()],
                     index=df.index, name="ipef")


def anos_completos():
    dias = pd.date_range("1961-01-01", "2025-12-31")
    sel = (dias >= INI) & (dias <= FIM)
    ok = None
    for v in ["Tmax", "Tmin", "RH", "Rs", "u2"]:
        z = np.load(OBS / f"{v}.npz", allow_pickle=True)
        d = pd.DataFrame(~np.isnan(z["data"][sel]), columns=z["ID"].astype(str))
        if ok is None:
            ok = d
        else:
            comuns = ok.columns.intersection(d.columns)
            ok = ok[comuns] & d[comuns]
    return (ok.sum() / 365.25).rename("anos_completos_2001_2020")


def mapa_cfa(x0=-58.5, y0=-13.0, x1=-38.5, y1=-34.0, dx=0.04):
    img = ee.Image(f"{KOPPEN}/koppen_l3").select("koppen_l3_Cfa").unmask(0)
    nx, ny = int((x1 - x0) / dx), int((y0 - y1) / dx)
    a = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": {
        "dimensions": {"width": nx, "height": ny}, "crsCode": "EPSG:4326",
        "affineTransform": {"scaleX": dx, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -dx,
                            "translateY": y0}}})
    np.savez_compressed(DADOS / "cfa_ipef_004.npz", cfa=np.array(a["koppen_l3_Cfa"], dtype=np.uint8),
                        x0=x0, y0=y0, dx=dx)


if __name__ == "__main__":
    meta = pd.read_csv(DADOS / "estacoes_Rs.csv", dtype={"cod_estacao": str, "cod_ibge": str})
    k = (meta.set_index("cod_estacao")[["uf", "municipio", "cod_ibge", "lat", "lon", "alt"]]
         .join(koppen_estacoes(meta)).join(anos_completos()))
    k.to_csv(DADOS / "koppen_estacoes.csv")
    s = k[(k.ipef == "Cfa") & (k.anos_completos_2001_2020 >= ANOS_MIN)].reset_index()
    s["tipo"] = ["automática" if c.startswith("A") else "convencional" for c in s.cod_estacao]
    s.to_csv(DADOS / "estacoes_cfa.csv", index=False)
    mapa_cfa()
    print(f"{(k.ipef == 'Cfa').sum()} estações Cfa; {len(s)} com >= {ANOS_MIN} anos completos;",
          s.uf.value_counts().to_dict())
