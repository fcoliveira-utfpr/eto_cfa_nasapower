"""Cálculo da ETo (PM-FAO56 e métodos empíricos) com o agrometeorologiapy."""
import numpy as np
import pandas as pd

import agrometeorologiapy as agm


def ra_astronomica(datas, lat):
    nda = pd.DatetimeIndex(datas).dayofyear.to_numpy()
    dec = agm.declinacao_solar(nda)
    hn = agm.angulo_horario_nascer(lat, dec)
    return agm.irradiancia_extraterrestre(lat, dec, hn, agm.fator_correcao_distancia(nda))


def auxiliares(Tmax, Tmin, UR, Rs, Ra, alt):
    Tmed = agm.temp_media_extremos(Tmax, Tmin)
    es = (agm.es_tetens(Tmax) + agm.es_tetens(Tmin)) / 2
    ea = agm.ea_umidade(es, UR)
    Delta = agm.declive_pressao_vapor(Tmed)
    gamma = agm.constante_psicrometrica(agm.patm_altitude(alt))
    Rso = (0.75 + 2e-5 * alt) * Ra
    Rn = agm.saldo_radiacao(agm.boc_saldo(Rs, r=0.23), agm.bol_saldo(Tmax, Tmin, ea, np.minimum(Rs, Rso), Rso))
    return Tmed, es, ea, Delta, gamma, Rn


def metodos(Tmax, Tmin, UR, Rs, u2, Ra, alt):
    """ETo (mm/d) pelo PM-FAO56 e pelos cinco métodos empíricos, com coeficientes originais."""
    Tmed, es, ea, Delta, gamma, Rn = auxiliares(Tmax, Tmin, UR, Rs, Ra, alt)
    turc = 0.31 * Tmed / (Tmed + 15) * (Rs + 2.09)
    turc = np.where(UR < 50, turc * (1 + (50 - UR) / 70), turc)
    return pd.DataFrame({
        "PM": agm.eto_penman_monteith_fao56(Rn, 0, Tmed, u2, es, ea, Delta, gamma),
        "HS": agm.etp_hargreaves_samani(Ra, Tmax, Tmin, Tmed),
        "Cam": 0.01 * 0.408 * Ra * Tmed,
        "Tur": turc,
        "PT": agm.etp_priestley_taylor(Rn, 0, Delta, gamma),
        "Mak": 0.61 * Delta / (Delta + gamma) * Rs / 2.45 - 0.12,
    })
