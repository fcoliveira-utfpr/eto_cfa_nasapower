#!/bin/bash
# Relança o orquestrador após erros transitórios de rede (as etapas são idempotentes).
for i in $(seq 1 30); do
  PYTHONIOENCODING=utf-8 python -u orquestrar.py Tmax Tmin RH Rs u2 pr >> ../nao_subir/logs/orquestrar.log 2>&1 && exit 0
  echo "$(date +%T) tentativa $i falhou; nova tentativa em 60 s" >> ../nao_subir/logs/orquestrar.log
  sleep 60
done
exit 1
