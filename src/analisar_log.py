import csv
import os
import sys

pasta = os.path.dirname(os.path.abspath(__file__))
padrao = os.path.join(os.path.dirname(pasta), "infer_log.csv")
path = sys.argv[1] if len(sys.argv) > 1 else padrao

if not os.path.exists(path):
    sys.exit(f"Arquivo nao encontrado: {path}\nInicie o app modificado e clique em Iniciar Deteccao.")

with open(path) as f:
    rows = list(csv.DictReader(f))[3:]

n = len(rows)
if n < 2:
    sys.exit(f"Poucos quadros no log ({n}). Deixe a deteccao rodando por mais tempo.")

t = [float(r["t"]) for r in rows]
inf = sorted(float(r["infer_ms"]) for r in rows)
tot = [float(r["total_ms"]) for r in rows]

dur = t[-1] - t[0]
fps = (n - 1) / dur

print(f"Arquivo            : {path}")
print(f"Quadros analisados : {n}")
print(f"Duracao            : {dur/60:.1f} min")
print(f"FPS medio          : {fps:.2f}  ({'atinge' if fps >= 0.5 else 'NAO atinge'} 0.5 FPS)")
print(f"Inferencia (ms)    : media {sum(inf)/n:.1f} | p95 {inf[int(0.95*(n-1))]:.1f} | max {inf[-1]:.1f}")
print(f"Tempo por quadro   : media {sum(tot)/n:.1f} ms")
