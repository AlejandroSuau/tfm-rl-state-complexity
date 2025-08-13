# TFM: Efecto de la complejidad del estado en RL (Pacman)

Este proyecto estudia cómo la **complejidad del estado** afecta al **rendimiento** de algoritmos de aprendizaje por refuerzo (RL) usando un entorno tipo Pacman implementado en Gymnasium y entrenado con Stable-Baselines3.

## Requisitos
- Python 3.10–3.11
- `pip install -r requirements.txt`

## Estructura
Ver árbol en la raíz. Los modos de observación se controlan con `ObsConfig.mode` en el entorno.

## Uso rápido
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Entrenamiento
python scripts/train.py --algo ppo --obs-mode minimal --seed 0 --steps 200000

# Evaluación del modelo guardado
python scripts/evaluate.py --algo ppo --obs-mode minimal --model experiments/runs/ppo_minimal/final_model.zip --episodes 25

# TensorBoard
tensorboard --logdir experiments/logs