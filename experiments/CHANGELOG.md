# 25.08.2025 – 1M pasos, shaping alineado (minimal vs power_time), 4 semillas

## Qué se ha hecho

* Entrenamiento con VecNormalize y recompensas alineadas
(coin=3.0, power=6.0, eat_ghost=8.0, clear=120.0, power_tick=0, coin_power_bonus=1.2).
* Modos evaluados: minimal y power_time. Episodios de eval: 50.

## Motivación

* Evitar “farmear power” y empujar a recoger monedas y terminar el nivel.
* Ver si la mayor complejidad del estado (tiempo de power) aporta ventaja cuando la recompensa está alineada.

## Resultados

* completion_ratio: minimal > power_time (≈ 0.68 vs 0.62).
* near_clear_rate: ≈ 10–12% en ambos (antes era ~0–2%).
* success_rate: 0 (posible límite de max_steps=600).
* mean_reward: power_time > minimal.

## Conclusión breve

* El shaping nuevo mejoró mucho el progreso (near-clear).
* power_time monetiza más (recompensa) pero convierte ligeramente peor ese puntuar en limpiar antes del límite de pasos.
* Siguiente paso: PPO multi-entorno para estabilidad/eficiencia; opcionalmente DQN piloto para comparar algoritmo.
* 