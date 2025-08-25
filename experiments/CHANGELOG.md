# 25.08.2025 – Shaping alineado (fase A), VecNormalize activado

## Cambios

* Recompensas:
coin=3.0, power=6.0, eat_ghost=8.0, clear=120.0, death=-30.0, step=0.0,
power_tick=0.0 (eliminado), coin_power_bonus=1.2 (↑).
* Train/Eval con VecNormalize (obs normalizadas; reward crudo en eval).
* Comparativa focal: minimal vs power_time (300k pasos).

## Motivación

* Evitar “farmear power” y empujar a recoger monedas y terminar el nivel.
* Ver si la mayor complejidad del estado (tiempo de power) aporta ventaja cuando la recompensa está alineada.

## Resultados

* completion_ratio: power_time > minimal (≈ 0.57 vs 0.44).
* near_clear_rate: power_time > 0 (~2%); minimal = 0.
* mean_reward: power_time claramente superior.
* success_rate (100%): 0 en ambos.

## Conclusión breve

* El shaping alineado hace que la info de power sí ayude a progresar (no solo a puntuar).
* Siguiente: replicar con semillas; si se mantiene, escalar a 1M pasos (minimal vs power_time) para resultados finales.