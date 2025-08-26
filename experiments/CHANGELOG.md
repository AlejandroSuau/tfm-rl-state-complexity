# 26.08.2025 – 1M pasos, shaping alineado (minimal vs power_time), 4 semillas

## Qué se añadió / cambió (esta tanda):

* Recompensas re-balanceadas: monedas bajas, muerte más costosa, paso negativo suave, clear muy alto. VecNormalize activo y evaluación con las stats correctas.
* Barrido 3M ts con 8 envs y rollouts largos (n_steps=1024, batch=8192, ent_coef=0.005), 4 modos y 4 seeds.
* Resultado: sube el completion ratio (~0.38–0.39, mejor en coins_quadrants) pero sin clears.

## Conclusiones:

* El shaping actual fomenta recoger monedas pero no da señal de fin; el agente se estabiliza en “coger bastantes y morir/tiempo”.
* Más timesteps sin cambiar señal rara vez desbloquea clears.

## Acciones recomendadas:

* Bonus de últimos coins (señal de final → debería subir near_clear_rate).
* Dirección al coin más cercano (coins_quadrants_dir) para guiar la ruta.
* Si hace falta, bajar leve la agresividad del fantasma como mini-currículum.
