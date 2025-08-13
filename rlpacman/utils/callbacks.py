from __future__ import annotations
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)


def build_callbacks(savedir: str, logdir: str, eval_env, eval_freq: int = 10000):
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=2, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=savedir,
        log_path=logdir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=early_stop,
    )
    ckpt_cb = CheckpointCallback(save_freq=5 * eval_freq, save_path=f"{savedir}/ckpts", name_prefix="model")
    return [eval_cb, ckpt_cb]