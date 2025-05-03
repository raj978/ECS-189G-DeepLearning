"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# import numpy as np

import json
import os

from local_code.base_class.setting import setting


class Setting_LRA(setting):
    model_save_dir = None
    curve_save_dir = None
    result_save_dir = None

    def load_run_save_evaluate(self):
        # load dataset
        train_set = self.dataset.load(train=True)
        test_set = self.dataset.load(train=False)

        self.method.data = {"train": train_set, "test": test_set}
        history, learned_result = self.method.run(model_save_dir=self.model_save_dir)

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()

        save_path = os.path.join(self.result_save_dir, "metrics.json")
        with open(save_path, "w") as f:
            json.dump(metrics, f)

        save_dir = os.path.join(self.curve_save_dir, "stage_2")
        save_path = os.path.join(save_dir, "mlp_learning_curve.png")
        os.makedirs(save_dir, exist_ok=True)

        self.evaluate.gen_learning_curve(history, save_path)

        return metrics
