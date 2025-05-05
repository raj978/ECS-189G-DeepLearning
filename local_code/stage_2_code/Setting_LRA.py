"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# import numpy as np

from local_code.base_class.setting import setting


class Setting_LRA(setting):
    def load_run_save_evaluate(self):
        # load dataset
        train_set = self.dataset.load(train=True)
        test_set = self.dataset.load(train=False)

        self.method.data = {"train": train_set, "test": test_set}
        history, learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()

        self.result.save_metrics(metrics)

        self.evaluate.gen_learning_curve(history)

        return metrics
