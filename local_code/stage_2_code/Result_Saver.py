"""
Concrete ResultModule class for a specific experiment ResultModule output
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import json
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_file_path = ""
    metrics_path = ""

    def save(self):
        print("saving results...")
        f = open(self.result_destination_file_path, "wb")
        pickle.dump(self.data, f)
        f.close()

    def save_metrics(self, metrics):
        print("saving metrics...")
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f)
