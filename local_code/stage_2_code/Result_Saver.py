"""
Concrete ResultModule class for a specific experiment ResultModule output
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print("saving results...")
        save_path = os.path.join(
            self.result_destination_folder_path, self.result_destination_file_name
        )
        f = open(save_path, "wb")
        pickle.dump(self.data, f)
        f.close()
