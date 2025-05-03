import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))


from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_LRA import Setting_LRA

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader("MLP", "")
    data_obj.dataset_source_folder_path = "../../data/stage_2_data/"
    data_obj.dataset_train_file_name = "train.csv"
    data_obj.dataset_test_file_name = "test.csv"

    method_obj = Method_MLP("multi-layer perceptron", "")

    result_obj = Result_Saver("saver", "")
    result_obj.result_destination_folder_path = "../../result/stage_2_result/MLP_"
    result_obj.result_destination_file_name = "prediction_result"

    os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)

    evaluate_obj = Evaluate_Metrics("metrics", "")

    setting_obj = Setting_LRA("validation", "")
    setting_obj.model_save_dir = "../../models"
    setting_obj.curve_save_dir = "../../figures"
    setting_obj.result_save_dir = result_obj.result_destination_folder_path

    os.makedirs(setting_obj.model_save_dir, exist_ok=True)
    os.makedirs(setting_obj.curve_save_dir, exist_ok=True)

    # ------------------------------------------------------

    # ---- running section ---------------------------------

    print("************ Start ************")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    # edit this one for measurements
    metrics = setting_obj.load_run_save_evaluate()

    print("************ Overall Performance ************")
    print("MLP Accuracy: " + str(metrics["accuracy"]))
    print("************ Finish ************")
    # ------------------------------------------------------
