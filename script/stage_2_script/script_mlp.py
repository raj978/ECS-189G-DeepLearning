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

    params = [
        # small vs. wide with Cross-Entropy
        ["small", "cross", "adam"],
        ["wide", "cross", "adam"],
        ["small", "cross", "sgd"],
        ["wide", "cross", "sgd"],
        # final: best (arch,opt) from above → test with MAE (L1Loss)
        # e.g. if “wide + adam” wins, uncomment this next line:
        # ["wide", "mae", "adam"],
    ]

    for i, (arch, loss_fn, opt) in enumerate(params):
        print(f"{i+1}th model", arch, loss_fn, opt)

        model_name = f"mlp_{arch}_{loss_fn}_{opt}"

        print("model name:", model_name)

        # ---- objection initialization setction ---------------
        data_obj = Dataset_Loader("MLP", "")
        dataset_source_folder_path = "../../data/stage_2_data/"
        data_obj.dataset_train_file_path = os.path.join(
            dataset_source_folder_path, "train.csv"
        )
        data_obj.dataset_test_file_path = os.path.join(
            dataset_source_folder_path, "test.csv"
        )

        os.makedirs(dataset_source_folder_path, exist_ok=True)

        method_obj = Method_MLP("multi-layer perceptron", "")
        model_save_dir = "../../models/stage_2"
        method_obj.model_path = os.path.join(
            model_save_dir, f"{model_name}_full_model.pt"
        )
        method_obj.hist_path = os.path.join(
            model_save_dir, f"{model_name}_history.json"
        )
        method_obj.architecture = arch
        method_obj.opt = opt
        method_obj.loss_fn = loss_fn

        os.makedirs(model_save_dir, exist_ok=True)

        result_obj = Result_Saver("saver", "")
        result_destination_folder_path = "../../result/stage_2_result/MLP_"
        result_obj.result_destination_file_path = os.path.join(
            result_destination_folder_path, f"{model_name}_prediction_result"
        )
        result_obj.metrics_path = os.path.join(
            result_destination_folder_path, f"{model_name}_metrics.json"
        )

        os.makedirs(result_destination_folder_path, exist_ok=True)

        evaluate_obj = Evaluate_Metrics("metrics", "")
        plot_save_dir = "../../figures/stage_2"
        evaluate_obj.plot_path = os.path.join(
            plot_save_dir, f"{model_name}_learning_curve.png"
        )

        os.makedirs(plot_save_dir, exist_ok=True)

        setting_obj = Setting_LRA("validation", "")

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
