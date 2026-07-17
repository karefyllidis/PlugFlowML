"""Run PlugFlowML workflow"""
import subprocess, sys

#subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_1_data_acquisition.ipynb"], check=True)
#subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_2_data_preprocessing_feature_engineering.ipynb"], check=True)
#subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_3_data_exploration_feature_engineering.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_8_PINN_PFR.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_9_PINN_PFR_tuning.ipynb"], check=True)