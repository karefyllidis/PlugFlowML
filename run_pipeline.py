"""Run HydrAI workflow"""
import subprocess, sys

subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_3_data_exploration_feature_engineering.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb"], check=True)
