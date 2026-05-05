"""Run HydrAI training notebooks in order."""
import subprocess, sys

subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_3_data_exploration_feature_engineering.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_4_train_tree_models.ipynb"], check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/Main_4b_tree_models_comparison.ipynb"], check=True)
