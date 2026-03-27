import os
import sys

if os.system("jupyter nbconvert --execute --to notebook notebooks/Main_3_data_exploration_feature_engineering.ipynb") != 0: sys.exit(1)
if os.system("jupyter nbconvert --execute --to notebook notebooks/Main_4_train_tree_models.ipynb") != 0: sys.exit(1)
if os.system("jupyter nbconvert --execute --to notebook notebooks/Main_5_tree_models_comparison.ipynb") != 0: sys.exit(1)