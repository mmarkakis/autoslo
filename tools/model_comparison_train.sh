python3 /home/markakis/chunkbench/src/autoslo/model_training/train.py --train_config_path /home/markakis/chunkbench/data/model_training_configs/comparison_base.yml
python3 /home/markakis/chunkbench/tools/visualize_model_performance.py comparison_base
python3 /home/markakis/chunkbench/src/autoslo/model_training/train.py --train_config_path /home/markakis/chunkbench/data/model_training_configs/comparison_+size.yml
python3 /home/markakis/chunkbench/tools/visualize_model_performance.py comparison_+size
# python3 /home/markakis/chunkbench/src/autoslo/model_training/train.py --train_config_path /home/markakis/chunkbench/data/model_training_configs/comparison_+censored.yml
# python3 /home/markakis/chunkbench/tools/visualize_model_performance.py comparison_+censored
