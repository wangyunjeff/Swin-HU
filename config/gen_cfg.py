import config
import yaml

print(config.cfg)
with open('./cfg_jasper_dataset_cnn_vit_init.yaml', "w") as f:
    yaml.dump(yaml.load(config.cfg.dump(), Loader=yaml.FullLoader), f)