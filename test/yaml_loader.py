import yaml

with open("../crowd_count/config/SHTB_ResNet101.yaml", 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config["MODEL"])
    print(type(config))
