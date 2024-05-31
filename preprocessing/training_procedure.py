from box import Box
import yaml
config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)

def decompose_dataframe(df):
    y_all_features = config_goce.y_all_feature_keys
    z_all_features = config_goce.z_all_feature_keys
    y_all = df[y_all_features]
    z_all = df[z_all_features]
    print("z_all: ", z_all.shape)
    print("z_all: ", z_all.columns)
    x_all = df.drop(z_all_features, axis=1).drop(y_all_features, axis=1)

    return (x_all, y_all, z_all)
