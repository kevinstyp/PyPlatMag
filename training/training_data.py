
def split_dataframe(data, y_features, meta_features):
    y_all = data[y_features]
    z_all = data[meta_features]
    x_all = data.drop(meta_features, axis=1).drop(y_features, axis=1)

    return (x_all, y_all, z_all)