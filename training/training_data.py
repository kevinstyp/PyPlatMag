
# TODO Rename this file

def split_dataframe(data, y_features, meta_features):
    y_all = data[y_features]
    z_all = data[meta_features]
    x_all = data.drop(meta_features, axis=1).drop(y_features, axis=1)
    print("x_all.columns: ", x_all.columns.tolist())
    print("y_all.columns: ", y_all.columns.tolist())
    print("z_all.columns: ", z_all.columns.tolist())
    weightings = z_all['weightings'].copy()

    return (x_all, y_all, z_all, weightings)


