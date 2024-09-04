
# TODO Rename this file

def split_dataframe(data, y_features, meta_features):
    y_all = data[y_features]
    z_all = data[meta_features]
    x_all = data.drop(meta_features, axis=1).drop(y_features, axis=1)
    print("x_all.columns: ", list(x_all.columns))
    print("y_all.columns: ", list(y_all.columns))
    print("z_all.columns: ", list(z_all.columns))
    weightings = z_all['weightings']

    return (x_all, y_all, z_all, weightings)


