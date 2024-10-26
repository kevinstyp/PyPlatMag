## PyPlatMag
**PyPlatmag** is a Python-based tool designed for the (re-)calibration of platform magnetometers aboard satellite missions. 
It provides a machine learning approach with a physics-informed layer to automatically correct artificial disturbances caused by electric current-induced magnetic fields from satellite systems. 
PyPlatmag offers enhanced calibration accuracy, allowing users to extend the spatiotemporal coverage of geomagnetic field measurements using data from non-dedicated geomagnetic satellite missions.

## Features
- Automatic post-launch calibration of platform magnetometer data.
- Incorporates a physics-informed layer based on the Biot-Savart formula for the co-estimation of magnetic dipoles.
- Currently, supports calibration of platform magnetometers from the GOCE satellite missions.
- Modular design for easy extension to other satellite missions with platform magnetometer data.

## Requirements
Tested / developed versions
- Python 3.8
- Required libraries:
  - NumPy 1.24.3
  - Pandas 2.0.3
  - TensorFlow 2.13.1

## Installation
Clone the repository and install the dependencies manually:
```bash
git clone https://github.com/yourusername/pyplatmag.git
cd pyplatmag
pip install -r requirements.txt
```

## Quick Start Guide
1. First, download the GOCE platform magnetometer data.
The format of the data is described in detail in the Data section.
2. Update the config.yaml and config_goce.yaml files with the correct paths to the data and other important paths listed next:
#### config.yaml
```yaml
# Calibration months
year_month_specifiers: ['200911', '200912']
# Path to the raw / input data
goce_data_path: /home/user/data/GOCE/rawdata/
# Path for the processed data
write_path: /home/user/data/
# Path to the CDF library 'lib' and 'bin' folders
CDF_LIB: "/home/user/cdf-lib/cdf38_0-dist/lib"
CDF_BIN: "/home/user/cdf-lib/cdf38_0-dist/bin"
```
#### config_goce.yaml
```yaml
# config_goce.yaml
# Path to write the calibration results as cdf files
cdf_config:
  cdf_path: '/home/user/cdf_pyplatmag/'
```

3. Then, run the calibration scripts:
```bash
python read_files.py
python read_files_nan_handling.py
python train_pinn.py
python generate_cdf_files.py
```

## Customizing for other satellite missions

TODO

## API Documentation

TODO: Do we want to do this? Needs thorough docstring documentation in the code.

## License

We use the MIT license, see LICENSE file.


## Data

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
