conda env create -f environment.yaml
pip install -r requirements.txt

wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/latest/linux/cdf39_0-dist-all.tar.gz && \
tar -xvf cdf39_0-dist-all.tar.gz
cd ./cdf39_0-dist
make OS=linux ENV=gnu CURSES=yes all
make test
make install
make clean

CDF_BIN=./cdf39_0-dist/bin/