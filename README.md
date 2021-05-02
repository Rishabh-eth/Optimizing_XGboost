[![pipeline status](https://gitlab.inf.ethz.ch/COURSE-ASL2020/team039/badges/master/pipeline.svg)](https://gitlab.inf.ethz.ch/COURSE-ASL2020/team039/-/commits/master)

## How to use
### Requirements
* cmake >= 3.15

### Compiling
```
cd xgboost
mkdir build && cd build
cmake ..
make
```

### Run tests
* Compile the code as shown above
* run `./test`

### Run XGBoost
* Compile the code as shown above
* run `./xgboost [path-to-csv-file] [label-col] [feature-col1] [feature-col2] ...`

Note: in CLion you can pass this path as follows
* Go to edit configurations
* Pass the path as following
![edit configurations](/uploads/4fcb7fec15fb68729e4e9891f866283a/Screenshot_2020-04-27_at_16.34.56.png)

