NCS
===

Neocortical Simulator: An extensible neural simulator for heterogeneous clusters.

#Installation instructions
Instructions available on webpage are outdated, so there are updated instructions
This code compiles successfully using g++-4.7
```
sudo apt-get install g++-4.7 libprotobuf-dev protobuf-compiler flex openmpi-bin graphviz cmake
```
clone the source code and create new directory
```
mkird build
cd build
```
Build and install:
```
cmake -D CMAKE_C_COMPILER=gcc-4.7 -D CMAKE_CXX_COMPILER=g++-4.7 ..
make
sudo make install
```

Notes
-----

Please visit the [NCS website] for more information.


[NCS website]: http://ncs.io/docs/installation/


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/BrainComputationLab/ncs/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

