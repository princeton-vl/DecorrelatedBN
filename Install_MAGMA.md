
MAGMA installation
==
MAGMA official website: http://icl.cs.utk.edu/magma/

1. Download MAGMA from http://icl.cs.utk.edu/magma/software/index.html (we used the version 2.0.2 in our paper).
2. Extract the files and choose a `make.inc.*` file depending on which BLAS you have installed (MKL, ATLAS, or OpenBLAS). We used OpenBLAS in the paper.
3. Copy the correct makefile to make.inc, and edit the file according to your need.
4. A common configurations is listed here. For more information, please see README in the MAGMA source.
    - `export CUDADIR=/usr/local/cuda`(your CUDA path) in your `~/.bashrc` or `~/.zshrc`
    - `export OPENBLASDIR=/opt/openblas` (your BLAS path) in your `~/.bashrc` or `~/.zshrc`
e.g.:  `export CUDADIR=/usr/local/cuda-8.0`
       `export OPENBLASDIR=/home/huanlei/opt/openblas-0.2.18`

       You can also include the configurations in the make.inc file.
5. run `make` in the MAGMA source directory.
6. run `make install`  or `sudo make install` in the same directory.
