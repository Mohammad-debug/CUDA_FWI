//fd_wave.cpp

/*
* Created by: Min Basnet
* 2020.April.17
* Kathmandu, Nepal
*/
#include <iostream>
#include <cmath>

#include "fd_wave.cuh"
#include "util.cu"


void FDWave::allocate_wave(int dimz, int dimx) {
    // Allocates velocity and stress tensors
    allocate_array_2d(vx, dimz, dimx);
    allocate_array_2d(vz, dimz, dimx);
    allocate_array_2d(sxx, dimz, dimx);
    allocate_array_2d(szx, dimz, dimx);
    allocate_array_2d(szz, dimz, dimx);

}


void FDWave::deallocate_wave(int dimz) {
    // Deallocates velocity and stress tensors
    deallocate_array_2d(vx, dimz);
    deallocate_array_2d(vz, dimz);
    deallocate_array_2d(sxx, dimz);
    deallocate_array_2d(szx, dimz);
    deallocate_array_2d(szz, dimz);

}
