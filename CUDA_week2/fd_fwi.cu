//fd_wave.cpp

/*
* Created by: Min Basnet
* 2020.April.17
* Kathmandu, Nepal
*/
#include <iostream>
#include <cmath>

#include "fd_fwi.cuh"
#include "fd_grid.cuh"
#include "util.cu"


void FDFWI::parse_configuration(const char* configuration_file_relative_path, FDGrid grid) {

    std::cout << "parsing FD FWI from file '"
        << configuration_file_relative_path << "'." << std::endl;

    INIReader reader(configuration_file_relative_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load .ini file\n";
        exit(1);
    }

    // PML boundary (static members)
    dt = reader.GetInteger("full_waveform_inversion", "time_interval", 1);
    dx = reader.GetInteger("full_waveform_inversion", "grid_interval_x", 1);
    dz = reader.GetInteger("full_waveform_inversion", "grid_interval_y", 1);

    x1 = reader.GetInteger("full_waveform_inversion", "left_grid", 0);
    x2 = reader.GetInteger("full_waveform_inversion", "right_grid", 0);
    z1 = reader.GetInteger("full_waveform_inversion", "top_grid", 0);
    z2 = reader.GetInteger("full_waveform_inversion", "bottom_grid", 0);

    if (z2 == 0) z2 = grid.nzt - 1;
    if (x2 == 0) x2 = grid.nxt - 1;

    // Padding of the grids from fdorder and PML layers
    x1 += grid.ppad;
    x2 += grid.ppad;

    if (grid.fsurf) {
        z1 += grid.fpad;
        z2 += grid.fpad;
    }
    else {
        z1 += grid.ppad;
        z2 += grid.ppad;
    }


    std::cout << std::endl << "FWI:" << std::endl;
    std::cout << "fwi.dt = " << dt << ", fwi.dx = " << dx << ", fwi.dz = " << dz << std::endl;
    std::cout << "fwi.x1 = " << x1 << ", fwi.x2 = " << x2 << std::endl;
    std::cout << "fwi.z1 = " << z1 << ", fwi.z2 = " << z2 << std::endl;
    std::cout << std::endl;


}



void FDFWI::allocate_fwi(int dimt, int dimz, int dimx) {
    // Allocates velocity and stress tensors
    allocate_array_3d(vx, dimt, dimz, dimx);
    allocate_array_3d(vz, dimt, dimz, dimx);

    allocate_array_3d(sxx, dimt, dimz, dimx);
    allocate_array_3d(szx, dimt, dimz, dimx);
    allocate_array_3d(szz, dimt, dimz, dimx);

    allocate_array_2d(grad_lam, dimz, dimx);
    allocate_array_2d(grad_mu, dimz, dimx);
    allocate_array_2d(grad_rho, dimz, dimx);

}

void FDFWI::reset_fwi_kernel(int dimt, int dimz, int dimx, bool grad_reset) {
    // Reset velocity and stress tensors
    reset_array_3d(vx, dimt, dimz, dimx);
    reset_array_3d(vz, dimt, dimz, dimx);

    reset_array_3d(sxx, dimt, dimz, dimx);
    reset_array_3d(szx, dimt, dimz, dimx);
    reset_array_3d(szz, dimt, dimz, dimx);

    if (grad_reset) {

        reset_array_2d(grad_rho, dimz, dimx);
        reset_array_2d(grad_lam, dimz, dimx);
        reset_array_2d(grad_mu, dimz, dimx);
    }

}


void FDFWI::deallocate_fwi(int dimt, int dimz) {
    // Deallocates velocity and stress tensors

    deallocate_array_3d(vx, dimt, dimz);
    deallocate_array_3d(vz, dimt, dimz);
    deallocate_array_3d(sxx, dimt, dimz);
    deallocate_array_3d(szx, dimt, dimz);
    deallocate_array_3d(szz, dimt, dimz);

    deallocate_array_2d(grad_lam, dimz);
    deallocate_array_2d(grad_mu, dimz);
    deallocate_array_2d(grad_rho, dimz);

}