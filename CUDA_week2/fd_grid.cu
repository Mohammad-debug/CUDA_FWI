//fd_grid.cpp

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/
#include <iostream>
#include <cmath>

#include "fd_grid.cuh"
#include "INIReader.cuh"
//#include "../ext/inih/INIReader.h"

void FDGrid::parse_configuration(const char* configuration_file_relative_path) {
    //
    std::cout << "parsing FD GRID from file '"
        << configuration_file_relative_path << "'." << std::endl;

    INIReader reader(configuration_file_relative_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load .ini file\n";
        exit(1);
    }
    // Grid
    real_sim t = reader.GetReal("grid", "time", 0); //time in seconds

    nx = reader.GetInteger("grid", "nx", 201);
    //ny = reader.GetInteger("domain", "ny", 0);
    nz = reader.GetInteger("grid", "nz", 101);

    dx = reader.GetReal("grid", "dx", 1.25);
    //dy = reader.GetReal("domain", "dy", 0.0);
    dz = reader.GetReal("grid", "dz", 1.25);
    dt = reader.GetReal("grid", "dt", 0.00025);

    nt = reader.GetInteger("grid", "nt", ceil(t / dt));
    t = (nt - 1) * dt;

    snap_interval = reader.GetInteger("grid", "snap_interval", nt - 1);

    fsurf = reader.GetInteger("grid", "free_surface", 0);

    std::cout << std::endl << "Grid:" << std::endl;
    std::cout << "time = " << t << " (sec)" << std::endl;
    std::cout << "nt = " << nt << ", nx = " << nx << ", nz = " << nz << std::endl;
    std::cout << "dt = " << dt << ", dx = " << dx << ", dz = " << dz << std::endl;
    std::cout << std::endl;

}

void FDGrid::total_grid(int fdorder, int npml) {
    // Calculates the total grid size
    fpad = fdorder / 2;
    ppad = npml + fpad;
    nxt = nx + 2 * ppad;

    if (fsurf) { // existance of free surface
        nzt = nz + ppad + fpad;
    }
    else { // no free surface and PML on the top boundary as well
        nzt = nz + 2 * ppad;
    }


    std::cout << std::endl << "Grid augmentation:" << std::endl;
    std::cout << "fpad = " << fpad << ", ppad = " << ppad << std::endl;
    std::cout << "nxt = " << nxt << ", nzt = " << nzt << std::endl;
    std::cout << std::endl;

}