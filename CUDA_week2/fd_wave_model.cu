
/*
* Created by: Min Basnet
* 2020.April.16
* Kathmandu, Nepal
*/

#include <iostream>
#include <fstream>
#include <cmath>

#include "fd_wave_model.cuh"
#include "globvar.cuh"
#include "INIReader.cuh"
//#include "../ext/inih/INIReader.h"
#include "fd_kernel.cuh"
#include "fd_fwi.cuh"

std::ofstream outReceiverFile; // output file to store receiver data

FDWaveModel::FDWaveModel(const char* configuration_file_relative_path) {
    // Constructor for creation of FD Grid

    std::cout << "FD Wave Model: Loading configuration file: '"
        << configuration_file_relative_path << "'." << std::endl;

    // Parsing FDM, grid, medium and Cpml from configuration file
    parse_configuration(configuration_file_relative_path);
    grid.parse_configuration(configuration_file_relative_path);
    medium.parse_configuration(configuration_file_relative_path);
    pml.parse_configuration(configuration_file_relative_path);

    // Add the grids (main grids + PML layers + FD order computational grids)
    grid.total_grid(fdorder, pml.npml);

    source.parse_configuration(configuration_file_relative_path, 's', grid);
    receiver.parse_configuration(configuration_file_relative_path, 'r', grid);



    // Modify grid numbers for source and receivers as per the total grid modification
    source.grid_adjustment(grid);
    receiver.grid_adjustment(grid);

    if (fwinv) { // full waveform inversion 

        fwi.parse_configuration(configuration_file_relative_path, grid);
    }


    // Model initialization (out of fwi loop)
    // Allocation for model initialization (CPU allocation and copied to  GPU in GPU model)
    if (pml.npml > 0) {
        pml.allocate_pml(); // Allocation of PML array
        pml.cpml_pro(grid.dt);  // Computation of PML parameters (static parameters)
    }


    std::cout << "Model Generated " << std::endl;
}



FDWaveModel::~FDWaveModel() {
    // Destruction of the model

    // GPU allocated arrays (In GPU parallelization)
    if (pml.npml > 0) {
        pml.deallocate_memory_arrays(grid.nzt); // Allocates the PML memory for velocity and derivatives
    }

    wave.deallocate_wave(grid.nzt); // deallocate wave velocity and stress tensors
    medium.deallocate_medium_av(grid.nzt - 1); // Deallocation of medium average parameters

    // Deallocation in CPU (In GPU programming also)
    medium.deallocate_medium(grid.nzt); // Deallocation of medium parameter over grid

    if (pml.npml > 0) {
        pml.deallocate_pml(); // Deallocate PML arrays in the end
    }

    if (fwinv) { // full waveform inversion 

        fwi.deallocate_fwi(1 + (grid.nt - 1) / fwi.dt, 1 + (fwi.z2 - fwi.z1) / fwi.dz);
    }

}



void FDWaveModel::parse_configuration(const char* configuration_file_relative_path) {
    //
    std::cout << "parsing FDMfrom file '"
        << configuration_file_relative_path << "'." << std::endl;

    INIReader reader(configuration_file_relative_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load .ini file\n";
        exit(1);
    }
    // FDM
    fdorder = reader.GetInteger("FDM", "fdorder", 2);
    fwinv = reader.GetInteger("FDM", "full_waveform_inversion", 0);
    nshots = reader.GetInteger("FDM", "nshots", 1);

    std::cout << "FDM:" << std::endl;
    std::cout << "fdorder = " << fdorder << " , full_waveform_inversion = " << fwinv
        << ", nshots = " << nshots << std::endl;


    if (fwinv) {
        maxIter = reader.GetInteger("FDM", "max_iteration", 500);
        maxError = reader.GetReal("FDM", "max_error", 1.0e-3);
        std::cout << "maxIter = " << maxIter << ", maxError = " << maxError << std::endl << std::endl;

    }

}



void FDWaveModel::forward_simulate() {
    // Forward simulation only
    // The initialization of the first model has to be done before it
    // becomes also part of fwi simulation

    std::cout << "Forward Simulation." << std::endl;
    real_sim hc[2] = { 1.0, 1.0 };

    medium.allocate_medium(grid.nzt, grid.nxt); // Initialization of density and Lame's parameters

    // Initial computation outside of FWI loop 
    medium.velocity_to_lame(); // Creating scalar Lame's parameters from scalar wave velocities
    medium.stagger_over_grid(grid.nzt, grid.nxt); // Initializing density and lame's parameter over staggered grid

    bool gpu_comp = false;

    if (gpu_comp == true) {
        // Bifurcation towards GPU computation
        // Do not pass any classes, just use the class variables
        // for example if it is pml.npml then use variable name as pml_npml

    }
    else { // Computation only in CPU

      // Allocation for parallel processing (For GPU model directly in GPU)
        medium.allocate_medium_av(grid.nzt - 1, grid.nxt - 1); // Harmonic and arithmatic average of medium parameters
        std::cout << "Allocating wave velocity and stress arrays." << std::endl;
        wave.allocate_wave(grid.nzt, grid.nxt);

        // Allocation of PML memory arrays
        if (pml.npml > 0) {
            pml.allocate_memory_arrays(grid.nzt, grid.nxt); // Allocates the PML memory for velocity and derivatives
        }

        for (unsigned int ishot = 0; ishot < nshots; ishot++) {


            // Averaging of material parameters
            medium.average_parameters(grid.nzt - 1, grid.nxt - 1);

            std::cout << std::endl << "Forward time integration [SHOT " << ishot << "]" << std::endl;

            // calling forward simulation for shot 1
            forward_kernel_PSV(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
                grid.dt, grid.dx, grid.dz, grid.snap_interval, grid.fsurf,
                hc, fdorder, wave.vx, wave.vz, wave.sxx, wave.szx, wave.szz,
                medium.lam, medium.mu, medium.mu_zx, medium.rho_zp, medium.rho_xp,
                pml.npml, pml.a, pml.b, pml.K, pml.a_half, pml.b_half, pml.K_half,
                pml.mem_vx_x, pml.mem_vx_z, pml.mem_vz_x, pml.mem_vz_z,
                pml.mem_sxx_x, pml.mem_szx_x, pml.mem_szz_z, pml.mem_szx_z,
                source.nseis, source.x_seis, source.z_seis,
                source.seis_comp, source.signal, source.shot_to_fire,
                receiver.nseis, receiver.x_seis, receiver.z_seis,
                receiver.seis_comp, receiver.signal, receiver.shot_to_fire,
                fwinv, fwi.dt, fwi.dx, fwi.dz, fwi.x1, fwi.x2, fwi.z1, fwi.z2,
                fwi.vx, fwi.vz, fwi.sxx, fwi.szx, fwi.szz);

            std::cout << "SHOT " << ishot << " COMPLETED." << std::endl << std::endl;

        }

        outReceiverFile.open("./io/receiver/rec_signal.csv");
        // Writing ASCCI output at the receivers
        for (unsigned int ir = 0; ir < receiver.nseis; ir++) {
            //
            for (int it = 0; it < grid.nt; it++) {
                outReceiverFile << receiver.signal[ir][it] << ", ";
            }
            outReceiverFile << std::endl;
        }
        outReceiverFile.close();

    }

}


void FDWaveModel::fwi_simulate() {
    // Forward simulation only
    // The initialization of the first model has to be done before it
    // becomes also part of fwi simulation

    std::cout << "Full Waveform Inversion simulation" << std::endl;
    real_sim hc[2] = { 1.0, 1.0 };

    const int nft = 1 + (grid.nt - 1) / fwi.dt;
    const int nfz = 1 + (fwi.z2 - fwi.z1) / fwi.dz;
    const int nfx = 1 + (fwi.x2 - fwi.x1) / fwi.dx;
    //Time stamp
    clock_t start = 0, end = 0;
    double start1 = 0;
    double cpu_time_used;

    //***********************
    medium.allocate_medium(grid.nzt, grid.nxt); // Initialization of density and Lame's parameters

    // Initial computation outside of FWI loop 
    medium.velocity_to_lame(); // Creating scalar Lame's parameters from scalar wave velocities
    medium.stagger_over_grid(grid.nzt, grid.nxt); // Initializing density and lame's parameter over staggered grid

    bool gpu_comp = false;

    if (gpu_comp == true) {
        // Bifurcation towards GPU computation
        // Do not pass any classes, just use the class variables
        // for example if it is pml.npml then use variable name as pml_npml

    }
    else { // Computation only in CPU

      // Allocation for parallel processing (For GPU model directly in GPU)
        medium.allocate_medium_av(grid.nzt - 1, grid.nxt - 1); // Harmonic and arithmatic average of medium parameters

        wave.allocate_wave(grid.nzt, grid.nxt); // Allocation of wave dynamic parameters array

        // Allocation of PML memory arrays
        if (pml.npml > 0) {
            pml.allocate_memory_arrays(grid.nzt, grid.nxt); // Allocates the PML memory for velocity and derivatives
        }

        // Allocation of FWI related parameters
        fwi.allocate_fwi(nft, nfz, nfx);


        // ------------------------------------------
        // Starting of full waveform inversion loop
        // ------------------------------------------

        int iter = 0;
        real_sim diff_L2 = 1.0;
       
        //cudaSetDevice(0);
        start1 = clock();
       
            maxIter = 1;//iteration set
           
        while (diff_L2 >= maxError) {

            if (iter >= maxIter) {
                end = clock();

                cpu_time_used = ((double)(end - start1)) / CLOCKS_PER_SEC;

                printf("\n TOTAL PROGRAM Execution time before exit a  = %f TOTAL ITERATIONS=%d\n", cpu_time_used, maxIter);

                std::cout << "Error: convergence not reached within maximum iteration" << std::endl;
                break;
              
            }

            std::cout << std::endl << std::endl;
            std::cout << std::endl << "Full Waveform Inversion simulation. Iteration step " << iter++ << std::endl;

            // ---------------------------------------------------
            // Preprocessing and kernel reset (TO BE ADDED LATER)
            // ---------------------------------------------------

            // Averaging of material parameters
            medium.average_parameters(grid.nzt - 1, grid.nxt - 1);

            // ---------------------------------------------------


            // ----------------------------------------------
            // Forward and adjoint simulation for each shot
            // ----------------------------------------------
            cudaSetDevice(0);
           
                for (unsigned int ishot = 0; ishot < nshots; ishot++) {
                    std::cout << std::endl << "Forward time integration [SHOT " << ishot << "]" << std::endl;
                    start = clock();
                    // Arguments reset for each shots
                    //wave.reset_kernel();

                    // ----------------------------------------------------------
                    // FORWARD SIMULATION
                    // -----------------------------------------------------------
                    // calling forward simulation for shot i
                    forward_kernel_PSV(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
                        grid.dt, grid.dx, grid.dz, grid.snap_interval, grid.fsurf,
                        hc, fdorder, wave.vx, wave.vz, wave.sxx, wave.szx, wave.szz,
                        medium.lam, medium.mu, medium.mu_zx, medium.rho_zp, medium.rho_xp,
                        pml.npml, pml.a, pml.b, pml.K, pml.a_half, pml.b_half, pml.K_half,
                        pml.mem_vx_x, pml.mem_vx_z, pml.mem_vz_x, pml.mem_vz_z,
                        pml.mem_sxx_x, pml.mem_szx_x, pml.mem_szz_z, pml.mem_szx_z,
                        source.nseis, source.x_seis, source.z_seis,
                        source.seis_comp, source.signal, source.shot_to_fire,
                        receiver.nseis, receiver.x_seis, receiver.z_seis,
                        receiver.seis_comp, receiver.signal, receiver.shot_to_fire,
                        fwinv, fwi.dt, fwi.dx, fwi.dz, fwi.x1, fwi.x2, fwi.z1, fwi.z2,
                        fwi.vx, fwi.vz, fwi.sxx, fwi.szx, fwi.szz);


                    // -------------------------------------------------------
                    // Residuals and argument reset for adjoint simulation
                    // --------------------------------------------------------
                    // Calculation of residials to be added later
                    adjsrc = receiver; // Takes the details of the receivers

                    // Calculate residuals from receiver signal


                    // Resetting of kernels from forward to adjoint simulation to be added later

                    // ----------------------------------------------------------
                    // ADJOINT SIMULATION
                    // ----------------------------------------------------------
                    std::cout << std::endl << "Adjoint time integration [SHOT " << ishot << "]" << std::endl;

                    // Adjoing simulation for shot i
                    adjoint_kernel_PSV(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
                        grid.dt, grid.dx, grid.dz, grid.snap_interval, grid.fsurf,
                        hc, fdorder, wave.vx, wave.vz, wave.sxx, wave.szx, wave.szz,
                        medium.lam, medium.mu, medium.mu_zx, medium.rho_zp, medium.rho_xp,
                        pml.npml, pml.a, pml.b, pml.K, pml.a_half, pml.b_half, pml.K_half,
                        pml.mem_vx_x, pml.mem_vx_z, pml.mem_vz_x, pml.mem_vz_z,
                        pml.mem_sxx_x, pml.mem_szx_x, pml.mem_szz_z, pml.mem_szx_z,
                        adjsrc.nseis, adjsrc.x_seis, adjsrc.z_seis,
                        adjsrc.seis_comp, adjsrc.signal, adjsrc.shot_to_fire,
                        fwinv, fwi.dt, fwi.dx, fwi.dz, fwi.x1, fwi.x2, fwi.z1, fwi.z2,
                        fwi.vx, fwi.vz, fwi.sxx, fwi.szx, fwi.szz,
                        fwi.grad_lam, fwi.grad_mu, fwi.grad_rho);
                    end = clock();

                    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

                   // printf("\nExecution time from GPU each ITERATION = %f\n", cpu_time_used);
                    // ---------------------------------------------------------
                    // OPTIMIZATION
                    // --------------------------------------------------------
                    // Optimization part
                    // Currently just simple material update only 


                    std::cout << "SHOT " << ishot << " COMPLETED." << std::endl << std::endl;
                }
            }

        }
       
  

}