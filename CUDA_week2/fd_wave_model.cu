
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
//*************
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
//***************************
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
//PARAMS SET

    bool gpu_comp = false;

    maxIter = 30;//iteration set


    if (gpu_comp == true) {
        // Bifurcation towards GPU computation
        // Do not pass any classes, just use the class variables
        // for example if it is pml.npml then use variable name as pml_npml

        // Allocation for parallel processing (For GPU model directly in GPU)
        medium.allocate_medium_av(grid.nzt - 1, grid.nxt - 1); // Harmonic and arithmatic average of medium parameters

        wave.allocate_wave(grid.nzt, grid.nxt); // Allocation of wave dynamic parameters array

        // Allocation of PML memory arrays
        if (pml.npml > 0) {
            pml.allocate_memory_arrays(grid.nzt, grid.nxt); // Allocates the PML memory for velocity and derivatives
        }

        // Allocation of FWI related parameters
      //  fwi.allocate_fwi(nft, nfz, nfx);


        // ------------------------------------------
        // Starting of full waveform inversion loop
        // ------------------------------------------

        int iter = 0;
        real_sim diff_L2 = 1.0;

        //cudaSetDevice(0);
        start1 = clock();

        
       //*************************************************************



        //*****           **********                      *************

        real_sim* d_a; real_sim* d_b; real_sim* d_K; real_sim* d_a_half; real_sim* d_b_half; real_sim* d_K_half;

        int size_1d = 2 * (pml.npml + 1);

        gpuErrchk(cudaMalloc((void**)&d_a, size_1d * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_b, size_1d * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_K, size_1d * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_a_half, size_1d * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_b_half, size_1d * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_K_half, size_1d * sizeof(real_sim)));

        gpuErrchk(cudaMemcpy(d_a, pml.a, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, pml.b, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_K, pml.K, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_a_half, pml.a_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b_half, pml.b_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_K_half, pml.K_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));

        real_sim* d_vx; real_sim* d_vz; real_sim* d_sxx;
        real_sim* d_szx; real_sim* d_szz;

        int size = grid.nzt * grid.nxt;

        gpuErrchk(cudaMalloc((void**)&d_vx, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_vz, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_sxx, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_szx, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_szz, size * sizeof(real_sim)));

       
        real_sim* d_mem_vx_x; real_sim* d_mem_vx_z; real_sim* d_mem_vz_x; real_sim* d_mem_vz_z;
        real_sim* d_mem_sxx_x; real_sim* d_mem_szx_x; real_sim* d_mem_szz_z; real_sim* d_mem_szx_z;

        size = grid.nzt * 2 * (pml.npml + 1);


        gpuErrchk(cudaMalloc((void**)&d_mem_vx_x, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_mem_vx_z, size * sizeof(real_sim)));
        size = grid.nxt * 2 * (pml.npml + 1);
        gpuErrchk(cudaMalloc((void**)&d_mem_vz_x, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_mem_vz_z, size * sizeof(real_sim)));
        size = grid.nzt * 2 * (pml.npml + 1);
        gpuErrchk(cudaMalloc((void**)&d_mem_sxx_x, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_mem_szx_x, size * sizeof(real_sim)));
        size = grid.nxt * 2 * (pml.npml + 1);
        gpuErrchk(cudaMalloc((void**)&d_mem_szz_z, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_mem_szx_z, size * sizeof(real_sim)));

        gpuErrchk(cudaMemcpy(d_mem_vx_x, pml.mem_vx_x[0], grid.nzt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mem_vx_z, pml.mem_vx_z[0], grid.nzt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_mem_vz_x, pml.mem_vz_x[0], grid.nxt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mem_vz_z, pml.mem_vz_z[0], grid.nxt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));


        gpuErrchk(cudaMemcpy(d_mem_sxx_x, pml.mem_sxx_x[0], grid.nzt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mem_szx_x, pml.mem_szx_x[0], grid.nzt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_mem_szz_z, pml.mem_szz_z[0], grid.nxt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mem_szx_z, pml.mem_szx_z[0], grid.nxt * 2 * (pml.npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));




        //**********************


        real_sim* d_fwi_vx;
        real_sim* d_fwi_vz;
        real_sim* d_fwi_sxx;
        real_sim* d_fwi_szx;
        real_sim* d_fwi_szz;

        const int nft = 1 + (grid.nt - 1) / fwi.dt;
        const int nfz = 1 + (fwi.z2 - fwi.z1) / fwi.dz;
        const int nfx = 1 + (fwi.x2 - fwi.x1) / fwi.dx;

        gpuErrchk(cudaMalloc((void**)&d_fwi_vx, (grid.nt * grid.nzt * grid.nxt) * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_fwi_vz, (grid.nt* grid.nzt* grid.nxt) * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_fwi_sxx, (grid.nt* grid.nzt* grid.nxt) * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_fwi_szx, (grid.nt* grid.nzt* grid.nxt) * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_fwi_szz, (grid.nt* grid.nzt* grid.nxt) * sizeof(real_sim)));

        
        real_sim* d_grad_lam;
        real_sim* d_grad_mu;
        real_sim* d_grad_rho;

        int size_grad = nfz * nfx;
        gpuErrchk(cudaMalloc((void**)&d_grad_lam, size_grad * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_grad_mu, size_grad * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_grad_rho, size_grad * sizeof(real_sim)));

    

        real_sim* d_lam;  real_sim* d_mu;
        real_sim* d_mu_zx; real_sim* d_rho_zp; real_sim* d_rho_xp;


        size = grid.nzt * grid.nxt;


        gpuErrchk(cudaMalloc((void**)&d_lam, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_mu, size * sizeof(real_sim)));
        size = (grid.nzt - 1) * (grid.nxt - 1);
        gpuErrchk(cudaMalloc((void**)&d_mu_zx, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_rho_zp, size * sizeof(real_sim)));
        gpuErrchk(cudaMalloc((void**)&d_rho_xp, size * sizeof(real_sim)));


        //******************************************************************
        cudaSetDevice(0);
        while (diff_L2 >= maxError) {

            if (iter >= maxIter) {
                end = clock();

                cpu_time_used = ((double)(end - start1)) / CLOCKS_PER_SEC;

                printf("\n TOTAL PROGRAM(GPU) Execution time before exit a  = %f TOTAL ITERATIONS=%d\n", cpu_time_used, maxIter);

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
            size = grid.nzt * grid.nxt;

            gpuErrchk(cudaMemcpy(d_lam, medium.lam[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_mu, medium.mu[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_mu_zx, medium.mu_zx[0], (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_rho_zp, medium.rho_zp[0], (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_rho_xp, medium.rho_xp[0], (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));


            // ----------------------------------------------
            // Forward and adjoint simulation for each shot
            // ----------------------------------------------



            for (unsigned int ishot = 0; ishot < nshots; ishot++) {
                std::cout << std::endl << "Forward time integration [SHOT " << ishot << "]" << std::endl;
                start = clock();
                // Arguments reset for each shots
                //wave.reset_kernel();


                // ----------------------------------------------------------
                // FORWARD SIMULATION
                // -----------------------------------------------------------
                // calling forward simulation for shot i
                forward_kernel_PSV_GPU(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
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
                    fwi.vx, fwi.vz, fwi.sxx, fwi.szx, fwi.szz,
                    //
                    d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
                    //
                    d_vx, d_vz, d_sxx, d_szx, d_szz,
                    // 
                    d_fwi_vx, d_fwi_vz, d_fwi_sxx, d_fwi_szx, d_fwi_szz,
                    //
                    d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
                    d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z,
                    //
                    d_lam, d_mu, d_mu_zx, d_rho_zp, d_rho_xp
                );


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
                adjoint_kernel_PSV_GPU(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
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
                    fwi.grad_lam, fwi.grad_mu, fwi.grad_rho,
                    //
                    d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
                    //
                    d_vx, d_vz, d_sxx, d_szx, d_szz,
                    // 
                    d_fwi_vx, d_fwi_vz, d_fwi_sxx, d_fwi_szx, d_fwi_szz,
                    //
                    d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
                    d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z,
                    //

                    //
                    d_grad_lam, d_grad_mu, d_grad_rho,
                    //
                    d_lam, d_mu, d_mu_zx, d_rho_zp, d_rho_xp
                );

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
            gpuErrchk(cudaMemcpy(medium.lam[0], d_lam, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(medium.mu[0], d_mu, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(medium.mu_zx[0], d_mu_zx, (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(medium.rho_zp[0], d_rho_zp, (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(medium.rho_xp[0], d_rho_xp, (grid.nzt - 1) * (grid.nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));

        }


    }
    //**********************************CPU****************************************
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
      //std::cout<< "Error mark 2" <<std::endl;

      // Updating the old material with existing material (with zero step length)
      // only density update (to keep record of the material)
      //medium.fwi_grad_update(medium.rho, medium.rho_old, medium.rho_old, 0.0, grid.nzt, grid.nxt, false);

      for (int iz = 0; iz < grid.nzt; iz++) {
          for (int ix = 0; ix < grid.nxt; ix++) {
              medium.rho_old[iz][ix] = medium.rho[iz][ix];
              medium.lam_old[iz][ix] = medium.lam[iz][ix];
              medium.mu_old[iz][ix] = medium.mu[iz][ix];
          }
      }
      //std::cout<< "Error mark 0" <<std::endl;

        // ------------------------------------------
        // Starting of full waveform inversion loop
        // ------------------------------------------

      iter = 0;
      L2_norm = 1000.0;
      real_sim L2_test[4];

      // computational time
      start_time_main = clock();
      iteration_time = clock();
      std::cout << "Clock started." << std::endl;

      // The condition needs to be checked for different condition
      while (L2_norm >= maxError) {

          if (iter >= maxIter) {
              std::cout << "Error: convergence not reached within maximum iteration" << std::endl;
              exit(1);
          }

          std::cout << std::endl << std::endl;
          std::cout << std::endl << "Full Waveform Inversion simulation. Iteration step " << iter++ << std::endl;
          std::cout << "L2 norm = " << L2_norm << std::endl;

          // ---------------------------------------------------
          // Preprocessing and kernel reset (TO BE ADDED LATER)
          // ---------------------------------------------------

          // Averaging of material parameters
          medium.average_parameters(grid.nzt - 1, grid.nxt - 1);
          medium.medium_average(C_lam, C_mu, C_rho, grid.nzt, grid.nxt);
          std::cout << "Material Average: " << C_rho << ", " << C_lam << ", " << C_mu << std::endl;

          // ---------------------------------------------------


          // ----------------------------------------------
          // Forward and adjoint simulation for each shot
          // ----------------------------------------------

          for (unsigned int ishot = 0; ishot < nshots; ishot++) {
              std::cout << std::endl << "Forward time integration [SHOT " << ishot << "]" << std::endl;

              // Arguments reset for each shots
              wave.reset_kernel(grid.nzt, grid.nxt);
              pml.reset_memory_arrays(grid.nzt, grid.nxt);

              //reset fwi kernels
              fwi.reset_fwi_kernel(nft, nfz, nfx, true);

              // ----------------------------------------------------------
              // FORWARD SIMULATION
              // -----------------------------------------------------------
              // calling forward simulation for shot i
              forward_kernel_PSV(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
                  grid.dt, grid.dx, grid.dz, grid.snap_interval, grid.fsurf,
                  hc, fdorder, wave.vx, wave.vz, wave.sxx, wave.szx, wave.szz, wave.We,
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
              // Calculation of residials to for adjoint modelling 


              std::cout << "Calculating residual....";
              L2_norm = calculate_l2_adjoint_sources(receiver.nseis, grid.nt, grid.dt,
                  receiver.signal, receiver.signal_meas);

              L2_test[0] = L2_norm; // Save for step length approximateion
              L2_test[3] = L2_norm; // Save for step length approximation

              // Storing the Energy Weights from Forward wavefield (Source)
              std::cout << "NF grid: nfz= " << nfz << ", nfx= " << nfx << std::endl;
              for (int iz = 0; iz < grid.nzt; iz++) {
                  for (int ix = 0; ix < grid.nxt; ix++) {

                      We[iz][ix] = wave.We[iz][ix];

                      //std::cout << "Source Energy:" << iz <<", " << ix << ", " <<We[iz][ix] << std::endl;

                  }
              }


              std::cout << " L2 norm = " << L2_test[0] << ". Max error = " << maxError << std::endl;

              // receiver now acts as adjoint source

              // Resetting of kernels from forward to adjoint simulation 
              wave.reset_kernel(grid.nzt, grid.nxt);
              pml.reset_memory_arrays(grid.nzt, grid.nxt);

              // ----------------------------------------------------------
              // ADJOINT SIMULATION
              // ----------------------------------------------------------
              std::cout << std::endl << "Adjoint time integration [SHOT " << ishot << "]" << std::endl;

              // Adjoing simulation for shot i
              adjoint_kernel_PSV(ishot, grid.nt, grid.nzt, grid.nxt, grid.fpad, grid.ppad,
                  grid.dt, grid.dx, grid.dz, grid.snap_interval, grid.fsurf,
                  hc, fdorder, wave.vx, wave.vz, wave.sxx, wave.szx, wave.szz, wave.We,
                  medium.lam, medium.mu, medium.mu_zx, medium.rho_zp, medium.rho_xp,
                  pml.npml, pml.a, pml.b, pml.K, pml.a_half, pml.b_half, pml.K_half,
                  pml.mem_vx_x, pml.mem_vx_z, pml.mem_vz_x, pml.mem_vz_z,
                  pml.mem_sxx_x, pml.mem_szx_x, pml.mem_szz_z, pml.mem_szx_z,
                  receiver.nseis, receiver.x_seis, receiver.z_seis,
                  receiver.seis_comp, receiver.signal, receiver.shot_to_fire,
                  fwinv, fwi.dt, fwi.dx, fwi.dz, fwi.x1, fwi.x2, fwi.z1, fwi.z2,
                  fwi.vx, fwi.vz, fwi.sxx, fwi.szx, fwi.szz,
                  fwi.grad_lam, fwi.grad_mu, fwi.grad_rho);

              // ---------------------------------------------------------
              // OPTIMIZATION
              // --------------------------------------------------------
              // Optimization part

              std::cout << "Scaling gradients" << std::endl;
              // Preconditioning
              // Scaling gradient using energy weights

              scale_gradients_with_energy_weights(We, wave.We, C_rho, C_lam, C_mu, nfz, nfx,
                  fwi.z1, fwi.x1, fwi.dz, fwi.dx);

              //-----------------------

       // ---------------------
      // Printing AASCI data to plot updated density
/*
        if (!(iter%1)){

          outFile.open("./io/density_snap/grad_lam_snap"+std::to_string(iter)+".csv");
          for(int j=0; j< nfz;j++){
            for(int i=0; i< nfx;i++){
              outFile<< fwi.grad_lam[j][i] << ", " ;
            }
            outFile<<std::endl;
          }
          outFile.close();

        }
        //-----------------------
*/
              std::cout << "Step length Estimation" << std::endl;
              // Calculate step length by parabolic line search algorithm
              step_length = fwi_step_length_estimation(step_length, L2_test);

              std::cout << "Material Update" << std::endl;

              // currently directly update material
              // only density update
              medium.fwi_grad_update(medium.rho_old, medium.rho, fwi.grad_rho,
                  0.5 * step_length, nfz, nfx, fwi.z1, fwi.x1, fwi.dz, fwi.dx, true, 1);


              medium.fwi_grad_update(medium.mu_old, medium.mu, fwi.grad_mu,
                  step_length, nfz, nfx, fwi.z1, fwi.x1, fwi.dz, fwi.dx, true, 3);
              medium.fwi_grad_update(medium.lam_old, medium.lam, fwi.grad_lam,
                  step_length, nfz, nfx, fwi.z1, fwi.x1, fwi.dz, fwi.dx, true, 2);


              std::cout << "Iteration " << iter << " complete" << std::endl;

              iteration_time = clock() - iteration_time;
              time_elasp = iteration_time / CLOCKS_PER_SEC;
              std::cout << "Time for iteration = " << time_elasp << " [sec]." << std::endl;

              // ---------------------
              // Printing AASCI data to plot updated density

              if (!(iter % 1)) {

                  outFile.open("./io/density_snap/density_snap" + std::to_string(iter) + ".csv");
                  for (int j = 0; j < grid.nzt; j++) {
                      for (int i = 0; i < grid.nxt; i++) {
                          outFile << medium.rho[j][i] << ", ";
                      }
                      outFile << std::endl;
                  }
                  outFile.close();



                  outFile.open("./io/density_snap/lam_snap" + std::to_string(iter) + ".csv");
                  for (int j = 0; j < grid.nzt; j++) {
                      for (int i = 0; i < grid.nxt; i++) {
                          outFile << medium.lam[j][i] << ", ";
                      }
                      outFile << std::endl;
                  }
                  outFile.close();

                  outFile.open("./io/density_snap/mu_snap" + std::to_string(iter) + ".csv");
                  for (int j = 0; j < grid.nzt; j++) {
                      for (int i = 0; i < grid.nxt; i++) {
                          outFile << medium.mu[j][i] << ", ";
                      }
                      outFile << std::endl;
                  }
                  outFile.close();



              }
              //-----------------------

          }

      }

  }

  deallocate_array_2d(We, grid.nzt); // Energy weight

  end_time_main = clock();
  time_elasp = (end_time_main - start_time_main) / CLOCKS_PER_SEC;
  std::cout << "Time required = " << time_elasp << " [sec]." << std::endl;


       
  

}