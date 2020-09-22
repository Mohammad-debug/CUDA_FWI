//fd_fwi_step_length_estimation.cpp

/*
* Created by: Min Basnet
* 2020.May.23
* Kathmandu, Nepal
*/

#include "fd_wave_model.cuh"
#include "fd_kernel.cuh"
#include "ext.h"
#include "globvar.cuh"
std::ofstream outReceiverFile1, outFile1; // output file to store receiver data

real_sim FDWaveModel::fwi_step_length_estimation(real_sim est_step_length, real_sim* L2_test) {

    // Estimation of step length for gradient update
    real_sim step_length[4];
    real_sim stl; // The factor for density update
    real_sim opt_step_length; // Optimum step length calculated

    real_sim step_factor_rho = 0.5; // Scale factor for updating density
    real_sim est_scale_factor = 2.0; // Step factor in next approximation
    real_sim stepmax = 6; // The maximum number of steps to find optimum step length

    int checkstep;
    checkstep = 0;

    //real_sim L2_test[];

    const int nft = 1 + (grid.nt - 1) / fwi.dt;
    const int nfz = 1 + (fwi.z2 - fwi.z1) / fwi.dz;
    const int nfx = 1 + (fwi.x2 - fwi.x1) / fwi.dx;


    // controllers for steps in approximation
    unsigned int step1 = 0, step2 = 0;
    //unsigned int step3;
    unsigned int itests = 1, iteste = 1; // initialize start and end of test
    unsigned int countstep = 0;

    // multiple material test checks to calculate L2 norms for these changes
    // three tests performed currently

    while ((step2 != 1) || (step1 != 1)) {
        for (unsigned int itest = itests; itest <= iteste; itest++) {
            //
            checkstep++;
            // Material update test

            medium.fwi_grad_update(medium.lam_old, medium.lam, fwi.grad_lam,
                est_step_length, nfz, nfx,
                fwi.z1, fwi.x1, fwi.dz, fwi.dx, false, 2);
            medium.fwi_grad_update(medium.mu_old, medium.mu, fwi.grad_mu,
                est_step_length, nfz, nfx,
                fwi.z1, fwi.x1, fwi.dz, fwi.dx, false, 3);

            // currently only density change 
            stl = step_factor_rho * est_step_length;
            medium.fwi_grad_update(medium.rho_old, medium.rho, fwi.grad_rho,
                stl, nfz, nfx,
                fwi.z1, fwi.x1, fwi.dz, fwi.dx, false, 1);

            // Averaging of material parameter
            medium.average_parameters(grid.nzt - 1, grid.nxt - 1);


            // now run forward modelling with updated parameter
            // ----------------------------------------------
            // Forward and adjoint simulation for each shot
            // ----------------------------------------------

            for (unsigned int ishot = 0; ishot < nshots; ishot++) {
                std::cout << std::endl << "Forward time integration [SHOT " << ishot << "]" << std::endl;

                // Arguments reset for each shots
                wave.reset_kernel(grid.nzt, grid.nxt);
                pml.reset_memory_arrays(grid.nzt, grid.nxt);

                reset_array_2d(receiver.signal, receiver.nseis, grid.nt);
                fwi.reset_fwi_kernel(nft, nfz, nfx, false);



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

                outReceiverFile1.open("./io/receiver/rec_signal_" + std::to_string(iter) + "_" + std::to_string(checkstep) + ".csv");
                // Writing ASCCI output at the receivers
                for (unsigned int ir = 0; ir < receiver.nseis; ir++) {
                    //
                    for (int it = 0; it < grid.nt; it++) {
                        outReceiverFile1 << receiver.signal[ir][it] << ", ";
                    }
                    outReceiverFile1 << std::endl;
                }
                outReceiverFile1.close();


                outReceiverFile1.open("./io/receiver/src_signal_" + std::to_string(iter) + "_" + std::to_string(checkstep) + ".csv");
                // Writing ASCCI output at the receivers
                for (unsigned int is = 0; is < source.nseis; is++) {
                    //
                    for (int it = 0; it < grid.nt; it++) {
                        outReceiverFile1 << source.signal[is][it] << ", ";
                    }
                    outReceiverFile1 << std::endl;
                }
                outReceiverFile1.close();


                std::cout << "Calculating residual....";
                L2_test[itest] = calculate_l2_adjoint_sources(receiver.nseis, grid.nt, grid.dt,
                    receiver.signal, receiver.signal_meas);

                step_length[itest] = est_step_length;
                step_length[0] = 0.0;

            }
            std::cout << "Step Length: " << est_step_length << std::endl;
            std::cout << "first step = " << step1 << " , second step = " << step2
                << " itest = " << itest << ", L2 = " << L2_test[itest] << ", counter = " << countstep;// <<std::endl;

        }
        // multiple tests performed

        // Different conditions arise here, which need to be calculated

        // Condition 4: first step is done and fount a stepsize with L2_test[1] < L2_text[2];
        if ((step1 == 1) && (L2_test[1] < L2_test[2])) {
            step_length[2] = est_step_length;
            step2 = 1;
            std::cout << ", Condition 2.2" << std::endl;
        }

        // Condition 3: first step is done and 
        //could not find step size with L2_test[1] < L2t[2]
        else if ((step1 == 1) && (L2_test[1] >= L2_test[2])) {
            step_length[2] = est_step_length;
            est_step_length = est_step_length + (est_step_length / est_scale_factor);
            countstep++;
            std::cout << ", Condition 2.1" << std::endl;

        }

        // Condition 2: Found a step size, which reduces the misfit function
        else if ((step1 == 0) && (L2_test[0] > L2_test[1])) {
            step_length[1] = est_step_length;
            step1 = 1;
            itests = 2;
            iteste = 2;
            countstep = 0;

            // find a second steplength with a larger misfit function than L2_test[1]
            est_step_length = est_step_length + (est_step_length / est_scale_factor);
            std::cout << ", Condition 1.2" << std::endl;
        }

        // Condition 1: Did not find a step size which reduces the misfit function
        // The L2_norm for this round is greater than that of from previous step
        else if ((step1 == 0) && (L2_test[0] <= L2_test[1])) {

            est_step_length = est_step_length / est_scale_factor;
            countstep++;
            std::cout << ", Condition 1.1" << std::endl;

        }


        //step3 = 0;

        // Abortion criteria for if appropriate L2 norms are not calculated

        // Condition 1 
        if ((step1 == 0) && (countstep > stepmax)) {
            std::cout << "Step length estimation failed !" << std::endl;
            end_time_main = clock();
            time_elasp = (end_time_main - start_time_main) / CLOCKS_PER_SEC;
            std::cout << "Time required = " << time_elasp << " [sec]." << std::endl;
            //step3 = 1;
            exit(0);
        }

        if ((step1 == 1) && (countstep > stepmax)) {
            std::cout << "Could not find a proper 3rd step length which brackets the minimum" << std::endl;
            step1 = 1;
            step2 = 1; // calculate with less optimum model
        }

    }

    if (step1 == 1) {
        std::cout << "Calculating optimal step length for vp and vs" << std::endl;

        opt_step_length = calc_opt_step(L2_test, step_length);
    }

    return opt_step_length;
}



real_sim calc_opt_step(real_sim* L2, real_sim* sl) {
    // Calculates the optimum step length from
    // Array of L2 norms with respect to step_lengths
    // L2 = L2 norm, sl = steplength

    int n = 3; // size of the system
    real_sim A[3][3]; // Defining coefficient matrix
    real_sim b[3]; // Defining RHS vector
    real_sim x[3]; // The variable to be solved vector

    real_sim opteps; // optimum step

    /* calculate optimal step size by line search */
    /* ------------------------------------------ */
    /* fit parabola function to L2 norm */

    std::cout << "L2: " << L2[0] << ", " << L2[1] << ", " << L2[2] << ", " << std::endl;
    std::cout << "SL: " << sl[0] << ", " << sl[1] << ", " << sl[2] << ", " << std::endl;

    //define coefficient matrix A 
    for (int i = 0; i < n; i++) {
        A[i][2] = (sl[i] * sl[i]);
        A[i][1] = (sl[i]);
        A[i][0] = (1.0);
    }


    //define RHS vector b
    for (int i = 0; i < n; i++) {
        b[i] = L2[i];
    }

    /* solve matrix equation using LU decomposition */
    /*LU_decomp(A,x,b,n);*/
    solvelin(A, b, x, n, 1);

    // calculate optimal step length -> extremum of the parabola */
    opteps = -x[1] / (2.0 * x[2]);

    std::cout << "Optimum Step Length: " << opteps << std::endl << std::endl;

    return opteps;



}