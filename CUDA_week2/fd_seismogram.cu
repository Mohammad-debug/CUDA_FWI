//fd_seismogram.cpp("source", "peak_frequency", 20.0)

/*
* Created by: Min Basnet
* 2020.April.25
* Kathmandu, Nepal
*/
#include <iostream>

#include "fd_seismogram.cuh"
#include "INIReader.cuh"
//#include "../ext/inih/INIReader.h"
#include "util.cu"
//#include "fd_grid.cuh"



void FDSeismogram::parse_configuration(const char* configuration_file_relative_path, const char sr, FDGrid grid) {
    //
    INIReader reader(configuration_file_relative_path);
    std::string seis_configuration_section;
    int measurement_records;

    if (sr == 's') { // The seismogram is parsed as a source
        seis_configuration_section = "source";
        std::cout << "parsing FD SOURCE from file '"
            << configuration_file_relative_path << "'." << std::endl;

        std::cout << std::endl << "Source:" << std::endl;
        if (reader.ParseError() < 0) {
            std::cout << "Can't load .ini file\n";
            exit(1);
        }
    }

    else {
        seis_configuration_section = "receiver";
        std::cout << "parsing FD RECEIVER from file '"
            << configuration_file_relative_path << "'." << std::endl;

        std::cout << std::endl << "Receivers:" << std::endl;
        if (reader.ParseError() < 0) {
            std::cout << "Can't load .ini file\n";
            exit(1);

        }
    }


    // Parameters for both sources and receivers
    // --------------------------------------------
    nseis = reader.GetInteger(seis_configuration_section, "nseis", 1);

    read_from_data_files = reader.GetInteger(seis_configuration_section, "read_from_data_files", 0);

    if (read_from_data_files == 1) {// read from signal files
        seis_file = reader.Get(seis_configuration_section, "seis_file", "inbuilt").c_str();
        signal_file = reader.Get(seis_configuration_section, "signal_file", "inbuilt").c_str();

        // Printing filenames in the string
        std::cout << "Reading the seismogram and corresponding signal from the files:" << std::endl;
        std::cout << "seis_file = '" << seis_file << "', signal_file= '" << signal_file << "'" << std::endl;
        std::cout << std::endl;

    }

    else if (read_from_data_files == 0) {// read seismograms from configuration files

      // reserving the space for the vectors
        x_seis.reserve(nseis);
        //y_seis.reserve(nseis);
        z_seis.reserve(nseis);
        v_or_s.reserve(nseis);
        seis_comp.reserve(nseis);
        // 
        // Now parsing from configuration file
        parse_string_to_vector(reader.Get(seis_configuration_section, "x_seis", "{0};"), &x_seis);
        //parse_string_to_vector( reader.Get(seis_configuration_section, "y_seis", "{0};"), &y_seis);
        parse_string_to_vector(reader.Get(seis_configuration_section, "z_seis", "{0};"), &z_seis);

        parse_string_to_vector(reader.Get(seis_configuration_section, "velocity_or_stress", "{0};"), &v_or_s);

        parse_string_to_vector(reader.Get(seis_configuration_section, "component_type", "{0};"), &seis_comp);

        parse_string_to_vector(reader.Get(seis_configuration_section, "shot_to_fire", "{0};"), &shot_to_fire);

        // Allocate receiver signal files
        allocate_array_2d(signal, nseis, grid.nt);


        // Input for inbuilt signal types
        if (sr == 's') {

            parse_string_to_vector(reader.Get(seis_configuration_section, "verticle_angle_deg", "{0.0};"), &angle_v);
            //parse_string_to_vector( reader.Get(seis_configuration_section, "horizontal_angle_deg", "{0.0};"), &angle_h);

            parse_string_to_vector(reader.Get(seis_configuration_section, "shape", "{0};"), &shape);
            parse_string_to_vector(reader.Get(seis_configuration_section, "amplitude", "{0.0};"), &amp);
            parse_string_to_vector(reader.Get(seis_configuration_section, "peak_frequency", "{0.0};"), &fc);
            parse_string_to_vector(reader.Get(seis_configuration_section, "time_shift", "{0.0};"), &ts);


            // Allocation of source signal file
            allocate_array_2d(signal, nseis, grid.nt);


            // Creating signal file
            for (unsigned int ii = 0; ii < nseis; ii++) {

                create_wavelet(signal[ii], grid.nt, grid.dt, amp[ii], fc[ii], ts[ii], shape[ii]);

                // Output
                std::cout << "source " << ii << ": " << std::endl;
                std::cout << "grid_x = " << x_seis[ii] << ", grid_z = " << z_seis[ii] << std::endl;
                std::cout << "velocity or stress = " << v_or_s[ii] << ", component = " << seis_comp[ii] << std::endl;
                std::cout << "signal shape = " << shape[ii] << std::endl;
                std::cout << "amplitude = " << amp[ii] << ", peak frequency = " << fc[ii]
                    << ", time shift = " << ts[ii] << std::endl;

                std::cout << std::endl;

            }

        }
        if (sr == 'r') {
            // if there are field measurement records
            measurement_records = reader.GetInteger(seis_configuration_section, "measurement_records", 0);

            if (measurement_records) {
                allocate_array_2d(signal_meas, nseis, grid.nt);
            }

            // Creating receiver files
            for (unsigned int ii = 0; ii < nseis; ii++) {

                // Output
                std::cout << "receiver " << ii << ": " << std::endl;
                std::cout << "grid_x = " << x_seis[ii] << ", grid_z = " << z_seis[ii] << std::endl;
                std::cout << "velocity or stress = " << v_or_s[ii] << ", component = " << seis_comp[ii] << std::endl;
                //std::cout << "signal shape = " << shape[ii]<<std::endl;
                //std::cout << "amplitude = " << amp[ii] <<", peak frequency = " << fc[ii] 
                //<< ", time shift = " << ts [ii] << std::endl;

                std::cout << std::endl;

            }
        }

    }

}

void FDSeismogram::grid_adjustment(FDGrid grid) {
    // Adjusts the grid indices of the seismogram location 

    for (unsigned int ii = 0; ii < nseis; ii++) {
        x_seis[ii] += grid.ppad;

        if (grid.fsurf) z_seis[ii] += grid.fpad;
        else z_seis[ii] += grid.ppad;

    }

}

void create_wavelet(real_sim* signal, int nt, real_sim dt, real_sim amp, real_sim fc, real_sim ts, int shape) {
    // Create signal
    // **signal: The array in which signal is to be written
    // isrc: The source number index to go to corresponding index of the signal array
    // nt: number of time steps, dt: time step size, ts: time shift
    // fc: peak frequency, amp: amplitude of the signal

    real_sim t, tau; // time 
    const real_sim fci = 1.0 / fc;

    switch (shape) {
    case(1): // Ricker wavelet
        for (int it = 0; it < nt; it++) {
            t = it * dt;
            tau = PI * (t - 1.5 * fci - ts) / (1.5 * fci);
            signal[it] = amp * (1.0 - 2.0 * tau * tau) * exp(-2.0 * tau * tau);
            //std::cout<<it<<", "<<t<<", "<<signal[it] << std::endl;
        }
    }

}