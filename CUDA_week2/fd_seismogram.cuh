//fd_seismogram.cuh

/*
* Created by: Min Basnet
* 2020.April.25
* Kathmandu, Nepal
*/

#ifndef FD_SEISMOGRAM_H				
#define FD_SEISMOGRAM_H	

#include "globvar.cuh"
#include "util.cuh"
#include "fd_grid.cuh"
#include "INIReader.cuh"

#include <iostream>

class FDSeismogram {
public:
	unsigned int nseis; // number of seismograms

	int read_from_data_files; // 0 = input from configuration file, 1 = from data files

	const char* seis_file; // seismogram data file name
	const char* signal_file; // signal file name

	std::vector<int> x_seis; // x_ordinate of the seismogram
	std::vector<int> y_seis; // y_ordinate of the seismogram
	std::vector<int> z_seis; // z_ordinate of the seismogram
	std::vector<int> v_or_s; // velocity or stress type (0 = velocity, 1 = stress)
	std::vector<int> seis_comp; // 0 = vel_amp, 1 = vel_x, 2 = vel_y, 3 = vel_z, 4 = explosive and so on
	std::vector<int> shot_to_fire; // Which signal to fire at which shot

	// Parameters only for sources
	std::vector<real_sim> angle_v; // verticle angle to z direction
	std::vector<real_sim> angle_h; // Horizontal angle to x direction

	// Signal parameters
	std::vector<real_sim> amp; // Amplitude
	std::vector<real_sim> fc; // Peak frequency
	std::vector<real_sim> ts; // Time shift
	std::vector<int> shape; // 0 = Signal file; 1 = Ricker

	real_sim** signal; // the signal arrays for all seismograms 
	real_sim** signal_meas; // Field measurements in the receivers

	void parse_configuration(const char* configuration_file_relative_path, const char sr, FDGrid grid);

	void grid_adjustment(FDGrid grid);

};

void create_wavelet(real_sim* signal, int nt, real_sim dt, real_sim amp, real_sim fc, real_sim ts, int shape);

#endif