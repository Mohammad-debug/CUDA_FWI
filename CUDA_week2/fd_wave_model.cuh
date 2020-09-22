

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef FD_WAVE_MODEL_H				
#define FD_WAVE_MODEL_H	

#include "globvar.cuh"
#include "fd_grid.cuh"
#include "fd_medium.cuh"
#include "fd_cpml.cuh"
#include "fd_wave.cuh"
#include "fd_seismogram.cuh"
#include "fd_fwi.cuh"

class FDWaveModel {
public:
	class FDGrid grid; // Finite difference grid
	class FDMedium medium; // The medium or the material
	class FDCpml pml; // CPML boundary condition
	class FDWave wave;  // The wave velocity and stresses
	class FDSeismogram source, receiver, adjsrc; // The sources, receivers and adjoint sources
	class FDFWI fwi; // parameters fir full waveform inversion
	bool fwinv; // Whether to do full waveform inversion or not



	unsigned int fdorder; // order of spatial finite difference scheme
	unsigned int nshots; // number of shots
	int maxIter; // Maximum number of iteration to escape, if no convergence is achieved
	real_sim maxError; // Maximum error in the Norm after each iteration

	explicit FDWaveModel(const char* configuration_file_relative_path); // constructor
	~FDWaveModel(); // Destructor

	void parse_configuration(const char* configuration_file_relative_path);

	void forward_simulate();

	void fwi_simulate();


	real_sim calculate_l2_adjoint_sources(unsigned int nr, int nt, real_sim dt,
		real_sim** signal, real_sim** signal_meas);

	real_sim fwi_step_length_estimation(real_sim est_step_length, real_sim* L2_test);

	void scale_gradients_with_energy_weights(real_sim** Ws, real_sim** Wr, real_sim C_rho, real_sim C_lam, real_sim C_mu,
		int nz, int nx, int z1, int x1, int fdz, int fdx);

};

#endif