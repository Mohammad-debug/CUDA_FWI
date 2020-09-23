
/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef FD_CPML_H				
#define FD_CPML_H	

#include "globvar.cuh"

class FDCpml {
public:
	static int npml; // number of PML grids in each side
	static real_sim dh; // Spacing of the grid in PML layers
	static real_sim fpml; // Frequency of PML layer (Hz)
	static real_sim npower, k_max_PML;
	static real_sim rcoef; // Reflection coefficient
	static real_sim damp_v_PML; // Damping value

	static int tpml; // Total array length of PML
	static real_sim* a, * b, * K;// CPML parameters
	static real_sim* a_half, * b_half, * K_half; // CPML interpolated parameters	

	real_sim** mem_vx_x, ** mem_vx_z, ** mem_vz_x, ** mem_vz_z; // PML velocity derivative memory
	real_sim** mem_sxx_x, ** mem_szx_x, ** mem_szz_z, ** mem_szx_z; // PML stress derivative memory


	//explicit FDCpml(const char *configuration_file_relative_path); // constructor
	//FDCpml();
	//~FDCpml(); // Destructor

	void parse_configuration(const char* configuration_file_relative_path); // parse input

	void cpml_pro(real_sim dt);

	void allocate_pml();

	void deallocate_pml();

	void allocate_memory_arrays(int dimz, int dimx);

	void reset_memory_arrays(int dimz, int dimx);

	void deallocate_memory_arrays(int dimz);

};

inline void memory_CPML(int ix, int iz, int fpad, int ppad, bool fsurf,
	real_sim* a, real_sim* b, real_sim* K,
	real_sim* a_half, real_sim* b_half, real_sim* K_half,
	real_sim** mem_xx, real_sim** mem_zx, real_sim** mem_xz, real_sim** mem_zz,
	real_sim** value_xx, real_sim** value_zx, real_sim** value_xz, real_sim** value_zz);

void allocate_pml_memory_arrays(int npml, int dimz, int dimx,
	real_sim** mem_vx_x, real_sim** mem_vz_x,
	real_sim** mem_vx_z, real_sim** mem_vz_z,
	real_sim** mem_sxx_x, real_sim** mem_szx_x,
	real_sim** mem_szx_z, real_sim** mem_szz_z);

void deallocate_pml_memory_arrays(int npml, int dimz,
	real_sim** mem_vx_x, real_sim** mem_vz_x,
	real_sim** mem_vx_z, real_sim** mem_vz_z,
	real_sim** mem_sxx_x, real_sim** mem_szx_x,
	real_sim** mem_szx_z, real_sim** mem_szz_z);


#endif