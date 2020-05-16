//fd_medium.h

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef FD_WAVE_H				
#define FD_WAVE_H	

#include "globvar.cuh"

class FDWave {
public:
	// 2D wave parameters (PSV and SH wave)
	real_sim** vx, ** vz; // Particle velocity
	real_sim** sxx, ** szz; // Stress tensors (Translational)
	real_sim** szx; // Stress tensors (Rotational)

	void allocate_wave(int dimz, int dimx); // Allocate velocity and stress tensors

	void deallocate_wave(int dimz); // Deallocates velocity and stress tensors

};

#endif
