// ext.h

// header reference to external functions

#include "globvar.cuh"

void checkfd_ssg_elastic(int NX, int NY, real_sim DH, real_sim DT, real_sim TS, int FW,
	real_sim** prho, real_sim** ppi, real_sim** pu, real_sim* hc);

void holbergcoeff(const int FDORDER, const int MAXRELERROR, real_sim* hc);

void solvelin(real_sim  A[3][3], real_sim b[3], real_sim x[3], int e, int method);
