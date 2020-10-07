# CUDA_FWI
CUDA applied to FWI kernels

Computational cost and storage requirement are the main obstacles that inhibit the research and practical application of full waveform inversion (FWI).
We have developed a fast parallel scheme to speed up FWI on graphics processing unit (GPU),which is a parallel computing device, via CUDA(an acronym for Compute Unified Device Architecture), developed by NVIDA and used as the programming environment. 
In this parallel scheme, to avoid frequent and low-bandwidth data transfer between host memory and device memory, almost the entire computing task, including propagator and backpropagator, are coded as a sequence of kernel functions that can be called from the compute host for each iterative inversion. 
The random boundaries conditions are used when propagating source wavefield to solve the storage requirement so that we do not have to save any additional wavefield data and the noise introduced into final inversion image is so weak that can be ignored due to iterations. 
To test our algorithm, we implement the FWI on Personal Computer (PC) with GTX2080Ti GPU. This numerical test indicates that the GPU-based FWI typically is 11 times faster than the CPU-based implementation.
<br />
 <br />
 Most of the work on parallelization is done on Forward and Adjoint kernel in CUDA_week2 folder. There are two sets of identical functions in these files, One does simple CPU computation and another GPU 
based parallelized computation. The user specifies which one of them will execute with the input. After computation ends result is stored in snap_data folder in io directory in CSV format and both and device and host memory is deallocated.
