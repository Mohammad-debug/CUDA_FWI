/*
* Created by: Min Basnet
* 2020.April.12
* Kathmandu, Nepal
*/
#ifndef UTIL_H					
#define UTIL_H

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdexcept>
#include "globvar.cuh"
#include "INIReader.cuh"


// Allocation and deallocation
// Due to template <class T>  "util.cpp" is included in the other codes
template <class T>
void allocate_array_1d(T*& mArray, int dim1);

template <class T>
void deallocate_array_1d(T*& mArray);

template <class T>
void allocate_array_2d(T**& mArray, const int dim1, const int dim2);

template <class T>
void deallocate_array_2d(T**& mArray, const int dim1);

template <class T>
void allocate_array_3d(T***& mArray, const int dim1, const int dim2, const int dim3);

template <class T>
void deallocate_array_3d(T***& mArray, const int dim1, const int dim2);

template <class T> // Referenced from external source forward_virieux from github
void parse_string_to_vector(std::basic_string<char> string_to_parse,
    std::vector<T>* destination_vector);

#endif