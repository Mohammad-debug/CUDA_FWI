/*
* Created by: Min Basnet
* 2020.April.12
* Kathmandu, Nepal
*/

#ifndef UTIL_CPP
#define UTIL_CPP
#include "util.cuh"

#include <iostream>


// Allocation and deallocation

// Allocation and deallocation

template <class T>
void allocate_array_1d(T*& mArray, const int dim1) {
    mArray = new T[dim1];
    for (int i = 0; i < dim1; ++i) mArray[i] = 0.0;
}

template <class T>
void allocate_array_2d(T**& mArray, const int dim1, const int dim2) {
    // Contiguous allocation of 2D arrays
    // Referenced from:
    // http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
    // and
    // https://dev.to/drakargx/c-contiguous-allocation-of-2-d-arrays-446m
    // with error correction i=0 to i=1

    mArray = new T * [dim1];
    mArray[0] = new T[dim1 * dim2];
    for (int i = 1; i < dim1; i++) mArray[i] = mArray[i - 1] + dim2;

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            mArray[i][j] = 0;
        }
    }
}

template <class T>
void allocate_array_3d(T***& mArray, const int dim1, const int dim2, const int dim3) {
    mArray = new T * *[dim1];
    mArray[0] = new T * [dim1 * dim2];

    mArray[0][0] = new T[dim1 * dim2 * dim3];

    int i, j, k;

    for (i = 0; i < dim1; i++) {

        if (i < dim1 - 1) {

            mArray[0][(i + 1) * dim2] = &(mArray[0][0][(i + 1) * dim3 * dim2]);

            mArray[i + 1] = &(mArray[0][(i + 1) * dim2]);

        }

        for (j = 0; j < dim2; j++) {
            if (j > 0) mArray[i][j] = mArray[i][j - 1] + dim3;
        }

    }
    /*for (int i = 0; i < dim1; ++i)
        allocate_array_2d(mArray[i], dim2, dim3);*/




}


template <class T>
void deallocate_array_1d(T*& mArray) {
    delete[] mArray;
    mArray = nullptr;
}


template <class T>
void deallocate_array_2d(T**& mArray, const int dim1) {

    // Deallocating the Array except for the first row
    for (int i = 1; i < dim1; i++) {
        mArray[i] = nullptr; // Contiguous memory allocation
    }

    // Deallocating the first row after it
    deallocate_array_1d(mArray[0]);

    delete[] mArray;
    mArray = nullptr;
}

template <class T>
void deallocate_array_3d(T***& mArray, const int dim1, const int dim2) {

    delete[] mArray[0][0];
    delete[] mArray[0];
    delete[] mArray;
    mArray = nullptr;
}


template <class T> // Referenced from external source forward_virieux from github
void parse_string_to_vector(std::basic_string<char> string_to_parse,
    std::vector<T>* destination_vector) {
    // Erase all spaces
    string_to_parse.erase(
        remove_if(string_to_parse.begin(), string_to_parse.end(), isspace),
        string_to_parse.end());
    // Find end of data and cut afterwards
    size_t pos = string_to_parse.find("}");
    string_to_parse.erase(pos, string_to_parse.length());
    // Cut leading curly brace
    string_to_parse.erase(0, 1);
    // Split up string
    std::string delimiter = ",";
    pos = 0;
    std::string token;
    while ((pos = string_to_parse.find(delimiter)) != std::string::npos) {
        token = string_to_parse.substr(0, pos);
        destination_vector->emplace_back(atof(token.c_str()));
        string_to_parse.erase(0, pos + delimiter.length());
    }
    token = string_to_parse.substr(0, pos);
    destination_vector->emplace_back(atof(token.c_str()));
}



#endif