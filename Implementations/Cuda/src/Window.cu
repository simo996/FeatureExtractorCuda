/*
 * Window.cpp
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#include "Window.h"

__host__ __device__ Window::Window(const short int dimension, const short int distance,
			   short int dirNumber, const bool symmetric){
	this->side = dimension;
	this->distance = distance;
	this->symmetric = symmetric;
	this->numberOfDirections = dirNumber;
}

__host__ __device__ void Window::setDirectionShifts(const int shiftRows, const int shiftColumns){
	this->shiftRows = shiftRows;
	this->shiftColumns = shiftColumns;
}

__host__ __device__ void Window::setSpacialOffsets(const int rowOffset, const int columnOffset){
	this->imageRowsOffset = rowOffset;
	this->imageColumnsOffset = columnOffset;
}
