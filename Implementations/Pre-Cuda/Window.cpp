//
// Created by simo on 26/07/18.
//

#include "Window.h"

Window::Window(const short int dimension, const short int distance, const bool symmetric){
	this->side = dimension;
	this->distance = distance;
	this->symmetric = symmetric;
}

void Window::setDirectionShifts(const int shiftRows, const int shiftColumns){
	this->shiftRows = shiftRows;
	this->shiftColumns = shiftColumns;
}

void Window::setSpacialOffsets(const int rowOffset, const int columnOffset){
	this->imageRowsOffset = rowOffset;
	this->imageColumnsOffset = columnOffset;
}
