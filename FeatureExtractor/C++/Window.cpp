//
// Created by simo on 26/07/18.
//

#include "Window.h"

Window::Window(int dimension, int distance, bool symmetric = false){
	this->dimension = dimension;
	this->distance = distance;
	this->symmetric = symmetric;
}

void Window::setDirectionOffsets(int shiftRows, int shiftColumns){
	this->shiftRows = shiftRows;
	this->shiftColumns = shiftColumns;
}
