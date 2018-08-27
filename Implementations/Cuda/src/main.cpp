/*
 ============================================================================
 Name        : test.cu
 Author      : Simone Galimberti
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */


#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono> // Performance monitor
#include "ImageFeatureComputer.h"

using namespace std;
using namespace cv;
using namespace chrono;

/*
__global__ void test(ImageData img, unsigned int * pxs){
	img.printElements(pxs);
}
*/

int main(int argc, char* argv[])
{
    cout << argv[0] << endl;
    ProgramArguments pa = ProgramArguments::checkOptions(argc, argv);

    typedef high_resolution_clock Clock;
    Clock::time_point t1 = high_resolution_clock::now();

    // Launch the external component
    ImageFeatureComputer ifc(pa);
    ifc.compute();

    // COMPUTE THE WORK
    Clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "* Processing took " << time_span.count() << " seconds." << endl;

	return 0;
}
