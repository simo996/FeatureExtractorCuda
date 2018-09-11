#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono> // Performance monitor
#include "ImageFeatureComputer.h"

using namespace std;
using namespace cv;
using namespace chrono;


int main(int argc, char* argv[])
{
    ProgramArguments pa = ProgramArguments::checkOptions(argc, argv);

    typedef high_resolution_clock Clock;
    Clock::time_point t1 = high_resolution_clock::now();

    // Launch the external component
    ImageFeatureComputer ifc(pa);
    ifc.compute();

    // COMPUTE THE WORK
    Clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << endl << endl << "* Processing took " << time_span.count() << " seconds." << endl;

	return 0;
}
