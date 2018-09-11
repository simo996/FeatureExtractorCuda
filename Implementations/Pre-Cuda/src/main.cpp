
#include <iostream>
#include <assert.h>
#include <fstream>
#include <chrono> // Performance monitor
#include "ImageFeatureComputer.h"

using namespace std;
using namespace chrono;


int main(int argc, char* argv[]) {
    cout << argv[0] << endl;

    ProgramArguments pa = ProgramArguments::checkOptions(argc, argv);

    typedef high_resolution_clock Clock;
    Clock::time_point t1 = high_resolution_clock::now();

    // Launch the external component
    ImageFeatureComputer ifc(pa);
    ifc.compute();

    Clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << endl << endl << "* Processing took " << time_span.count() << " seconds." << endl;

    return 0;
}