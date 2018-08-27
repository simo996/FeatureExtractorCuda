/*
 * WindowFeatureComputer.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef WINDOWFEATURECOMPUTER_H_
#define WINDOWFEATURECOMPUTER_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "FeatureComputer.h"
#include "Direction.h"

using namespace std;

class WindowFeatureComputer {
    /*
   * RESPONSABILITA CLASSE: Computare le feature per la finestra nelle 4 direzioni
     * Fornire un stream di rappresentazione verso file
   */

public:
    CUDA_DEV WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    // Will be computed features in the directions specified
    // Default = 4 = all feautures ; oder 0->45->90->135°
    CUDA_DEV void computeWindowFeatures();
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
private:
        // Initialization data to pass to each FeatureComputer
        WorkArea& workArea;
        unsigned int * pixels;
        ImageData image;
        Window windowData;
};
#endif /* WINDOWFEATURECOMPUTER_H_ */
