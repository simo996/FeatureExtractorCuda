#include <stdio.h>
#include <stdlib.h>
#include "GLCM.h"

using namespace std;

/* Class that represents a single windows in an image 
	Will generate GLCMs for each direction and demand feature computation
	Will recycle GLCMs shifting 2 pixel left
*/

class Window
{
	public:
		int * pixels; // Linear matrix
		int firstRow;
		int firstColumn;
		int side; // Assumption: all windows are squared
		void generateGLCMs();
		int * features[14]; // 14 for each direction 14*4
		int * computeFeatures();
		void next(); // Move 2 rows, adjust metaGLCM, restart computation
	private:
		GLCM glcm0;
		void generateGLCM();
}