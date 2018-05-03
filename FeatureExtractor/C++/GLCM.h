#include <stdio.h>
#include <stdlib.h>

using namespace std;

/* Class that represents a single GLCM with its angle and distance */
class GLCM
{
	public:
		int * MetaGLCM;
		// Values necessary to identify neighbor pixel
		// A pair for each direction 0°, 90°, 45°, 135°
		int shiftX;
		int shiftY;
		// Sub Borders in the windows obtained according to direction
		int length; // number of DIFFERENT gray level pairs
		void initializeData(int shiftX, int shiftY, int grayLevels, int windowColumns, int windowRows);
		int codify(int * imageElements, int length);
		void sort();
		int dwarf();
	private:
		int distance; // shift size to identify neighbor pixel
		int numberOfPairs; // number of pixel pairs to be found
		int borderX;
		int borderY;
		int maxGrayLevel;
		int computeBorderX(int windowColumns, int windowRows);
		int computeBorderY(int windowColumns, int windowRows);
		int computePairsNumber();
};