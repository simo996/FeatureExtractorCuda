#include "ImageLoader.h"

#define IMG16MAXGRAYLEVEL 65535
#define IMG8MAXGRAYLEVEL 255

Mat ImageLoader::readImage(string fileName, bool cropResolution){
    Mat inputImage;
    try{
            inputImage = imread(fileName, CV_LOAD_IMAGE_ANYDEPTH);
    }
    catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cerr << "Exception occurred: " << err_msg << endl;
    }
    if(! inputImage.data )  // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }
    // If not a grayscale 256/6536 depth, it must be a color image
    if((inputImage.depth() != CV_8UC1) && (inputImage.depth() != CV_16UC1)){
        // reduce color channel from 3 to 1
        cvtColor(inputImage, inputImage, CV_RGB2GRAY);
        inputImage.convertTo(inputImage, CV_8UC1);
    }
    // Eventually reduce gray levels to range 0,255
    if((cropResolution) && (inputImage.depth() != CV_8UC1))
        inputImage.convertTo(inputImage, CV_8UC1);

    return inputImage;
}

// TODO detect and read other values that can be mapped into uint
/* SEE Mat::at() documentation to understand addressing
 * If matrix is of type CV_8U then use Mat.at<uchar>(y,x).
 * If matrix is of type CV_8S then use Mat.at<schar>(y,x).
 * If matrix is of type CV_16U then use Mat.at<ushort>(y,x).
 * If matrix is of type CV_16S then use Mat.at<short>(y,x).
 * If matrix is of type CV_32S then use Mat.at<int>(y,x).
 * If matrix is of type CV_32F then use Mat.at<float>(y,x).
 * If matrix is of type CV_64F then use Mat.at<double>(y,x).
*/

Mat ImageLoader::createDoubleMat(const int rows, const int cols,
        const vector<double>& input){
    Mat_<double> output = Mat(rows, cols, CV_64F);
    // Copy the values into the image
    memcpy(output.data, input.data(), rows * cols * sizeof(double));
    return output;
}

// TODO use generics
// Utility method to iterate on the pysical pixels expressed as uchars
inline void readUchars(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<uchar> MI;
    int address = 0;
    for(MI element = img.begin<uchar>() ; element != img.end<uchar>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

// Utility method to iterate on the pysical pixels expressed as uint
inline void readUint(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<ushort> MI;
    int address = 0;
    for(MI element = img.begin<ushort>() ; element != img.end<ushort>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

Image ImageLoader::readImage(const string fileName, bool cropResolution, bool quantitize, int quantizationMax,
        int borderSize){
    // Open image from file system
    Mat imgRead = readImage(fileName, cropResolution);

    // Create borders to the image
    copyMakeBorder(imgRead, imgRead, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT, 0);

    if((quantitize) && (imgRead.depth() == CV_16UC1) && (quantizationMax > IMG16MAXGRAYLEVEL)){
        cout << "Warning! Provided a quantization level > maximum gray level of the image";
        quantizationMax = IMG16MAXGRAYLEVEL;
    }
    if((quantitize) && (imgRead.depth() == CV_8UC1) && (quantizationMax > IMG8MAXGRAYLEVEL)){
        cout << "Warning! Provided a quantization level > maximum gray level of the image";
        quantizationMax = IMG8MAXGRAYLEVEL;
    }
    if(quantitize)
        imgRead = quantitizeImage(imgRead, quantizationMax);

    // Get the pixels from the image to a standard uint array
    vector<uint> pixels(imgRead.total());

    int maxGrayLevel;
   // TODO think again this mechanism , DRY
    switch (imgRead.type()){
        case CV_16UC1:
            readUint(pixels, imgRead);
            maxGrayLevel = IMG16MAXGRAYLEVEL;
            break;
        case CV_8UC1:
            readUchars(pixels, imgRead);
            maxGrayLevel = IMG8MAXGRAYLEVEL;
            break;
        default:
            cerr << "ERROR! Unsupported depth type: " << imgRead.type();
            exit(-4);
    }
    // CREATE IMAGE abstraction structure
    Image image = Image(pixels, imgRead.rows, imgRead.cols, maxGrayLevel);
    return image;
}


// Debug method
void ImageLoader::showImagePaused(const Mat& img, const string& windowName){
    namedWindow(windowName, WINDOW_AUTOSIZE );// Create a window for display.
    imshow(windowName, img );                   // Show our image inside it.
    waitKey(0);
}

// GLCM can work only with grayscale images
Mat ImageLoader::convertToGrayScale(const Mat& inputImage) {
    // Convert image to a 255 grayscale
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
}

unsigned int quantizationStep(int intensity, int maxLevel, int oldMax){
    return (intensity * maxLevel / oldMax);
}

Mat ImageLoader::quantitizeImage(Mat& img, int maxLevel) {
    Mat convertedImage = img.clone();

    switch(img.depth()){
        case CV_8UC1:{
            typedef MatIterator_<uchar> MI;
            for(MI element = convertedImage.begin<uchar>() ; element != convertedImage.end<uchar>() ; element++)
            {
                int intensity = *element;
                int newIntensity = quantizationStep(intensity, maxLevel, IMG8MAXGRAYLEVEL);
                *element = newIntensity;
            }
            break;
        }
        case CV_16UC1: {
            typedef MatIterator_<ushort> MI;
            for(MI element = convertedImage.begin<ushort>() ; element != convertedImage.end<ushort>() ; element++)
            {
                int intensity = *element;
                int newIntensity = quantizationStep(intensity, maxLevel, IMG16MAXGRAYLEVEL);
                *element = newIntensity;
            }
            break;
        }
    }

    return convertedImage;
}

// Improve clarity in very dark/bright images
Mat ImageLoader::stretchImage(const Mat& inputImage){
    Mat stretched;

    // Stretch can only be applied to gray scale CV_8U
    if(inputImage.type() != CV_8UC1){
        inputImage.convertTo(inputImage, CV_8U);
    }

    Ptr<CLAHE> clahe = createCLAHE(4);
    clahe->apply(inputImage, stretched);

    return stretched;
}

// Perform needed transformation and save the image
void ImageLoader::saveImage(const Mat &img, const string &fileName, bool stretch){
    // Transform to a format that opencv can save with imwrite
    Mat convertedImage = ImageLoader::convertToGrayScale(img);

    if(stretch){
        Mat stretched = stretchImage(convertedImage);
        saveImageToFileSystem(stretched, fileName);
    }
    else
        saveImageToFileSystem(convertedImage, fileName);
}

void ImageLoader::saveImageToFileSystem(const Mat& img, const string& fileName){
    try {
        imwrite(fileName +".png", img);
    }catch (exception& e){
        cout << e.what() << '\n';
        cerr << "Fatal Error! Couldn't save the image";
        exit(-3);
    }
}