//
// Created by simo on 05/08/18.
//


#include "ImageLoader.h"


Mat ImageLoader::readMriImage(const string fileName, bool cropResolution){
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

void ImageLoader::writeToDouble(const int rows, const int cols,
        const vector<double>& input, Mat& output){
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output.at<double>(r,c) = input[r * rows+ cols];
        }
    }
}

Mat ImageLoader::createDoubleMat(const int rows, const int cols,
                                 const vector<double>& input){
    Mat_<double> output = Mat(rows, cols, CV_64F);
    // Copy the values into the image
    memcpy(output.data, input.data(), rows * cols * sizeof(double));
    return output;
}


inline void readUchars(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<uchar> MI;
    int address = 0;
    for(MI element = img.begin<uchar>() ; element != img.end<uchar>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

inline void readUint(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<ushort> MI;
    int address = 0;
    for(MI element = img.begin<ushort>() ; element != img.end<ushort>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

Image ImageLoader::readImage(const string fileName, bool cropResolution, int borderSize){
    // Open image from file system
    Mat imgRead = readMriImage(fileName, cropResolution);
    printMatImageData(imgRead);
    // Create borders to the image
    copyMakeBorder(imgRead, imgRead, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT, 0);
    // COPY THE IMAGE DATA TO SMALL array
    vector<uint> pixels(imgRead.total());

    int maxGrayLevel;
   // TODO think again this mechanism , DRY
    switch (imgRead.type()){
        case CV_16UC1:
            readUint(pixels, imgRead);
            maxGrayLevel = 65535;
            break;
        case CV_8UC1:
            readUchars(pixels, imgRead);
            maxGrayLevel = 255;
            break;
        default:
            cerr << "ERROR! Unsupported depth type: " << imgRead.type();
            exit(-4);
    }

    // CREATE IMAGE abstraction structure
    Image img = Image(pixels, imgRead.rows, imgRead.cols, maxGrayLevel);
    return img;
}

void ImageLoader::printMatImageData(const Mat& img){
    cout << "\t- Image metadata -" << endl;
    cout << "\tRows: " << img.rows << " x Columns: "  << img.cols << endl;
    cout << "\tPixel count: " << img.total() << endl;
    cout << "\tDynamic: ";
    switch (img.type()){
        case 0:
            cout << "256 gray levels depth";
            break;
        case 2:
            cout << "65536 gray levels depth";
            break;
        default:
            // TODO allow translation from signed to unsigned int
            cerr << "ERROR! Unsupported depth type: " << img.type();
            exit(-4);
            break;
    }
    cout << endl;
    //cout << img;
}

void ImageLoader::showImage(const Mat& img, const string& windowName){
    namedWindow(windowName, WINDOW_AUTOSIZE );// Create a window for display.
    imshow(windowName, img );                   // Show our image inside it.
}

void ImageLoader::showImagePaused(const Mat& img, const string& windowName){
    namedWindow(windowName, WINDOW_AUTOSIZE );// Create a window for display.
    imshow(windowName, img );                   // Show our image inside it.
    waitKey(0);
}

Mat ImageLoader::convertToGrayScale(const Mat& inputImage) {
    // Convert image to a 255 grayscale
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
}

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


void ImageLoader::showImageStretched(const Mat& img, const string& windowName){
    Mat stretched = stretchImage(img);

    showImage(img, "Original" + windowName);
    showImage(stretched, "Stretched" + windowName);
}

void ImageLoader::stretchAndSave(const Mat &img, const string &fileName){
    Mat stretched = stretchImage(img);
    try {
        imwrite(fileName +".png", stretched);
    }catch (exception& e){
        cout << e.what() << '\n';
        cerr << "Fatal Error! Couldn't save the image";
        exit(-3);
    }
}

void ImageLoader::saveImageToFile(const Mat& img, const string& fileName){
    try {
        imwrite(fileName +".png", img);
    }catch (exception& e){
        cout << e.what() << '\n';
        cerr << "Fatal Error! Couldn't save the image";
        exit(-3);
    }
}