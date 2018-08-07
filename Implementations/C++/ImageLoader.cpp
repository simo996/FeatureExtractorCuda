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
    if((inputImage.depth() != CV_8UC1) && (inputImage.depth() != CV_16UC1)){
        cerr << "ERROR! Unsupported depth type: " << inputImage.type();
        exit(-4);
    }
    if(! inputImage.data )  // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }
    // TODO think if this burns pixels > 256
    if(cropResolution)
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

Image ImageLoader::readImage(const string fileName, bool cropResolution){
    // Open image from file system
    Mat imgRead = readMriImage(fileName, cropResolution);
    printMatImageData(imgRead);
    // COPY THE IMAGE DATA TO SMALL array
    // This array need to be moved to gpu shared memory
    // Where the data will be put
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

void ImageLoader::printMatImageData(Mat& img){
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

void ImageLoader::showImage(Mat& img, string windowName){
    namedWindow("Original" + windowName, WINDOW_AUTOSIZE );// Create a window for display.
    imshow("Original" + windowName, img );                   // Show our image inside it.
    waitKey(0);
}

void ImageLoader::showImageStretched(Mat& img, string windowName){
    Mat stretched;
    // Transformation need to be called on a gray scale CV_8U
    //img.convertTo(stretched, CV_8U);
    //equalizeHist(img, stretched);

    Ptr<CLAHE> clahe = createCLAHE(4);
    clahe->apply(img, stretched);

    // Original
    showImage(img, "Original-" + windowName);

    // Stretched smartly
    showImage(stretched, "Stretched-" + windowName);

    // TODO think about showing a single image with 2 math hstacked

    waitKey(0);
}

void ImageLoader::saveImageToFile(const Mat& img, const string fileName){
    try {
        imwrite(fileName +".png", img);
    }catch (exception& e){
        cout << e.what() << '\n';
        cerr << "Fatal Error! Couldn't save the image";
        exit(-3);
    }
}