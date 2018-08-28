nvcc src/GrayPair.cu src/AggregatedGrayPair.cu \
src/Direction.cu src/WorkArea.cu src/Features.cu src/Image.cu src/ImageData.cu src/Window.cu \
src/GLCM.cu src/FeatureComputer.cu src/WindowFeatureComputer.cu src/ImageFeatureComputer.cu \
src/main.cpp src/ProgramArguments.cpp src/ImageLoader.cu \
-o ManuallyCompiledCufeat \
-Xptxas -std=c++11 \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
-gencode arch=compute_35,code=compute_35 \
-gencode arch=compute_37,code=compute_37 \
-gencode arch=compute_50,code=compute_50 \
-gencode arch=compute_52,code=compute_52 \
-gencode arch=compute_60,code=compute_60 \
-gencode arch=compute_61,code=compute_61 \
; rm bin/ManuallyCompiledCufeat ; mv ManuallyCompiledCufeat bin/