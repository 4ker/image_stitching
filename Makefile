CXX = clang++
CXXFLAGS = -std=c++14 -pedantic-errors -Wall -Ofast -march=native -pipe -flto

# OpenCV Sample 2.4 :
# 	opencv_core opencv_highgui opencv_stitching
#
# OpenCV selfwrite :
# 	opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_calib3d opencv_imgproc

LIBRARIES = opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_calib3d opencv_imgproc opencv_stitching

LDFLAGS = -s $(foreach library, $(LIBRARIES), -l$(library))

opencv_sample_2_4: bin_dir
	$(CXX) $(CXXFLAGS) $(LDFLAGS) src/opencv_samples/stitching_2_4.cpp -o bin/stitching_2_4

opencv_selfwrite: bin_dir
	$(CXX) $(CXXFLAGS) $(LDFLAGS) src/opencv_selfwrite/stitching.cpp -o bin/stitching_selfwrite

bin_dir:
	mkdir -p bin
