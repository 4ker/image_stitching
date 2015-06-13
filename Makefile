CXX = g++
CXXFLAGS = -Wall -Ofast -march=native -flto

LIBRARIES = opencv_core opencv_stitching opencv_highgui
LDFLAGS = -s $(foreach library, $(LIBRARIES), -l$(library))

opencv_sample_2_4: bin_dir
	$(CXX) $(CXXFLAGS) $(LDFLAGS) src/opencv_samples/stitching_2_4.cpp -o bin/stitching_2_4

bin_dir:
	mkdir -p bin
