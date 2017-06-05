#ifndef TRAIN_VOC
#define TRAIN_VOC

#include <iostream>
#include <vector>
#include <dirent.h>
#include <locale>
#include <algorithm>
#include <iterator>
#include <string>
// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
// defines SiftVocabulary and SiftDatabase

#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DVision/DVision.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>


// Execution Time
#include <sys/time.h>

// Directory creation
//#include <boost/filesystem.hpp>
#include <sys/stat.h>

#include "ORBExtractor.h"

using namespace DBoW2;
using namespace DUtils;
using namespace std;

#define USE_BINARY_FEAT

/// SIFT Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

/// SIFT Database
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBDatabase;

class TrainVOC
{
public:
	TrainVOC(int k_,int L);
	~TrainVOC();

    void trainVoc(const string& imageDirectory,const string& outDirectory,const string& vocName);

	bool writeListFile(string filename, vector<string>& imageNames);

	bool writeDescToBinFile(string filename, 
#ifdef USE_BINARY_FEAT
	vector<unsigned char>& descriptors
#else
	vector<float>& descriptors
#endif
	);

///read descs from binary file
#ifdef USE_BINARY_FEAT
vector<unsigned char> 
#else
vector<float> 
#endif
 readDescFromBinFile(const char* path);

	bool writeFeatToFile(const std::string path, const std::vector<cv::KeyPoint>& kpt);

	vector<cv::KeyPoint> readFeatFromFile(const char* path);

	bool loadFeatures(
#ifdef USE_BINARY_FEAT
	std::vector<std::vector<cv::Mat> > &features,
#else
	vector<vector<vector<float> > > &features,
#endif
        string sDatasetDirectory, string sOutDirectory,
        vector<string> imagesNames, bool justDesc);

	void storeImages(const char* imagesDirectory, vector<string>& imagesNames, bool desc);
	
	void changeStructure(
#ifdef USE_BINARY_FEAT
	const vector<unsigned char> &plain, vector<cv::Mat> &out,
#else
	const vector<float>& plain, vector<vector<float> > &out,
#endif
	int L);
	
	bool createDir(string& path);
	
	bool fileAlreadyExists(string& fileName, string& sDirectory);

private:
	int k;
	int L;
};

#endif