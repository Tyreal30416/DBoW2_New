#include "trainVoc.h"


TrainVOC::TrainVOC(int k_,int L):k(k_),L(L)
{

}

TrainVOC::~TrainVOC()
{

}

void TrainVOC::trainVoc(const string& imageDirectory,const string& outDirectory,const string& vocName)
{
	//vector to store images' name or desc name
	vector<string> imagesNames;
	bool justDesc = false;
	storeImages(imageDirectory.c_str(), imagesNames, false);
	if(imagesNames.empty())
	{
		cout<<"There is no image in the directory, looking for .desc files...."<<endl;
		justDesc = true;
		storeImages(imageDirectory.c_str(), imagesNames, true);

		if(imagesNames.empty())
		{
			cerr<<"Error: no images or desc file in the directory..."<<endl;
			exit(0);
		}
	}

	cout<<"Dataset images: "<<endl;
	for(unsigned int i=0;i<imagesNames.size();++i)
	{
		cout<<"image "<<i<<" = "<<imagesNames[i]<<endl;
	}

	cout<<endl;

	//extract features, and store desc
#ifdef USE_BINARY_FEAT
	string featDir = outDirectory+"/featBinaryDir";
#else
	string featDir = outDirectory+"/featFloatDir";
#endif

	createDir(featDir);

#ifdef USE_BINARY_FEAT
	vector<vector<cv::Mat> > imageFeatures;
#else
	vector<vector<vector<float> > > imageFeatures;
#endif 

	bool isOK=loadFeatures(imageFeatures, imageDirectory, 
		featDir, imagesNames, justDesc);

	//create voc
	const WeightingType weight = TF_IDF;    const ScoringType score = L1_NORM;

#ifdef USE_BINARY_FEAT
    ORBVocabulary voc(k,L,weight,score);
#else
    SiftVocabulary voc(k, L, weight, score);
#endif

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;

    voc.create(imageFeatures);
    cout << "... done!" << endl;
    cout << endl;

    cout << "Vocabulary information: " << endl
        << voc << endl;

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    cout<<outDirectory + "/" + vocName<<endl;
    voc.save(outDirectory + "/" + vocName);
    cout <<"done..." <<endl;
}
    
	
bool TrainVOC::writeListFile(string filename, vector<string>& imageNames)
{
	ofstream file(filename.c_str());
    // write the number of descriptors
    for (int i = 0; i < imageNames.size(); ++i)
    {
        file << imageNames[i] << endl ;
    }
    bool isOk = file.good();
    file.close();
    return isOk;
}

bool TrainVOC::writeDescToBinFile(string filename, 
#ifdef USE_BINARY_FEAT
	vector<unsigned char>& descriptors
#else
	vector<float>& descriptors
#endif
)
{
	ofstream file(filename.c_str(), ofstream::out | ofstream::binary);
    // write the number of descriptors
#ifdef USE_BINARY_FEAT
	const size_t numDesc = descriptors.size()/32;
#else
    const size_t numDesc = descriptors.size()/128;
#endif

    file.write((const char*) &numDesc, sizeof(std::size_t));
    for (int i = 0; i < descriptors.size(); ++i)
    {
#ifdef USE_BINARY_FEAT
    	file.write( (char*) &descriptors[i], sizeof(unsigned char));
#else
        file.write( (char*) &descriptors[i], sizeof(float));
#endif
    }
    bool isOk = file.good();
    file.close();
    return isOk;
}

#ifdef USE_BINARY_FEAT
vector<unsigned char> 
#else
vector<float> 
#endif
TrainVOC::readDescFromBinFile(const char* path)
{
	fstream fs;
    size_t ndesc;

    // Open file and get the number of descriptors
    fs.open(path, ios::in | ios::binary);

    // get the number of descriptors
    fs.read((char*) &ndesc, sizeof(size_t));

#ifdef USE_BINARY_FEAT
    vector<unsigned char> res;
    // Fill the matrix in the air
    for (int i = 0; i < ndesc*32; i++)
    {
        unsigned char cur;
        fs.read((char*) &cur, sizeof(unsigned char));
        res.push_back(cur);
    }
#else
    vector<float> res;
    // Fill the matrix in the air
    for (int i = 0; i < ndesc*128; i++)
    {
        float cur;
        fs.read((char*) &cur, sizeof(float));
        res.push_back(cur);
    }
#endif

    // Close file and return
    fs.close();
    return res;
}

bool TrainVOC::writeFeatToFile(const std::string path, const std::vector<cv::KeyPoint>& kpt)
{
	ofstream file(path.c_str());
    for (vector<cv::KeyPoint>::const_iterator it = kpt.begin(); it != kpt.end(); ++it)
        file << (*it).pt.x << " " << (*it).pt.y << " " << (*it).size << " " << (*it).angle << endl;

    bool isOk = file.good();
    file.close();

    return isOk;
}

vector<cv::KeyPoint> TrainVOC::readFeatFromFile(const char* path)
{
	vector<cv::KeyPoint> res;
    fstream fs;

    // Load file
    fs.open(path, ios::in);
    if(!fs.is_open())
    {
        std::cout << "Error when opening directory" << std::endl;
        return res;
    }

    // Fill the vector
    float x, y, size, angle;
    while (fs >> x >> y >> size >> angle)
    {
        res.push_back(cv::KeyPoint(x,y,size,angle));
    }

    // Close file and return
    fs.close();
    return res;
}

bool TrainVOC::loadFeatures(
#ifdef USE_BINARY_FEAT
	std::vector<std::vector<cv::Mat> > &features,
#else
	vector<vector<vector<float> > > &features,
#endif
        string sDatasetDirectory, string sOutDirectory,
        vector<string> imagesNames, bool justDesc)
{
	bool goodDescType = true;
    features.clear();
    //features.reserve(imagesNames.size());

#ifdef USE_BINARY_FEAT
    MarkerDetector::ORBextractor extractor(1000,1.2f,8, 15, 10);
#else
    cv::SIFT sift(500, 3, 0.04, 10, 1.6);
#endif

    bool loadFile = false;

    for(int i = 0; i < imagesNames.size(); ++i)
    {
        stringstream ss;
        ss << sDatasetDirectory << "/" << imagesNames[i];

        string descFileName = imagesNames[i].substr(0, imagesNames[i].find_last_of(".")) + ".desc";
        string featFileName = imagesNames[i].substr(0, imagesNames[i].find_last_of(".")) + ".feat";

#ifdef USE_BINARY_FEAT
        vector<unsigned char> descriptors;
#else  
        vector<float> descriptors;
#endif

        bool descFileExists = fileAlreadyExists(descFileName, sOutDirectory);
        bool featFileExists = fileAlreadyExists(featFileName, sOutDirectory);

//        if (!fileAlreadyExists(descFileName, sOutDirectory))
        if (!descFileExists || !featFileExists)
        {
            cout << "File " << sOutDirectory + "/" + descFileName << " does not exist" << endl;
            cv::Mat image = cv::imread(ss.str(), 0);
            cv::Mat mask;
            vector<cv::KeyPoint> keypoints;
            cv::Mat matDescriptors;
#ifdef USE_BINARY_FEAT
            extractor(image,mask,keypoints,matDescriptors);
            if(keypoints.empty())
            {
            	cerr<<"Warning: image can not extract features!!"<<endl;
            	continue;
            }

            descriptors.assign((unsigned char*)matDescriptors.datastart,
                    (unsigned char*)matDescriptors.dataend);
#else
            sift(image, mask, keypoints, matDescriptors);
            descriptors.assign((float*)matDescriptors.datastart,
                    (float*)matDescriptors.dataend);
#endif      
			
			//draw keypoints
			cv::Mat outImage;
			cv::drawKeypoints(image, keypoints, outImage);
			cv::waitKey(5);
			cv::imshow("kpts", outImage);

            if (!descFileExists)
                writeDescToBinFile(sOutDirectory+"/"+descFileName, descriptors);
            if (!featFileExists)
                writeFeatToFile(sOutDirectory+"/"+featFileName, keypoints);
        }

       	//if features were stored , just loaded 
       	string path = sOutDirectory + "/" + descFileName;
        descriptors = readDescFromBinFile(path.c_str());

#ifdef USE_BINARY_FEAT
        //notice: for orb 
        cout<<"Descriptor size is "<<descriptors.size()/32<<endl;
        cout<<endl;

        /// some images maybe have no binary features
        bool validDesc = true;
        int invalidCont=0;
        for(int i=0; i<descriptors.size();++i)
        {
        	if(descriptors[i] == '0')
        		invalidCont++;
        }   

        if(invalidCont==descriptors.size())
        	validDesc = false;

        if(descriptors.empty() || !validDesc)
        {
        	cerr<<"Warning: invalid descriptor detected..."<<endl;
        	continue;
        }

        features.push_back(vector<cv::Mat>());
        changeStructure(descriptors, features.back(), 32);
#else
        cout<<"Descriptor size is "<<descriptors.size()/128<<endl;
        cout<<endl;

        /// some images maybe have no binary features
        bool validDesc = true;
        int invalidCont=0;
        for(int i=0; i<descriptors.size();++i)
        {
        	if(descriptors[i] == 0.)
        		invalidCont++;
        }   

        if(invalidCont==descriptors.size())
        	validDesc = false;

        if(descriptors.empty() || !validDesc)
        {
        	cerr<<"Warning: invalid descriptor detected..."<<endl;
        	continue;
        }

        //push back
        features.push_back(vector<vector<float> >());
        changeStructure(descriptors, features.back(), sift.descriptorSize());
#endif
    }
    return goodDescType;
}

void TrainVOC::storeImages(const char* imagesDirectory, vector<string>& imagesNames, bool desc)
{
// image extensions
    vector<string> extensions;
    if (desc)
    {
        extensions.push_back("desc");
    }
    else
    {
        extensions.push_back("png");
        extensions.push_back("jpg");
        extensions.push_back("jpeg");
    }

    DIR * repertoire = opendir(imagesDirectory);

    if ( repertoire == NULL)
    {
        cout << "The images directory: " << imagesDirectory
            << " cannot be found" << endl;
    }
    else
    {
        struct dirent * ent;
        while ( (ent = readdir(repertoire)) != NULL)
        {
            string file_name = ent->d_name;
            string extension = file_name.substr(file_name.find_last_of(".") +1);
            locale loc;
            for (std::string::size_type j = 0; j < extension.size(); j++)
            {
                extension[j] = std::tolower(extension[j], loc);
            }
            for (unsigned int i = 0; i < extensions.size(); i++)
            {
                if (extension == extensions[i])
                {
                    imagesNames.push_back(file_name);
                }
            }
        }
    }
    closedir(repertoire);
}
	
void TrainVOC::changeStructure(
#ifdef USE_BINARY_FEAT
	const vector<unsigned char> &plain, vector<cv::Mat> &out,
#else
	const vector<float>& plain, vector<vector<float> > &out,
#endif
	int L)
{
	out.resize(plain.size() / L);

#ifdef USE_BINARY_FEAT
	unsigned int j = 0;
	for(unsigned int i = 0; i < plain.size(); i += L, ++j)
	{
		vector<unsigned char> tmp;
		tmp.clear();
		std::copy(plain.begin()+i, plain.begin()+i+L, back_inserter(tmp));

		cv::Mat desc(1,L,CV_8U);

		memcpy(desc.data, tmp.data(), tmp.size()*sizeof(uchar));

		out[i/L]=desc;

	}

#else
	unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        out[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
    }
#endif
}
	
bool TrainVOC::createDir(string& path)
{
	DIR * repertoire = opendir(path.c_str());

    if (repertoire == NULL)
    {
//        boost::filesystem::path dir(path);
//        if (boost::filesystem::create_directory(dir))
//        {
          mkdir(path.c_str(), S_IRWXU);
          return true;
//        }
//        else
//        {
//            std::cout << "Error when creating directory " << path << std::endl;
//            return false;
//        }
    }
    closedir(repertoire);

    return false;
}
	
bool TrainVOC::fileAlreadyExists(string& fileName, string& sDirectory)
{
	DIR * repertoire = opendir(sDirectory.c_str());

    if (repertoire == NULL)
    {
        cout << "The output directory: " << sDirectory
            << " cannot be found" << endl;
    }
    else
    {
        struct dirent * ent;
        while ( (ent = readdir(repertoire)) != NULL)
        {
            if (strncmp(ent->d_name, fileName.c_str(), fileName.size()) == 0)
            {
                cout << "The file " << fileName
                    << " already exists, there is no need to recreate it" << endl;
                closedir(repertoire);
                return true;
            }
        }
    }
    closedir(repertoire);
    return false;
}