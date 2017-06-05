#include "trainVoc.h"


int main(int argc, char** argv)
{
	if(argc!=6)
	{
		cerr<<"Usage: ./trainVoc imageDirectory outDirectory vocName k L"<<endl;
		return -1;
	}

	string imageDir=(string)argv[1];
	string outDir=(string)argv[2];
	string vocName=(string)argv[3];
	int k = stoi(argv[4]);
	int L = stoi(argv[5]);

	TrainVOC train(k,L);
	train.trainVoc(imageDir,outDir,vocName);
}