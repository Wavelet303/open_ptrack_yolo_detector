#include "network.h"
#include "box.h"
#include "image.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

typedef struct adjBox
{
	int x;
	int y;
	int w;
	int h;
} adjBox;

typedef struct boxInfo
{
	adjBox* boxes;
	int num;
	IplImage* im;
} boxInfo;



image **load_alphabet_(char* path);
boxInfo* extractPerson(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes, bool updateIm);

boxInfo* run_yolo_detection(image im, network net, float thresh, float hier_thresh, image **alphabet, char **names, bool updateIm);

