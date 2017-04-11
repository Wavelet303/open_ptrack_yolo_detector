#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "run_yolo.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"

#include <time.h>


IplImage* image_to_ipl(image p)
{
	image copy = copy_image(p);
	if(p.c == 3) rgbgr_image(copy);
	int x,y,k;

	IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
	int step = disp->widthStep;
	for(y = 0; y < p.h; ++y){
	    for(x = 0; x < p.w; ++x){
	        for(k= 0; k < p.c; ++k){
	            disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
	        }
	    }
	}
	
	free_image(copy);
	
	return disp;
}

image **load_alphabet_(char* path)
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "%s/labels/%d_%d.png", path, i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void draw_people_detections(image im, int num,  box *boxes)
{
	int i;
	for(i = 0; i < num; i++)
    {
		int width = im.h * .012;

        float red = 1;
        float green = 0;
        float blue = 1;
        float rgb[3];

        box b = boxes[i];

        int left  = (b.x-b.w/2.)*im.w;
        int right = (b.x+b.w/2.)*im.w;
        int top   = (b.y-b.h/2.)*im.h;
        int bot   = (b.y+b.h/2.)*im.h;

        if(left < 0) left = 0;
        if(right > im.w-1) right = im.w-1;
        if(top < 0) top = 0;
        if(bot > im.h-1) bot = im.h-1;

        draw_box_width(im, left, top, right, bot, width, red, green, blue);
	}
}

boxInfo* extractPerson(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes, bool updateIm)
{
	
	
	int i;
	int newNum = 0;

    for(i = 0; i < num; i++)
    {
        int classI = max_index(probs[i], classes);
        float prob = probs[i][classI];
        if(prob > thresh)
        {	
	        if (strcmp(names[classI], "person") == 0)
			{
				newNum++;
			}
        }
    }
    
    
 
   
    
    adjBox* newBoxes = (adjBox*)calloc(newNum, sizeof(adjBox));
    box* newDrawBoxes = (box*)calloc(newNum, sizeof(box));
    int j = 0;
    
    for(i = 0; i < num; i++)
    {
    	int classI = max_index(probs[i], classes);
        float prob = probs[i][classI];
        if(prob > thresh)
        {	
	        if (strcmp(names[classI], "person") == 0)
			{
				//printf( "label = %s\n", names[classI]);
				int left  = (boxes[i].x-boxes[i].w/2.)*im.w;
				int right = (boxes[i].x+boxes[i].w/2.)*im.w;
				int top   = (boxes[i].y-boxes[i].h/2.)*im.h;
				int bot   = (boxes[i].y+boxes[i].h/2.)*im.h;

				if(left < 0) left = 0;
				if(right > im.w-1) right = im.w-1;
				if(top < 0) top = 0;
				if(bot > im.h-1) bot = im.h-1;

				newBoxes[j].x = left;
				newBoxes[j].y = top;
				newBoxes[j].w = right-left;
				newBoxes[j].h = bot-top;
				
				newDrawBoxes[j].x = boxes[i].x;
				newDrawBoxes[j].y = boxes[i].y;
				newDrawBoxes[j].w = boxes[i].w;
				newDrawBoxes[j].h = boxes[i].h;
				
				j++;
			}
        }
    }
    
    
    
    if(updateIm)
    {
    	draw_people_detections(im, newNum,  newDrawBoxes);
	}
    
    
    boxInfo* personBoxes = calloc(1, sizeof(boxInfo));
    personBoxes->boxes = newBoxes;
    personBoxes->num = newNum;
    personBoxes->im = image_to_ipl(im);
    
    //clock_t t;
    //t = clock();
    
    
    //printf( "H in extract = %d\n", personBoxes->im->h);
    //printf( "Number of People In Extract = %d\n", newNum);
   // printf( "Size of boxinfo = %d\n", sizeof(boxInfo));
    free_image(im);
    
    
    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   // printf("took %f seconds to execute \n", time_taken);
    
    return personBoxes;
}

boxInfo* run_yolo_detection(image im, network net, float thresh, float hier_thresh, image **alphabet, char **names, bool updateIm)
{
	int j;
    float nms=.4;
    
    image sized = resize_image(im, net.w, net.h);
    
    layer l = net.layers[net.n-1];
    box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    
    for(j = 0; j < l.w*l.h*l.n; ++j) 
    {
    	probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
    }

    float *X = sized.data;
    network_predict(net, X);
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

    if (l.softmax_tree && nms) 
    {
    	do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    }
    else if (nms) 
    {
    	do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    }

   // free_image(im);
    free_image(sized);
   // free_ptrs((void **)probs, l.w*l.h*l.n);
  // printf( "detect layer (layer %d) w = %d h = %d n = %d\n", net.n, l.w, l.h, l.n);
    return extractPerson(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes, updateIm);
}

