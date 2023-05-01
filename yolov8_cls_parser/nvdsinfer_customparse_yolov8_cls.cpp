#include <cstring>
#include <iostream>
#include <string.h>
#include "nvdsinfer_custom_impl.h"

int num_class = 2;
std::vector<std::string> class_list = { "daisy", "dandelion" };

extern "C" bool NvDsInferParseCustomYolov8Cls(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString
    );

extern "C" bool NvDsInferParseCustomYolov8Cls(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString
    )
{
    NvDsInferAttribute attr;

    unsigned int numAttributes = outputLayersInfo.size();
    
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        if (strcmp(outputLayersInfo[l].layerName, "output1") == 0)
        {
            
            float *outputBuffer = (float *) outputLayersInfo[l].buffer;           

            // get conf
            unsigned int max_conf_index = -1;
            float max_conf = 0.0;
            float conf = 0;

            for (int i=0; i<num_class; i++)
            {
                conf = outputBuffer[i];

                if (max_conf < conf)
                {
                    max_conf = conf;
                    max_conf_index = i;
                }
            }
            
            std::string s = class_list[max_conf_index];
            attr.attributeLabel = strdup(s.c_str()); 
            
            attr.attributeIndex = max_conf_index;
            attr.attributeConfidence = max_conf;                   
            attrList.push_back(attr);

            descString.append(s);
            



 
            
        }
        
    }

    return true;
        
}


