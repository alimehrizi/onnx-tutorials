#include"cuda_utils.h"


__device__ float calc_iou(float *box1, float*box2){
    float b1 = box1[5]*2000;
    float b2 = box2[5]*2000;
    float x0 = max(box1[0]+b1,box2[0]+b2);
    float y0 = max(box1[1],box2[1]);
    float x1 = min(box1[2]+b1,box2[2]+b2);
    float y1 = min(box1[3],box2[3]);
    float w = x1-x0;
    float h = y1-y0;
    float area1 = (box1[2]-box1[0])*(box1[3]-box1[1]);
    float area2 = (box2[2]-box2[0])*(box2[3]-box2[1]);
    float intersection = w*h;
    if(w<0 | h<0)return -1;
    float iou = (intersection)/(area1+area2-intersection);
    return iou;
}

__global__ void zero_detections(float* detections){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>MAX_DET_NMS)
        return;
    for(int i=0;i<NUM_DET_ATTR;i++){
        detections[blockIdx.y*MAX_DET_NMS*NUM_DET_ATTR+id*NUM_DET_ATTR+i] = 0;
    }

}


__device__ int argMax(float *data, int num_elements){
    int max_id=0;
    float max_value=-INFINITY;
    for(int i=0;i<num_elements;i++){
        if(data[i]>max_value){
            max_value = data[i];
            max_id = i;
        }
    }
    return max_id;
}

__global__ void naive_nms_kernel(float*raw_detections,float *detections, int* sorted_indices,int *mask_indices, int num_detections, int num_attr){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int idx1 = sorted_indices[blockIdx.y*MAX_DET_NMS+id];
    bool valid=true;
    float *box1 = &raw_detections[blockIdx.y*num_detections*num_attr+idx1*num_attr];
    if(box1[4]<IGNORE_THRESH)
        valid = false;
    else{
        for(int i=0;i<MAX_DET_NMS & i<id;i++){
            int idx2 = sorted_indices[blockIdx.y*MAX_DET_NMS+i];
            float *box2 = &raw_detections[blockIdx.y*num_detections*num_attr+idx2*num_attr];
            if(box2[4]<IGNORE_THRESH)break;
            float iou = calc_iou( box1, box2);

            if(iou>IOU_THR){
                valid = false;
                break;
            }

        }
    }
    if(valid)
        mask_indices[blockIdx.y*MAX_DET_NMS+id] = 1;
    else
        mask_indices[blockIdx.y*MAX_DET_NMS+id] = 0;
    __syncthreads();

    if(id==0)
        for(int i=1;i<MAX_DET_NMS;i++){
            mask_indices[blockIdx.y*MAX_DET_NMS+i] += mask_indices[blockIdx.y*MAX_DET_NMS+i-1];
        }
    __syncthreads();
    if(valid){

        int mask_id = mask_indices[blockIdx.y*MAX_DET_NMS+id] - 1;
        int det_id = blockIdx.y*MAX_DET_NMS*NUM_DET_ATTR+mask_id*NUM_DET_ATTR;
        int data_id = blockIdx.y*num_detections*num_attr+idx1*num_attr;
        //printf(">>>id=%d, mask_id=%d, det_id=%d, data_id=%d||  ",id, mask_id, det_id, data_id);
        for(int i=0;i<NUM_DET_ATTR;i++){
            detections[det_id+i] =  box1[i];
        }
      }

}


__global__ void sort_kernel(float* detections, int* sorted_indices, int num_detections, int num_attr){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // sort indices:
    for(int k=0;k<MAX_DET_NMS;k++){
        int m = k%2;
        if((2*id+m+1)<MAX_DET_NMS){
            int idx1 = sorted_indices[blockIdx.y*MAX_DET_NMS+2*id+m];
            int idx2 = sorted_indices[blockIdx.y*MAX_DET_NMS+2*id+m+1];
            if(detections[blockIdx.y*num_detections*num_attr+idx1*num_attr+4]<detections[blockIdx.y*num_detections*num_attr+idx2*num_attr+4]){
                sorted_indices[blockIdx.y*MAX_DET_NMS+2*id+m] = idx2;
                sorted_indices[blockIdx.y*MAX_DET_NMS+2*id+m+1] = idx1;
            }
        }
        __syncthreads();

    }

}



__global__ void mask_func(float *raw_data, int *conf_mask, int num_detections, int num_attr){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>num_detections)return;

//    __shared__ int mask[1024];



    if(raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+4]>CONF_THR){
        int cls_idx = argMax(&raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+5],num_attr-5);
        conf_mask[blockIdx.y*num_detections+id] = 1;
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+4] *= raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+5+cls_idx];
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+5] = cls_idx;
        float xc = raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+0];
        float yc = raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+1];
        float w = raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+2];
        float h = raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+3];
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+0] = xc - w/2;
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+1] = yc - h/2;
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+2] = xc + w/2;
        raw_data[blockIdx.y*num_detections*num_attr+id*num_attr+3] = yc + h/2;
    }else{
        conf_mask[blockIdx.y*num_detections+id] = 0;
    }

//    __syncthreads();


}


__global__ void gather_indices(float *raw_data, int *conf_mask, int *indices, int*selected, int num_detections, int num_attr){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    selected[blockIdx.y*5+id] = 0;

    for(int i=0;i<num_detections;i++){
        if(conf_mask[blockIdx.y*num_detections+i]){

            indices[blockIdx.y*MAX_DET_NMS+selected[blockIdx.y*5+id]] = i;
            selected[blockIdx.y*5+id]++;
        }

    }

}




void cudaNMS(float *data_d,int batch_size, int num_detections, int num_attr, float * results){
    int * conf_mask_d;
    HANDLE_ERROR(cudaMalloc((void**)&conf_mask_d,batch_size*num_detections*sizeof(int)));

    int * indices_d;
    HANDLE_ERROR(cudaMalloc((void**)&indices_d,batch_size*MAX_DET_NMS*sizeof(int)));

    int * selected_d;
    HANDLE_ERROR(cudaMalloc((void**)&selected_d,batch_size*5*sizeof(int)));

    float * detections_d;
    HANDLE_ERROR(cudaMalloc((void**)&detections_d,batch_size*MAX_DET_NMS*NUM_DET_ATTR*sizeof(float)));


    dim3 dimGrid_0(1,batch_size,1);
    dim3 dimBlock_0(MAX_DET_NMS,1,1);
    zero_detections<<< dimGrid_0, dimBlock_0,0>>>(detections_d);

    dim3 dimGrid(num_detections/1000+1,batch_size,1);
    dim3 dimBlock(1000,1,1);
    mask_func<<< dimGrid, dimBlock>>>(data_d, conf_mask_d, num_detections, num_attr);






    dim3 dimGrid_2(1,batch_size,1);
    dim3 dimBlock_2(1,1,1);
    gather_indices<<< dimGrid_2, dimBlock_2>>>(data_d, conf_mask_d,indices_d, selected_d, num_detections, num_attr);

    dim3 dimGrid_3(1,batch_size,1);
    dim3 dimBlock_3(MAX_DET_NMS/2,1,1);
    sort_kernel<<< dimGrid_3, dimBlock_3>>>(data_d, indices_d, num_detections, num_attr);

    dim3 dimGrid_4(1,batch_size,1);
    dim3 dimBlock_4(MAX_DET_NMS,1,1);
    naive_nms_kernel<<< dimGrid_4, dimBlock_4>>>(data_d,detections_d, indices_d,conf_mask_d, num_detections, num_attr);



    HANDLE_ERROR(cudaMemcpy((void*)results,(void*)detections_d,batch_size*MAX_DET_NMS*NUM_DET_ATTR*sizeof(float),cudaMemcpyDeviceToHost));


    HANDLE_ERROR(cudaFree(detections_d));
    HANDLE_ERROR(cudaFree(conf_mask_d));
    HANDLE_ERROR(cudaFree(indices_d));
    HANDLE_ERROR(cudaFree(selected_d));




    cudaDeviceSynchronize();
    return;
}
