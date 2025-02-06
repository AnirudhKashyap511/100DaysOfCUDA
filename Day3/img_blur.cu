#include <stdio.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
__global__ void imgblur(uint8_t *img, uint8_t *blur_img, int w, int h)
{
    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);
    int avg=0;
    if (row < 1 || row >= h-1 || col < 1 || col >= w-1)
    {  
        return;
    }
    else
    {
        int idx = row*w+col;
        for(int i=-1; i<=1; i++)
        {
            for(int j=-1; j<=1; j++)
            {
                avg += img[row*w+col+(i*w)+j];             
            }
        }
        avg = avg/9.0f;
        blur_img[row*w+col] = (uint8_t)(avg);  
    }
}

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("gray_output.png", &width, &height, &bpp, 1);
    
    if (rgb_image == NULL) {
        printf("Error: Failed to load image\n");
        return 1;
    }
    uint8_t* blur_image_d;
    cudaMalloc((void**)&blur_image_d,sizeof(uint8_t)*width*height);
    uint8_t* blur_image_h = (uint8_t*)malloc(sizeof(uint8_t)*width*height);
    printf("Loaded image with width: %d, height: %d, channels: %d\n", width, height, bpp);
    uint8_t *d_img;
    cudaMalloc((void**)&d_img,sizeof(uint8_t)*width*height*1);
    cudaMemcpy(d_img,rgb_image,sizeof(uint8_t)*width*height*1,cudaMemcpyHostToDevice);
    //create the grid and block
    dim3 threadsperblock(16,16);
    dim3 gridsize((width+15)/16,(height+15)/16);
    //imgblurkernel
    imgblur<<<gridsize,threadsperblock>>>(d_img,blur_image_d,width,height);
    cudaMemcpy( blur_image_h, blur_image_d, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);
    stbi_write_png("blur_output.png", width, height, 1, blur_image_h, width);
    cudaFree(blur_image_d);
    free(blur_image_h);
}
