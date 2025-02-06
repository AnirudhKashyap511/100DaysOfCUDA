#include <stdio.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
__global__ void img2gray(uint8_t *img, uint8_t *gray_img, int w, int h)
{
    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if (row < h && col < w) 
    {
        int idx = (row * w + col) * 3;
        uint8_t r = img[idx];
        uint8_t g = img[idx + 1];
        uint8_t b = img[idx + 2];
        uint8_t L = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        gray_img[row * w + col] = L; 
    }
}

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("scenery.jpg", &width, &height, &bpp, 3);
    
    if (rgb_image == NULL) {
        printf("Error: Failed to load image\n");
        return 1;
    }
    uint8_t* gray_image_d;
    cudaMalloc((void**)&gray_image_d,sizeof(uint8_t)*width*height);
    uint8_t* gray_image_h = (uint8_t*)malloc(sizeof(uint8_t)*width*height);
    printf("Loaded image with width: %d, height: %d, channels: %d\n", width, height, bpp);
    uint8_t *d_img;
    cudaMalloc((void**)&d_img,sizeof(uint8_t)*width*height*3);
    cudaMemcpy(d_img,rgb_image,sizeof(uint8_t)*width*height*3,cudaMemcpyHostToDevice);
    //create the grid and block
    dim3 threadsperblock(16,16);
    dim3 gridsize((width+15)/16,(height+15)/16);
    //img2graykernel
    img2gray<<<gridsize,threadsperblock>>>(d_img,gray_image_d,width,height);
    cudaMemcpy( gray_image_h, gray_image_d, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);
    stbi_write_png("gray_output.png", width, height, 1, gray_image_h, width);
    cudaFree(gray_image_d);
    free(gray_image_h);
}
