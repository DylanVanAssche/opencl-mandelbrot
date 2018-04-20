#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "time_utils.h"
#include "ocl_utils.h"

#include <opencv2/imgcodecs/imgcodecs_c.h>

#define WIDTH (640 * 10)
#define HEIGHT (480 * 10)
#define DIMENSION 2

#define MIN_REAL -2.0
#define MAX_REAL 1.0
#define MIN_IMAGINARY -1.1f
#define MAX_IMAGINARY (MIN_IMAGINARY + (MAX_REAL - MIN_REAL) *\
(HEIGHT) / (WIDTH))
//#define MAX_IMAGINARY 1.2
#define MAX_ITERATIONS 200

#define IMAGINARY_POS(y)\
(float)(MAX_IMAGINARY -\
(y) * ((MAX_IMAGINARY - MIN_IMAGINARY) / (float)((HEIGHT) - 1)))

#define REAL_POS(x)\
(float)(MIN_REAL + (x) * ((MAX_REAL - MIN_REAL) / (float)((WIDTH) - 1)))

// Make a buffer on the GPU to provide access to the image
cl_mem makeBufferOnGPU(void)
{
	// Error code
	cl_int error;

	// Make 2D buffer die leeg is
    cl_mem image_buffer = clCreateBuffer(
			g_context,
            CL_MEM_WRITE_ONLY,
            sizeof(cl_float) * WIDTH * HEIGHT, 
			NULL, 
			&error
		);

	// Set error location code
    ocl_err(error);

	// Process and check for errros
    ocl_err(clFinish(g_command_queue));
    return image_buffer;
}

void render_mandelbrot(CvMat * output_image)
{
	cl_int error;
    // Create device buffer
    cl_mem dev_img_buffer = makeBufferOnGPU();

    // Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "calc_mandel_pixel", &error);
    ocl_err(error);

    // Set kernel arguments
    int arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_img_buffer));

    // Call kernel 2D
    size_t global_work_sizes[] = {WIDTH, HEIGHT}; // 2D
    time_measure_start("computation"); 
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, DIMENSION, NULL, global_work_sizes, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, dev_img_buffer, CL_TRUE, 0, sizeof(cl_float) * WIDTH * HEIGHT, output_image->data.fl, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");
}

int main(int argc, char *argv[])
{
	cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");
	makeBufferOnGPU();
    CvMat * output_image = cvCreateMat(HEIGHT, WIDTH, CV_32FC1);

    time_measure_start("total");
    render_mandelbrot(output_image);
    time_measure_stop_and_print("total");

    cvSaveImage("mandelbrot.png", output_image, 0);
}
