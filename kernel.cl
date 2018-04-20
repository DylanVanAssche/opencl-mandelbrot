#define WIDTH (640 * 10)
#define HEIGHT (480 * 10)

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

__kernel void calc_mandel_pixel(__global float *buffer_image)
{
	// 2D array so get_global_id = 1
    const int x_pos = get_global_id(0);
	const int y_pos = get_global_id(1);

	// Avoid too much allocation of threads
    if (x_pos*y_pos >= (WIDTH * HEIGHT))
        return;

	// Execute
  	float real = REAL_POS(x_pos);
    float img = IMAGINARY_POS(y_pos);

    float z_real = real;
    float z_img = img;

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        float z_real_squared = z_real * z_real;
        float z_img_squared = z_img * z_img;

        if (z_real_squared + z_img_squared > 4)
        {
            buffer_image[y_pos * WIDTH + x_pos] = ((float) i / (float)MAX_ITERATIONS) * 255.f;
			return;
        }
        float tmp = z_real_squared - z_img_squared + real;
        z_img = 2 * z_real * z_img + img;
        z_real = tmp;
    }

	buffer_image[y_pos * WIDTH + x_pos] = 0.0; // empty
}

