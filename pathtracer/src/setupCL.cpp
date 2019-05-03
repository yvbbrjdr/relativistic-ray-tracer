#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define __CL_ENABLE_EXTENSIONS
#include <CL/cl.hpp>
#include <CL/opencl.h>
#endif

using namespace std;
using namespace cl;

cl_context context;
cl_command_queue queue;
cl_program program;
cl_int result;
cl_kernel kernel;

void pickPlatform(cl_platform_id& platform, const vector<cl_platform_id>& platforms) {
	char* buffer = (char *) malloc(100);
	unsigned long int ret;
	for (int i = 0; i < platforms.size(); i++) {
		clGetPlatformInfo(platforms[i],
				CL_PLATFORM_NAME,
				100,
				buffer,
				&ret
				);
		cout << "available platforms:" << endl;
		cout << i + 1 << ": \t" << buffer << endl;
	}
	if (platforms.size() == 1){
		cout << "only one platform" << endl;
		platform = platforms[0];
	}
	else {
		int input = 0;
		cout << "\nChoose and OpenCL platform: ";
		cin >> input;

		while (input < 1 || input > platforms.size()) {
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "no option, try again: ";
			cin >> input;
		}
		platform = platforms[input - 1];

	}
	free(buffer);
}

void pickDevice(cl_device_id& device, vector<cl_device_id>& devices) {
	if (devices.size() == 1)
	{
		cout << "only one device" << endl;
		device = devices[0];
	} else
	{
		int i = 0;
		cout << "\nChoose an OpenCL device: \t" << endl;
		cin >> i;
		while (i < 1 || i > devices.size()) {
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "No option, try again: \t" << endl;
			cin >> i;
		}
		device = devices[i - 1];
	}
	char* buffer = (char *) malloc(100);
	unsigned long int ret;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 100, buffer, &ret);
	cout << "Chose: \t" << buffer << endl;
	free(buffer);
}

int initOpenCL(void) {
	// Create the two input vectors
	int i, j;
	cl_platform_id platform;
	cl_device_id device;
	char* value;
	long unsigned int valueSize;
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	cl_uint maxComputeUnits;

	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	int n = sizeof(platforms) / sizeof(platforms[0]);
	clGetPlatformIDs(platformCount, platforms, NULL);
	vector<cl_platform_id> dest(platforms, platforms + platformCount);
	pickPlatform(platform, dest);
	value = (char *) malloc(100);
	clGetPlatformInfo(platform,
				CL_PLATFORM_NAME,
				100,
				value,
				&valueSize);
	cout << "\nChose platform: \t" << value << endl;
	free(value);
    	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
	// get all devices
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

        }
	vector<cl_device_id> device_ids(devices, devices + deviceCount);
	pickDevice(device, device_ids);
	context = clCreateContext(NULL, device_ids.size(), devices, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);
	

	//Convert source code to a string
	string source;
	std::ifstream file;
	file.open("pathracer.cpp");
	if (!file.is_open())
		{
			cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
			system("PAUSE");
			exit(1);
		}
	while (!file.eof()) {
		char line[256];
		file.getline(line, 255);
		source += line;
	}
	
	const char* kernel_source = source.c_str();

	//Create OpenCL program by performing runtime source compilation for chosen device
	unsigned long int lengths[] = {source.length()};
	program = clCreateProgramWithSource(context, 1, &kernel_source, lengths, NULL);
	result = clBuildProgram(program, device_ids.size(), devices, NULL, NULL, NULL);
	if (result) cout << "Erroro during compilation OpenCL code!!!\n" << result << ")" << endl;
	kernel = clCreateKernel(program,
 "pathtracer", NULL);
	free(devices); 
	free(platforms);
	return 0;
}

int main(void) {
	initOpenCL();
}
