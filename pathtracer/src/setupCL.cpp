#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define __CL_ENABLE_EXTENSIONS
#include <CL/cl.hpp>
#include <CL/opencl.h>
#endif

using namespace std;
using namespace cl;

void pickPlatform(cl_platform_id& platform, const vector<cl_platform_id>& platforms) {
 		
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
}

int main(void) {
    // Create the two input vectors
    int i, j;
    cl_platform_id platform;
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
    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

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

        free(devices);

    }

    free(platforms);
    return 0;
}
