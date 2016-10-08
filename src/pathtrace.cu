#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <ctime>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
	thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
		int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
		return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
							   int iter, glm::vec3* image) {
								   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
								   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

								   if (x < resolution.x && y < resolution.y) {
									   int index = x + (y * resolution.x);
									   glm::vec3 pix = image[index];

									   glm::ivec3 color;
									   color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
									   color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
									   color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

									   // Each thread writes one pixel location in the texture (textel)
									   pbo[index].w = 0;
									   pbo[index].x = color.x;
									   pbo[index].y = color.y;
									   pbo[index].z = color.z;
								   }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment * dev_cachePaths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static bool * dev_flag = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_cachePaths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_flag, pixelcount * sizeof(bool));
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_flag);
	cudaFree(dev_cachePaths);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	bool depthOfField = false;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f);
		
		
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		if (depthOfField)
		{
			// depth of field
			cam.lensDistance = 2.0f;
			cam.focalDistance = 4.0f;
			cam.lensRadius = 10.0f;

			glm::vec3 pointOnFocalPlane = segment.ray.origin + cam.focalDistance * segment.ray.direction;
			glm::vec3 pointOnScreen = segment.ray.origin + segment.ray.direction;
			thrust::default_random_engine rngx = makeSeededRandomEngine(iter, x, 0);
			thrust::default_random_engine rngy = makeSeededRandomEngine(iter, y, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);
			float jitterx = (u01(rngx) - 0.5) * 2 / 40.0f;
			float jittery = (u01(rngy) - 0.5) * 2 / 40.0f;
			//if (x + y == 10)
			//	printf("%f\n", u01(rngx));
			pointOnScreen += cam.right * jitterx + cam.up * jittery;

			segment.ray.origin = pointOnScreen;
			segment.ray.direction = pointOnFocalPlane - pointOnScreen;
		}
		segment.ray.direction = glm::normalize(segment.ray.direction);
		//glm::vec3 lensCentralPoint = cam.position + cam.lensDistance * cam.view;
		//glm::vec3 pointOnScreen = segment.ray.origin + segment.ray.direction;
		//glm::vec3 pointOnLens = segment.ray.origin + cam.lensDistance * segment.ray.direction;
		//glm::vec3 centralDirection = lensCentralPoint - pointOnScreen;
		//glm::vec3 pointOnFocalPlane = pointOnScreen + centralDirection / (cam.lensDistance - 1.0f) * (cam.focalDistance + cam.lensDistance - 1.0f);
		//glm::vec3 temp = centralDirection / (cam.lensDistance - 1.0f);
		//printf("%f %f %f\n", segment.ray.direction[0], segment.ray.direction[1], segment.ray.direction[2]);
		//glm::vec3 pointOnFocalPlane = pointOnScreen + glm::normalize(centralDirection) * (cam.focalDistance + cam.lensDistance - 1.0f);
		//glm::vec3 newDirection = pointOnFocalPlane - pointOnLens;

		//glm::vec3 direction = glm::normalize(cam.view
		//		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		//		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		//		);
		//if (glm::length(pointOnLens - lensCentralPoint) <= cam.lensRadius)
		//{
		//	thrust::default_random_engine rng = makeSeededRandomEngine(iter, x + y, 0);
		//	thrust::uniform_real_distribution<float> u01(0, 1);
		//	float t = u01(rng);
		//	
		//	segment.ray.origin = pointOnLens;
		//	segment.ray.direction = glm::normalize(glm::normalize(newDirection) * (1 - t) + direction * t);
		//	//segment.ray.direction = glm::slerp(glm::(glm::normalize(newDirection), 0.0f), glm::vec4(direction, 0.0f), t);


		//}
		//else
		//{
		//	segment.ray.origin = cam.position;
		//	segment.ray.direction = glm::normalize(cam.view
		//		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		//		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		//		);
		//}
		//segment.ray.direction.g = -
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			else if (material.specular.exponent > 0) {
				pathSegments[idx].color *= materialColor;
			}

			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		} else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeRealMaterial (
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, bool * flag
	, glm::vec3 * image
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment &refPath = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;


			if (refPath.remainingBounces)
			{
				scatterRay(refPath, intersection.t * refPath.ray.direction + refPath.ray.origin, intersection.surfaceNormal, material, rng);
				flag[idx] = true;
			}
			else
			{
				flag[idx] = false;
				image[refPath.pixelIndex] += refPath.color;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.

		} else {
			//image[refPath.pixelIndex] += glm::vec3(0.0f);
			//refPath.remainingBounces = 0; // this line is not necessary if no bug exists
			flag[idx] = false;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

void compressedPathandIntersection(int& num_paths, PathSegment *paths, bool *flag)
{
	thrust::device_ptr<bool> dev_ptrFlag(flag);
	thrust::device_ptr<PathSegment> dev_ptrPaths(paths);
	//thrust::device_ptr<ShadeableIntersection> dev_ptrIntersections(intersections);

	//typedef thrust::tuple<thrust::device_ptr<PathSegment>, thrust::device_ptr<ShadeableIntersection>> IteratorTuple;
	//typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	//ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_ptrPaths, dev_ptrIntersections));
	//ZipIterator zip_end = zip_begin + num_paths;
	thrust::remove_if(dev_ptrPaths, dev_ptrPaths + num_paths, dev_ptrFlag, thrust::logical_not<bool>());
	num_paths = thrust::count_if(dev_ptrFlag, dev_ptrFlag + num_paths, thrust::identity<bool>());
}

// Sort by materialId
typedef thrust::tuple<PathSegment, ShadeableIntersection> Tuple;
class cmp
{
public:
	__host__ __device__ bool operator()(const Tuple &a, const Tuple &b)
	{
		return a.get<1>().materialId < b.get<1>().materialId;
	}
};

void SortByMaterial(int num_paths, PathSegment *dev_paths, ShadeableIntersection *dev_intersections)
{
	thrust::device_ptr<PathSegment> ptrPath(dev_paths);
	thrust::device_ptr<ShadeableIntersection> ptrIntersection(dev_intersections);
	
	typedef thrust::tuple<thrust::device_ptr<PathSegment>, thrust::device_ptr<ShadeableIntersection>> IteratorTuple;
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptrPath, ptrIntersection));
	ZipIterator zip_end = zip_begin + num_paths;
	thrust::sort(zip_begin, zip_end, cmp());
}
/**
* Wrapper for the __global__ call that sets up the kernel calls and does a ton
* of memory management
*/
float total=0.0f;
int counter = 0;
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;


	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	//if (iter == 1)
	{
		generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_cachePaths);
		checkCUDAError("generate camera ray");
	}
	cudaMemcpy(dev_paths, dev_cachePaths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;



		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		//SortByMaterial(num_paths, dev_paths, dev_intersections);
//float time_elapsed=0;
//cudaEvent_t start,stop;
//cudaEventCreate(&start);
//cudaEventCreate(&stop);
//cudaEventRecord( start,0);
		shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_flag,
			dev_image
			);
//cudaEventRecord( stop,0);
//cudaEventSynchronize(start);
//cudaEventSynchronize(stop);
//cudaEventElapsedTime(&time_elapsed,start,stop);
//total+=time_elapsed;
//counter++;


		compressedPathandIntersection(num_paths, dev_paths, dev_flag);
		//printf("numpaths:%d\n", num_paths);
		if (!num_paths)
			iterationComplete = true; // TODO: should be based off stream compaction results.
	}

	// Assemble this iteration and apply it to the image
	//dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////
//if (iter == 100)
//{
//FILE *fp = fopen("shadeTime.txt", "a+");
//fprintf(fp, "%f\n", total / counter);
//fclose(fp);
//}
	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
