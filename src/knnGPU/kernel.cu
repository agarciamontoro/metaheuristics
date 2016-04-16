#define MAX_NUM_SAMPLES {{ MAX_NUM_SAMPLES }}
#define MAX_NUM_FEATURES {{ MAX_NUM_FEATURES }}
#define K   {{ K }}

/**
 * Compute the euclidean distance between myFeatures and otherFeatures.
 * @param  myFeatures    Float pointer to the characteristics values of the first sample.
 * @param  otherFeatures Float pointer to the characteristics values of the second sample.
 * @param  numFeatures   Number of features that both samples have.
 * @return               The euclidean distance between myFeatures and otherFeatures.
 */
__device__ float computeDistance(float* myFeatures, float* otherFeatures,
								 int* idxMask, int numIdx){
	float distance = 0;
	float difference;
	// Compute the squared euclidean distance.
    for (size_t i = 0; i < numIdx; i++) {
		int idx = idxMask[i];
		difference = myFeatures[idx] - otherFeatures[idx];
	    distance += difference * difference;

    }

	// Returns the euclidean distance.
	return distance;
}

/**
 * You know, yet another ordering method.
 * @param a Float2 pointer to a pair of floats (where a.x = index of the sample,
 *          a.y = distance to the current considered sample).
 * @param n Number of samples in the array pointed by a.
 */
__device__ void bubble_sort (float2* a, int n) {
    int i, s = 1;
	float2 t;

	// Stuff copied from a village of La Mancha, the name of which I have no
	// desire to call to mind.
    while (s) {
        s = 0;
        for (i = 1; i < n; i++) {
            if (a[i].y < a[i - 1].y) {
                t = a[i];
                a[i] = a[i - 1];
                a[i - 1] = t;
                s = 1;
            }
        }
    }
}

/**
 * Order kNearest array depending on the stored distance.
 * @param kNearest  Float2 pointer to K+1 samples
 * @param newSample [description]
 */
__device__ void updateKNearest(float2* kNearest, float2 newSample){
	// The last (unconsidered element) is the new one
	kNearest[K] = newSample;

	// Ordering :)
	bubble_sort(kNearest, K+1);
}

/**
 * Returns the most repeated element in a sequence, given that there exist
 * an element with absolute majority. It works with k = 3 but not with k > 3.
 * @param  arr  Array of integer elements
 * @param  size Number of elements in the array
 * @return      The most repeated element in arr
 */
__device__ int votingMethod(int* arr, int size) {
    int current_candidate = arr[0], counter = 0, i;
    for (i = size-1; i >= 0; --i) {
        if (current_candidate == arr[i]) {
            ++counter;
        } else if (counter == 0) {
            current_candidate = arr[i];
            ++counter;
        } else {
            --counter;
        }
    }
    return current_candidate;
}

/**
 * Execute leave-one-out and computes the mean ratio of success
 * @param[in]	devSamples	Pointer to an array of size numSamples*numFeatures,
 * 							where the features of the i-th sample are stored in the slice from
 * 							devSamples[i*numFeatures] to devSamples[i*numFeatures + numFeatures]
 * @param[in]	devTarget   Pointer to an array of size numSamples, where the
 * 							i-th value stores the class of the i-th sample.
 * @param[out]	devResult	Pointer to an array of size numSamples, where the i-th sample will
 * 							store a 1 or a 0 depending on the success of the class prediction.
 * @param[in]	numFeatures Number of features of each sample.
 * @param[in]	numSamples  Number of samples in the data.
 */
__global__ void scoreSolution(void *devSamples, void *devTarget,
							  void *devResult, void * devIdxMask,
							  int numFeatures,int numSamples, int numIdx){
    // Pointers to the features, the target and the result: CUDA global memory :(
    float* globalSamples = (float*)devSamples;
    int* globalTarget = (int*)devTarget;
	int* globalResult = (int*)devResult;
	int* globalIdxMask = (int*)devIdxMask;
    // The sample represented by this thread is the global identifier of the
    // thread
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

	// Stop execution if the sample id is not in the samples range (necessary for
	// generalizing the number of blocks and threads)
	if(sample >= numSamples){
		return;
	}

	// Index of this thread sample features start in the globalSamples array
    int initOfMyFeatures = sample * numFeatures;
    //float myFeatures[MAX_NUM_FEATURES];

    // Population of this thread sample features
    /*for(int i=0; i<numFeatures; i++){
            myFeatures[i] = globalSamples[initOfMyFeatures + i];
    }*/

	// Aux sample with invalid index and  infinite distance for initializing the
	// K nearest neighbours array.
	float2 inf;
	inf.x = -1;
	inf.y = 99999999;

	// K (plus one in order to ease the update function) nearest neighbours, stored
	// as float2, where:
	// 		x: sample index
	// 		y: distance to the thread sample
	float2 kNearest[K+1];

	// initialization of the K nearest neighbours array
	for (size_t i = 0; i < K+1; i++) {
		kNearest[i] = inf;
	}

	// Loop aux variable for storing each remaining sample
	float2 newSample;

    // Computation of distances between this thread sample and the remaining ones.
    // TODO: Improve the efficiency of this loop: the matrix of distances is symmetric, use that!
    // TODO: Maybe use shared memory to improve efficiency.
    for(int i=0; i<numSamples; i++){
		// Leave one out main behaviour: do not consider this sample in order not
		// to bias the final score
        if(i == sample){
            continue;
        }

		// New sample index and distance to this thread sample
		newSample.x = i;
		newSample.y = computeDistance(globalSamples + initOfMyFeatures,
									  globalSamples + i * numFeatures,
									  globalIdxMask, numIdx);

		// Check whether this new sample should be in the K nearest neighbours.
		updateKNearest(kNearest, newSample);
	}

	// Array for storing the classes of the K nearest neighbours.
	int classes[K];

	// Populate the classes array with the classes of the K nearest neighbours.
	for (size_t i = 0; i < K; i++) {
		classes[i] = globalTarget[(int)kNearest[i].x];
	}

	// Voting method. Choose the most repeated class in the classes array.
	// TODO: Generalize to k != 3
	int computedClass = votingMethod(classes, K);

	// Check wether the computed class is equal to the stored class in the actual
	// target array. Set to 1 if success, to 0 if failure.
	globalResult[sample] = computedClass == globalTarget[sample] ? 1 : 0;
}
