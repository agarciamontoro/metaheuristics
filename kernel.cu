#define NUM_SAMPLES {{ NUM_SAMPLES }}
#define MAX_NUM_FEATURES {{ MAX_NUM_FEATURES }}
#define K   {{ K }}

__device__ float computeDistance(float* myFeatures, float* otherFeatures,
								 int numFeatures){
	float distance = 0;

    for (size_t i = 0; i < numFeatures; i++) {
        distance += (myFeatures[i] - otherFeatures[i]) *
					(myFeatures[i] - otherFeatures[i]);
    }

	return sqrt(distance);
}

// Copied from Rosetta Code :)
__device__ void bubble_sort (float2* a, int n) {
    int i, s = 1;
	float2 t;
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

__device__ void updateKNearest(float2* kNearest, float2 newSample){
	// The last (unconsidered element) is the new one
	kNearest[K] = newSample;

	bubble_sort(kNearest, K+1);
}

/**
 * Returns the most repeated element in a sequence
 * @param  arr  Array of integer elements
 * @param  size Number of elements in the array
 * @return      The most repeated element
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

__global__ void scoreSolution(void *devSamples, void *devTarget,
							  void *devResult, int numFeatures){
    // Pointers to the features, the target and the result
    float* globalSamples = (float*)devSamples;
    int* globalTarget = (int*)devTarget;
	int* globalResult = (int*)devResult;

    // The sample represented by this thread is the global identifier of the
    // thread
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

	if(sample >= NUM_SAMPLES){
		return;
	}

    int initOfMyFeatures = sample * numFeatures;
    float myFeatures[MAX_NUM_FEATURES];

    // Population of my features
    for(int i=0; i<numFeatures; i++){
            myFeatures[i] = globalSamples[initOfMyFeatures + i];
    }

	float2 inf;
	inf.x = -1;
	inf.y = 99999999;

	// K (plus one in order to ease the update function) nearest neighbours
	// x: sample index
	// y: distance to the thread sample
	float2 kNearest[K+1];

	for (size_t i = 0; i < K+1; i++) {
		kNearest[i] = inf;
	}

	float2 newSample;

    // Distances between my sample and all the others
    for(int i=0; i<NUM_SAMPLES; i++){
        if(i == sample){
            continue;
        }

		newSample.x = i;
		newSample.y = computeDistance(myFeatures,
									  globalSamples + i * numFeatures,
									  numFeatures);

		updateKNearest(kNearest, newSample);
	}

	int classes[K];

	// Voting method
	for (size_t i = 0; i < K; i++) {
		classes[i] = globalTarget[(int)kNearest[i].x];
	}

	int computedClass = votingMethod(classes, K);

	globalResult[sample] = computedClass == globalTarget[sample] ? 1 : 0;
}
