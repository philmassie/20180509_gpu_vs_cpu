from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pickle
from tensorflow.python.client import device_lib

# Use tensorflow library to check the name of my GPU device
# From https://gist.github.com/jovianlin/b5b2c4be45437992df58a7dd13dbafa7
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

# this function returns the time taken to compute the dot products.
# based on https://medium.com/@erikhallstrm/hello-world-tensorflow-649b15aed18c
def get_times(maximum_time, matrix_sizes):
    # I like to declare globals in functions sometimes so I can see the progress
    global msize, device_times
    # dictionary to store results
    device_times = {
        "/device:GPU:0":[0],
        "/device:CPU:0":[0]
    }

    time_taken = 0
    # while(max(device_times["/device:CPU:0"]) < maximum_time):
    for msize in matrix_sizes:
        for device_name in device_times.keys():

            print("####### Calculating " + str(msize) + " on the " + device_name + " #######")

            # build matrices and define the dot product
            shape = (msize, msize)
            data_type = tf.float16
            with tf.device(device_name):
                r1 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                r2 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                dot_operation = tf.matmul(r2, r1)

            # Execute and time the dot product, add the time to the dictionary
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                start_time = time.time()
                result = session.run(dot_operation)
                time_taken = time.time() - start_time
                device_times[device_name].append(time_taken)

        if max(device_times["/device:CPU:0"]) > maximum_time:
            break

    return device_times, matrix_sizes

# call the function
# maximum time - The function will exit when a calculation exceeds this time (seconds).
#                Helps to quit the function a bit sooner if it takes too long, CPU calculations take a long time on my little laptop.
# matrix_sizes - range of matrix dimensions
device_times, matrix_sizes = get_times(maximum_time = 240, matrix_sizes = range(500,3000,50))

# save/read results
pickle.dump(device_times, open( "benchmarking/bm_device_times.p", "wb" ) )
pickle.dump(matrix_sizes, open( "benchmarking/bm_matrix_sizes.p", "wb" ) )

device_times = pickle.load( open( "benchmarking/bm_device_times.p", "rb" ) )
matrix_sizes = pickle.load( open( "benchmarking/bm_matrix_sizes.p", "rb" ) )


# Prepare to plot
gpu_times = device_times["/device:GPU:0"][0:len(matrix_sizes)]
cpu_times = device_times["/device:CPU:0"][0:len(matrix_sizes)]

gpu_curve, = plt.plot(matrix_sizes[:len(gpu_times)], gpu_times, 'o-', label = "GPU Time")
cpu_curve, = plt.plot(matrix_sizes[:len(cpu_times)], cpu_times, 'o-', label = "CPU Time")
plt.legend(handles=[gpu_curve, cpu_curve])
plt.ylabel('Time (Seconds)')
plt.xlabel('Matrix Size')
plt.show()
plt.savefig("benchmark.png", bbox_inches="tight", dpi=150)