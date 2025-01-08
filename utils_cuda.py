import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Функція для розрахунку Allan Variance з використанням CUDA
def calculate_allan_variance_cuda(data, dt, max_cluster=100):
    """
    Розраховує Allan Variance з використанням CUDA.
    """
    data = np.array(data, dtype=np.float32)
    n = len(data)
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    mod = SourceModule("""
        __global__ void allan_variance(float *data, int n, int max_cluster, float dt, float *taus, float *allan_vars) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < max_cluster) {
                int cluster_size = tid + 1;
                int num_clusters = n / cluster_size;
                if (num_clusters >= 2) {
                    float sum_diffs = 0.0;
                    for (int i = 0; i < num_clusters - 1; i++) {
                        float mean1 = 0.0;
                        float mean2 = 0.0;
                        for (int j = 0; j < cluster_size; j++) {
                            mean1 += data[i * cluster_size + j];
                            mean2 += data[(i + 1) * cluster_size + j];
                        }
                        mean1 /= cluster_size;
                        mean2 /= cluster_size;
                        sum_diffs += (mean2 - mean1) * (mean2 - mean1);
                    }
                    taus[tid] = cluster_size * dt;
                    allan_vars[tid] = 0.5 * sum_diffs / (num_clusters - 1);
                } else {
                    taus[tid] = 0.0;
                    allan_vars[tid] = 0.0;
                }
            }
        }
    """)

    func = mod.get_function("allan_variance")
    taus = np.zeros(max_cluster, dtype=np.float32)
    allan_vars = np.zeros(max_cluster, dtype=np.float32)
    block_size = min(max_cluster, 256)
    grid_size = (max_cluster + block_size - 1) // block_size
    func(data_gpu, np.int32(n), np.int32(max_cluster), np.float32(dt), cuda.InOut(taus), cuda.InOut(allan_vars),
         block=(block_size, 1, 1), grid=(grid_size, 1))

    # Фільтруємо нульові значення, які виникають, коли num_clusters < 2
    mask = taus > 0
    return taus[mask], allan_vars[mask]


# Функція для розрахунку Drift Rate з використанням CUDA
def calculate_drift_rate_cuda(data, dt):
    """
    Розраховує Drift Rate з використанням CUDA.
    """
    data = np.array(data, dtype=np.float32)
    n = len(data)
    time = np.arange(0, n, dtype=np.float32) * dt

    data_gpu = cuda.mem_alloc(data.nbytes)
    time_gpu = cuda.mem_alloc(time.nbytes)
    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(time_gpu, time)

    mod = SourceModule("""
        __global__ void linear_regression(float *x, float *y, int n, float *slope, float *intercept) {
            float sum_x = 0.0;
            float sum_y = 0.0;
            float sum_xy = 0.0;
            float sum_xx = 0.0;

            for (int i = 0; i < n; i++) {
                sum_x += x[i];
                sum_y += y[i];
                sum_xy += x[i] * y[i];
                sum_xx += x[i] * x[i];
            }

            *slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
            *intercept = (sum_y - *slope * sum_x) / n;
        }
    """)

    func = mod.get_function("linear_regression")
    slope = np.zeros(1, dtype=np.float32)
    intercept = np.zeros(1, dtype=np.float32)
    func(time_gpu, data_gpu, np.int32(n), cuda.InOut(slope), cuda.InOut(intercept), block=(1, 1, 1), grid=(1, 1))

    return slope[0]


# Функція для розрахунку Offset (зсув) з використанням CUDA
def calculate_offset_cuda(data):
    """
    Розраховує Offset (зсув) з використанням CUDA.
    """
    data = np.array(data, dtype=np.float32)
    n = len(data)
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    mod = SourceModule("""
        __global__ void calculate_mean(float *data, int n, float *mean) {
            float sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += data[i];
            }
            *mean = sum / n;
        }
    """)

    func = mod.get_function("calculate_mean")
    mean = np.zeros(1, dtype=np.float32)
    func(data_gpu, np.int32(n), cuda.InOut(mean), block=(1, 1, 1), grid=(1, 1))

    return mean[0]


# Функція для розрахунку PSD з використанням CUDA (потребує доопрацювання)
def calculate_psd_cuda(data, dt):
    """
    Розраховує Power Spectral Density (PSD) з використанням CUDA.
    Потребує подальшого розвитку для повної інтеграції з CUDA.
    """
    # Поки що використовуємо версію для CPU
    frequencies = np.fft.fftfreq(len(data), d=dt)
    psd = np.abs(np.fft.fft(data)) ** 2
    mask = frequencies >= 0
    return frequencies[mask], psd[mask]