#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <chrono>
#include <xsimd/xsimd.hpp>

using namespace Eigen;

static MatrixX3d makeOscillRotMat(const double chi, const VectorXd& ome) {
  MatrixX3d rots(3 * ome.size(), 3);
  const double cchi = cos(chi), schi = sin(chi);

  size_t batch_size = xsimd::simd_traits<double>::size;
  size_t vec_size = ome.size();
  size_t num_full_batches = vec_size / batch_size;
  
  // SIMD loop for full batches
  for (size_t i = 0; i < num_full_batches * batch_size; i += batch_size) {
    auto ome_v = xsimd::load_unaligned<double>(&ome[i]);
    auto come_v = xsimd::cos(ome_v);
    auto some_v = xsimd::sin(ome_v);
    for(size_t j = 0; j < batch_size; ++j) {
      rots.block<3, 3>(3*(i+j), 0) <<         come_v[j],    0,         some_v[j],
                                       schi * some_v[j], cchi, -schi * come_v[j],
                                      -cchi * some_v[j], schi,  cchi * come_v[j];
    }
  }

  // Loop for remaining elements, if any
  for (size_t i = num_full_batches * batch_size; i < vec_size; ++i) {
    double ome_v = ome[i];
    double come_v = std::cos(ome_v);
    double some_v = std::sin(ome_v);
    rots.block<3, 3>(3*i, 0) <<         come_v,    0,         some_v,
                                 schi * some_v, cchi, -schi * come_v,
                                -cchi * some_v, schi,  cchi * come_v;
  }

  return rots;
}


void make_sample_rmat_array(const double chi, const double* ome_ptr, size_t ome_count, double* result_ptr) {
  const double cchi = cos(chi), schi = sin(chi);
  size_t batch_size = xsimd::simd_traits<double>::size;
  size_t i = 0;

  for (; i + batch_size - 1 < ome_count; i += batch_size) {
    auto ome_v = xsimd::load_unaligned(&ome_ptr[i]);

    auto come_v = xsimd::cos(ome_v);
    auto some_v = xsimd::sin(ome_v);

    for (size_t j = 0; j < batch_size; ++j) {
      result_ptr[i * 9 + j * 3] = come_v[j];
      result_ptr[i * 9 + j * 3 + 1] = 0.0;
      result_ptr[i * 9 + j * 3 + 2] = some_v[j];

      result_ptr[i * 9 + j * 3 + 3] = schi * some_v[j];
      result_ptr[i * 9 + j * 3 + 4] = cchi;
      result_ptr[i * 9 + j * 3 + 5] = -schi * come_v[j];

      result_ptr[i * 9 + j * 3 + 6] = -cchi * some_v[j];
      result_ptr[i * 9 + j * 3 + 7] = schi;
      result_ptr[i * 9 + j * 3 + 8] = cchi * come_v[j];
    }
  }
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  const int size = 200 * 9;
  const int N = 100000;  // number of tests to run

  VectorXd chi(N);
  std::vector<VectorXd> ome;

  for (int i = 0; i < N; ++i) {
    chi(i) = dis(gen);
    VectorXd _ome(size);
    for (int j = 0; j < size; ++j)
      _ome(j) = dis(gen);
    ome.push_back(_ome);
  }
  auto start = std::chrono::system_clock::now();

  for (int n = 0; n < N; ++n) {
    MatrixX3d result = makeOscillRotMat(chi(n), ome[n]);
  }

  auto end = std::chrono::system_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
  
  start = std::chrono::system_clock::now();

  for (int n = 0; n < N; ++n) {
    double result[9 * size];
    make_sample_rmat_array(chi(n), ome[n].data(), size, result);
  }

  end = std::chrono::system_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  return 0;
}
