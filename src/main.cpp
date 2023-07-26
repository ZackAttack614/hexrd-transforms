#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <xsimd/xsimd.hpp>

using namespace Eigen;

namespace py = pybind11;

const Vector3f Zl = {0.0, 0.0, 1.0};
const float eps = std::numeric_limits<float>::epsilon();
const float sqrtEps = std::sqrt(std::numeric_limits<float>::epsilon());

float quat_distance(Vector4f q1, Vector4f q2, MatrixX4f qsym) {
  float q0_max = 0.;
  for (int i = 0; i < qsym.rows(); i++) {
    const Vector4f qs = qsym.row(i);
    Vector4f q2s(
      q2(0)*qs(0) - q2(1)*qs(1) - q2(2)*qs(2) - q2(3)*qs(3),
      q2(1)*qs(0) + q2(0)*qs(1) - q2(3)*qs(2) + q2(2)*qs(3),
      q2(2)*qs(0) + q2(3)*qs(1) + q2(0)*qs(2) - q2(1)*qs(3),
      q2(3)*qs(0) - q2(2)*qs(1) + q2(1)*qs(2) + q2(0)*qs(3)
    );

    q0_max = std::max(fabs(q1.dot(q2s)), q0_max);
  }

  if (q0_max <= 1.0)
    return 2.0 * acos(q0_max);
  else if (q0_max - 1. < 1e-12)
    return 0.;
  return NAN;
}

MatrixXf rotate_vecs_about_axis(VectorXf angles, MatrixXf axes, MatrixXf vecs)
{
  MatrixXf rVecs(vecs.rows(), vecs.cols());

  // Normalize axes, as apparently required for Eigen's AngleAxisf
  axes.rowwise().normalize();

  for (int i = 0; i < angles.size(); ++i) 
  {
    AngleAxisf rotation_vector(angles(i), axes.row(i));
    Vector3f vec3f = vecs.row(i);
    rVecs.row(i) = (rotation_vector * vec3f).transpose();
  }

  return rVecs;
}

Array<bool, Dynamic, 1> validateAngleRanges(const VectorXf& angList, const VectorXf& angMin, const VectorXf& angMax, bool ccw)
{
    const float twoPi = 2.0 * M_PI;

    VectorXf startPtr = ccw ? angMin : angMax;
    VectorXf stopPtr  = ccw ? angMax : angMin;

    Array<bool, Dynamic, 1> result(angList.size());
    result.fill(false);

    for (int i = 0; i < angList.size(); i++) {
        for (int j = 0; j < startPtr.size(); j++) {
            float thetaMax = std::fmod(stopPtr[j] - startPtr[j], twoPi);
            float theta = std::fmod(angList[i] - startPtr[j], twoPi);

            if (std::fabs(thetaMax) < sqrtEps ||
                std::fabs(thetaMax - twoPi) < sqrtEps ||
                (theta >= -sqrtEps && theta <= thetaMax + sqrtEps) )
            {
                result[i] = true;
                break;
            }
        }
    }
    return result;
}

const static Matrix3f makeEtaFrameRotMat(const Vector3f& b, const Vector3f& e) noexcept {
  const Vector3f yHat = e.cross(b).normalized();
  const Vector3f bHat = b.normalized();
  const Vector3f xHat = bHat.cross(yHat);
  Matrix3f r;
  r << xHat, yHat, -bHat;

  return r;
}

const static MatrixXf makeBinaryRotMat(const Vector3f& a) {
    Matrix3f r = 2.0 * a * a.transpose();
    r.diagonal() -= Vector3f::Ones();
    return r;
}

MatrixXf makeRotMatOfExpMap(const Vector3f& e) {
    AngleAxisf rotation(e.norm(), e.normalized());
    return rotation.toRotationMatrix();
}

const static MatrixXf makeOscillRotMat_internal(float chi, float ome) {
  Matrix3f r;
  r = AngleAxisf(chi, Vector3f::UnitX())
    * AngleAxisf(ome, Vector3f::UnitY());

  return r;
}

static MatrixX3f makeOscillRotMat(float chi, const VectorXf& ome) {
  MatrixX3f rots(3 * ome.size(), 3);
  const float cchi = cos(chi), schi = sin(chi);
  xsimd::batch<float, 4UL> come_v, some_v;

  const size_t batch_size = xsimd::simd_traits<float>::size;
  const size_t vec_size = ome.size();

  for (size_t i = 0; i < vec_size; i += batch_size) {
    auto ome_v = xsimd::load_unaligned<float>(&ome[i]);
    come_v = xsimd::cos(ome_v);
    some_v = xsimd::sin(ome_v);

    for(size_t j = 0; j < batch_size && (i+j) < vec_size; ++j) {
      rots.block<3, 3>(3*(i+j), 0) <<         come_v[j],    0,         some_v[j],
                                       schi * some_v[j], cchi, -schi * come_v[j],
                                      -cchi * some_v[j], schi,  cchi * come_v[j];
    }
  }
  return rots;
}

const static MatrixXf makeOscillRotMatSingle(float chi, float ome) {
  return makeOscillRotMat_internal(chi, ome);
}

const static MatrixXf unitRowVectors(const MatrixXf& cIn) {
  return cIn.rowwise().normalized();
}

const static VectorXf unitRowVector(const VectorXf& cIn) {
  return cIn.normalized();
}

const static MatrixXf anglesToGvec(const MatrixXf& angs, const Vector3f& bHat_l, const Vector3f& eHat_l, float chi, const Matrix3f& rMat_c) noexcept {
  const size_t batch_size = xsimd::simd_traits<float>::size;
  const size_t vec_size = angs.rows();
  const size_t num_full_batches = vec_size / batch_size;

  const float cchi = cos(chi), schi = sin(chi);
  MatrixXf gVec_l(angs.rows(), 3);
  MatrixXf gVec_c(angs.rows(), 3);

  // SIMD loop for full batches
  for (size_t i = 0; i < num_full_batches * batch_size; i += batch_size) {
    auto half_angs_v = xsimd::load_unaligned<float>(angs.col(0).data() + i) * 0.5;
    auto cosHalfAngs = xsimd::cos(half_angs_v);
    auto sinHalfAngs = xsimd::sin(half_angs_v);
    
    auto angs1_v = xsimd::load_unaligned<float>(angs.col(1).data() + i);
    auto cosAngs1 = xsimd::cos(angs1_v);
    auto sinAngs1 = xsimd::sin(angs1_v);
    
    for (size_t j = 0; j < batch_size; ++j) {
      gVec_l.row(i + j) << cosHalfAngs[j] * cosAngs1[j], cosHalfAngs[j] * sinAngs1[j], sinHalfAngs[j];
    }
  }

  // Loop for remaining elements, if any
  for (size_t i = num_full_batches * batch_size; i < vec_size; ++i) {
    float half_ang = angs(i, 0) * 0.5;
    float cosHalfAng = std::cos(half_ang);
    float sinHalfAng = std::sin(half_ang);

    float ang1 = angs(i, 1);
    float cosAng1 = std::cos(ang1);
    float sinAng1 = std::sin(ang1);

    gVec_l.row(i) << cosHalfAng * cosAng1, cosHalfAng * sinAng1, sinHalfAng;
  }

  // Transform gVec_l to lab frame
  gVec_l = (makeEtaFrameRotMat(bHat_l, eHat_l) * gVec_l.transpose()).eval();

  #pragma omp parallel for
  for (int i = 0; i < angs.rows(); i++) {
    float come = cos(angs(i, 2));
    float some = sin(angs(i, 2));

    gVec_c.row(i) << (rMat_c(0,0)*come + rMat_c(2,0)*some) * gVec_l(0, i) + (rMat_c(0,0)*some*schi + rMat_c(1,0)*cchi - rMat_c(2,0)*come*schi) * gVec_l(1, i) + (-rMat_c(0,0)*some*cchi + rMat_c(1,0)*schi + rMat_c(2,0)*come*cchi) * gVec_l(2, i),
                     (rMat_c(0,1)*come + rMat_c(2,1)*some) * gVec_l(0, i) + (rMat_c(0,1)*some*schi + rMat_c(1,1)*cchi - rMat_c(2,1)*come*schi) * gVec_l(1, i) + (-rMat_c(0,1)*some*cchi + rMat_c(1,1)*schi + rMat_c(2,1)*come*cchi) * gVec_l(2, i),
                     (rMat_c(0,2)*come + rMat_c(2,2)*some) * gVec_l(0, i) + (rMat_c(0,2)*some*schi + rMat_c(1,2)*cchi - rMat_c(2,2)*come*schi) * gVec_l(1, i) + (-rMat_c(0,2)*some*cchi + rMat_c(1,2)*schi + rMat_c(2,2)*come*cchi) * gVec_l(2, i);
  }

  return gVec_c;
}

MatrixXf anglesToDvec(MatrixXf& angs, Vector3f& bHat_l, Vector3f& eHat_l, float chi, Matrix3f& rMat_c) {
    // Construct matrix where each row represents a gVec_l calculated for each angle
    MatrixXf sinAngs0 = angs.col(0).array().sin();
    MatrixXf cosAngs0 = angs.col(0).array().cos();
    MatrixXf cosAngs1 = angs.col(1).array().cos();
    MatrixXf sinAngs1 = angs.col(1).array().sin();
    MatrixXf gVec_l(angs.rows(), 3);
    gVec_l << sinAngs0.cwiseProduct(cosAngs1), sinAngs0.cwiseProduct(sinAngs1), -cosAngs0;

    // Make eta frame cob matrix and transform gVec_l to lab frame
    Matrix3f rMat_e = makeEtaFrameRotMat(bHat_l, eHat_l);
    gVec_l = (rMat_e * gVec_l.transpose()).transpose();

    // Calculate rotation matrices for each pair of angles chi and ome
    std::vector<Matrix3f> rotation_matrices;
    for (int i = 0; i < angs.rows(); i++) {
        rotation_matrices.push_back(makeOscillRotMat_internal(chi, angs(i, 2)));
    }

    // Multiply rotation matrices by rMat_c
    std::vector<Matrix3f> result_matrices;
    for (auto &rotMat_s : rotation_matrices) {
        result_matrices.push_back(rMat_c.transpose() * rotMat_s.transpose());
    }

    // Compute the dot product of result_matrices and gVec_l
    MatrixXf gVec_c(angs.rows(), 3);
    for (int i = 0; i < angs.rows(); i++) {
        gVec_c.row(i) = (result_matrices[i] * gVec_l.row(i).transpose()).transpose();
    }

    return gVec_c;
}

const static Vector2f gvecToDetectorXYOne(const Vector3f& gVec_c, const Matrix3f& rMat_d,
                                    const Matrix3f& rMat_sc, const Vector3f& tVec_d,
                                    const Vector3f& bHat_l, const Vector3f& nVec_l,
                                    float num, const Vector3f& P0_l)
{
  Vector3f gHat_c = gVec_c.normalized();
  Vector3f gVec_l = rMat_sc * gHat_c;
  float bDot = -bHat_l.dot(gVec_l);

  if ( bDot >= eps && bDot <= 1.0 - eps ) {
    Matrix3f brMat = makeBinaryRotMat(gVec_l);

    Vector3f dVec_l = -brMat * bHat_l;
    float denom = nVec_l.dot(dVec_l);

    if ( denom < -eps ) {
      Vector3f P2_l = P0_l + dVec_l * num / denom;
      Vector2f result;
      result[0] = (rMat_d.col(0).dot(P2_l - tVec_d));
      result[1] = (rMat_d.col(1).dot(P2_l - tVec_d));

      return result;
    }
  }

  return Vector2f(NAN, NAN);
}

const static MatrixXf gvecToDetectorXY(const MatrixXf& gVec_c, const Matrix3f& rMat_d,
                                 const MatrixXf& rMat_s, const Matrix3f& rMat_c,
                                 const Vector3f& tVec_d, const Vector3f& tVec_s,
                                 const Vector3f& tVec_c, const Vector3f& beamVec)
{
    int npts = gVec_c.rows();
    MatrixXf result(npts, 2);

    Vector3f bHat_l = beamVec.normalized();

    for (int i=0; i<npts; i++) {
        Vector3f nVec_l = rMat_d * Zl;
        Vector3f P0_l = tVec_s + rMat_s.block<3,3>(i*3, 0) * tVec_c;
        Vector3f P3_l = tVec_d;

        float num = nVec_l.dot(P3_l - P0_l);

        Matrix3f rMat_sc = rMat_s.block<3,3>(i*3, 0) * rMat_c;

        result.row(i) = gvecToDetectorXYOne(gVec_c.row(i), rMat_d, rMat_sc, tVec_d,
                                            bHat_l, nVec_l, num, P0_l).transpose();
    }

    return result;
}

const static MatrixXf gvecToDetectorXYArray(
    const MatrixXf& gVec,
    const Matrix3f& rMat_d, const MatrixXf& rMat_s, const Matrix3f& rMat_c,
    const Vector3f& tVec_d, const Vector3f& tVec_s, const Vector3f& tVec_c,
    const Vector3f& beamVec)
{
    MatrixXf result(gVec.rows(), 2);
    Vector3f bHat_l = beamVec.normalized();

    #pragma omp parallel for
    for (int i = 0; i < gVec.rows(); ++i) {
        // Replaced loop with matrix-vector operations
        Vector3f nVec_l = rMat_d * Zl;
        Vector3f P0_l = tVec_s + rMat_s.block<3, 3>(i * 3, 0) * tVec_c;

        result.row(i) = gvecToDetectorXYOne(gVec.row(i), rMat_d, rMat_s.block<3, 3>(i * 3, 0) * rMat_c,
                                            tVec_d, bHat_l, nVec_l, nVec_l.dot(tVec_d - P0_l), P0_l);
    }

    return result;
}

Vector3f rotateVecAboutAxis(const Vector3f& vec, const Vector3f& axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);

    // Normalize the axis
    Vector3f axis_n = axis.normalized();

    // Compute the cross product of the axis with vec
    Vector3f aCrossV = axis_n.cross(vec);

    // Compute projection of vec along axis
    float proj = axis_n.dot(vec);

    // Combine the three terms to compute the rotated vector
    Vector3f rVec = c * vec + s * aCrossV + (1.0 - c) * proj * axis_n;
    return rVec;
}

std::tuple<float, float, Vector3f> detectorXYToGVecOne(
    const Vector2f& xy,
    const Matrix3f& rMat_d, const Matrix3f& rMat_e,
    const Vector3f& tVec1, const Vector3f& bVec)
{
    Vector3f xy_transformed = rMat_d.leftCols(2) * xy;
    Vector3f dHat_l = tVec1;
    for (int i = 0; i < xy_transformed.size(); ++i)
        dHat_l[i] += xy_transformed[i];
    dHat_l.normalize();

    // Compute tTh
    float b_dot_dHat_l = bVec.dot(dHat_l);
    float tTh = acos(b_dot_dHat_l);

    // Compute eta
    Vector3f tVec2 = rMat_e.transpose() * dHat_l;
    float eta = atan2(tVec2[1], tVec2[0]);

    // Compute n_g vector
    Vector3f n_g = bVec.cross(dHat_l);
    n_g.normalize();

    // Rotate dHat_l vector
    float phi = 0.5*(M_PI - tTh);
    Vector3f gVec_l_out = rotateVecAboutAxis(dHat_l, n_g, phi);

    return std::make_tuple(tTh, eta, gVec_l_out);
}


MatrixXf detectorXYToGvec(
    const MatrixXf& xy,
    const Matrix3f& rMat_d, const Matrix3f& rMat_s,
    const Vector3f& tVec_d, const Vector3f& tVec_s, const Vector3f& tVec_c,
    const Vector3f& beamVec, const Vector3f& etaVec)
{
    Matrix3f rMat_e = makeEtaFrameRotMat(beamVec, etaVec);
    Vector3f bVec = beamVec.normalized();
    Vector3f tVec1 = tVec_d - tVec_s - rMat_s * tVec_c;

    MatrixXf result(3, xy.cols());

    for (long i = 0; i < xy.cols(); ++i) {
        float tTh;
        float eta;
        Vector3f gVec_l;

        std::tie(tTh, eta, gVec_l) = detectorXYToGVecOne(xy.col(i), rMat_d, rMat_e, tVec1, bVec);
        result.col(i) = gVec_l;
    }

    return result;
}

std::pair<MatrixXf, MatrixXf> oscillAnglesOfHKLs(
    const MatrixXf& hkls, 
    float chi,
    const Matrix3f& rMat_c, 
    const Matrix3f& bMat, 
    float wavelength,
    const Matrix3f& vInv_s, 
    const Vector3f& beamVec, 
    const Vector3f& etaVec
){
    long int npts = hkls.rows();

    Vector3f gHat_c, gHat_s, bHat_l, eHat_l, tVec0, tmpVec, gVec_e;
    Vector2f oVec;
    Matrix3f rMat_e, rMat_s;
    float a, b, c, sintht, cchi, schi;
    float abMag, phaseAng, rhs, rhsAng;
    float nrm0;

    // output matrices
    MatrixXf oangs0(npts, 3), oangs1(npts, 3);

    /* Normalize the beam vector */
    bHat_l = beamVec.normalized();

    /* Normalize the eta vector */
    eHat_l = etaVec.normalized();

    /* Check for consistent reference coordinates */
    bool crc = false;
    if (fabs(bHat_l.dot(eHat_l)) < 1.0 - sqrtEps) 
        crc = true;

    /* Compute the sine and cosine of the oscillation axis tilt */
    cchi = cos(chi);
    schi = sin(chi);

    for (long int i = 0; i < npts; i++) {

        /* Compute gVec_c */
        gHat_c = bMat * hkls.row(i).transpose();

        /* Apply rMat_c to get gVec_s */
        gHat_s = rMat_c * gHat_c;

        /* Apply vInv_s to gVec_s and store in tmpVec*/
        tmpVec[0] = vInv_s(0, 0)*gHat_s[0] + (vInv_s(2, 1)*gHat_s[1] + vInv_s(2, 0)*gHat_s[2])/sqrt(2.);
        tmpVec[1] = vInv_s(1, 1)*gHat_s[1] + (vInv_s(2, 1)*gHat_s[0] + vInv_s(1, 2)*gHat_s[2])/sqrt(2.);
        tmpVec[2] = vInv_s(2, 2)*gHat_s[2] + (vInv_s(2, 0)*gHat_s[0] + vInv_s(1, 2)*gHat_s[1])/sqrt(2.);

        /* Apply rMat_c.T to get stretched gVec_c and store norm in nrm0*/
        gHat_c = rMat_c.transpose() * tmpVec;
        nrm0 = gHat_c.norm();

        /* Normalize both gHat_c and gHat_s */
        gHat_c.normalize();
        gHat_s = tmpVec / nrm0;

        /* Compute the sine of the Bragg angle */
        sintht = 0.5 * wavelength * nrm0;

        /* Compute the coefficients of the harmonic equation */
        a = gHat_s[2]*bHat_l[0] + schi*gHat_s[0]*bHat_l[1] - cchi*gHat_s[0]*bHat_l[2];
        b = gHat_s[0]*bHat_l[0] - schi*gHat_s[2]*bHat_l[1] + cchi*gHat_s[2]*bHat_l[2];
        c = -sintht - cchi*gHat_s[1]*bHat_l[1] - schi*gHat_s[1]*bHat_l[2];

        /* Form solution */
        abMag = sqrt(a*a + b*b); 
        assert(abMag > 0.0);
        phaseAng = atan2(b, a);
        rhs = c / abMag;

        if (fabs(rhs) > 1.0) {
            oangs0.row(i).setConstant(std::nan(""));
            oangs1.row(i).setConstant(std::nan(""));
            continue;
        }

        rhsAng = asin(rhs);

        /* Write ome angles */
        oangs0(i, 2) = rhsAng - phaseAng;
        oangs1(i, 2) = M_PI - rhsAng - phaseAng;

        if (crc) {
            rMat_e = makeEtaFrameRotMat(bHat_l, eHat_l);

            oVec[0] = chi;

            oVec[1] = oangs0(i, 2);
            rMat_s = makeOscillRotMat_internal(oVec[0], oVec[1]);

            tVec0 = rMat_s * gHat_s;
            gVec_e = rMat_e.transpose() * tVec0;
            oangs0(i, 1) = atan2(gVec_e[1], gVec_e[0]);

            oVec[1] = oangs1(i, 2);
            rMat_s = makeOscillRotMat_internal(oVec[0], oVec[1]);

            tVec0 = rMat_s * gHat_s;
            gVec_e = rMat_e.transpose() * tVec0;
            oangs1(i, 1) = atan2(gVec_e[1], gVec_e[0]);

            oangs0(i, 0) = 2.0 * asin(sintht);
            oangs1(i, 0) = oangs0(i, 0);
        }
    }

    return {oangs0, oangs1};
}


PYBIND11_MODULE(example, m)
{
  m.doc() = "pybind11 hexrd plugin";

  m.def("quat_distance", &quat_distance, "Function to calculate the distance between two quaternions");
  m.def("rotate_vecs_about_axis", &rotate_vecs_about_axis, "Function to rotate vectors about axes");
  m.def("validate_angle_ranges", &validateAngleRanges, "Function to validate angle ranges");
  m.def("make_eta_frame_rot_mat", &makeEtaFrameRotMat, "Function to compute rotation matrix");
  m.def("make_binary_rot_mat", &makeBinaryRotMat, "Function that computes a rotation matrix from a binary vector");
  m.def("make_rot_mat_of_exp_map", &makeRotMatOfExpMap, "Function that computes a rotation matrix from an exponential map");
  m.def("makeOscillRotMat", &makeOscillRotMat, "Function that generates a collection of rotation matrices from two angles (chi, ome)");
  m.def("makeOscillRotMatSingle", &makeOscillRotMatSingle, "Function that generates a rotation matrix from two angles (chi, ome)");
  m.def("unit_row_vectors", &unitRowVectors, "Function that normalizes row vectors");
  m.def("unit_row_vector", &unitRowVector, "Function that normalizes a row vector");
  m.def("anglesToGVec", &anglesToGvec, "Function that converts angles to g-vectors");
  m.def("angles_to_dvec", &anglesToDvec, "Function that converts angles to d-vectors");
  m.def("gvec_to_detector_xy_one", &gvecToDetectorXYOne, "Function that converts g-vectors to detector xy coordinates");
  m.def("gvecToDetectorXY", &gvecToDetectorXY, "A function that converts gVec to detector XY");
  m.def("gvecToDetectorXYArray", &gvecToDetectorXYArray, "Function that converts g-vectors to detector xy coordinates");
  m.def("detector_xy_to_gvec", &detectorXYToGvec, "Function that converts detector xy coordinates to g-vectors");
  m.def("detector_xy_to_gvec_one", &detectorXYToGVecOne, "Function that converts detector xy coordinates to g-vectors");
  m.def("oscill_angles_of_hkls", &oscillAnglesOfHKLs, "Function that computes oscillation angles of HKLs");
}