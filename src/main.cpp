#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <xsimd/xsimd.hpp>

using namespace Eigen;

namespace py = pybind11;

const Vector3d Zl = {0.0, 0.0, 1.0};
const double eps = std::numeric_limits<double>::epsilon();
const double sqrtEps = std::sqrt(std::numeric_limits<double>::epsilon());

double quat_distance(Vector4d q1, Vector4d q2, MatrixX4d qsym) {
  double q0_max = 0.;
  for (int i = 0; i < qsym.rows(); i++) {
    const Vector4d qs = qsym.row(i);
    Vector4d q2s(
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

MatrixXd rotate_vecs_about_axis(VectorXd angles, MatrixXd axes, MatrixXd vecs)
{
  MatrixXd rVecs(vecs.rows(), vecs.cols());

  // Normalize axes, as apparently required for Eigen's AngleAxisd
  axes.rowwise().normalize();

  for (int i = 0; i < angles.size(); ++i) 
  {
    AngleAxisd rotation_vector(angles(i), axes.row(i));
    Vector3d vec3d = vecs.row(i);
    rVecs.row(i) = (rotation_vector * vec3d).transpose();
  }

  return rVecs;
}

Array<bool, Dynamic, 1> validateAngleRanges(const VectorXd& angList, const VectorXd& angMin, const VectorXd& angMax, bool ccw)
{
    const double twoPi = 2.0 * M_PI;

    VectorXd startPtr = ccw ? angMin : angMax;
    VectorXd stopPtr  = ccw ? angMax : angMin;

    Array<bool, Dynamic, 1> result(angList.size());
    result.fill(false);

    for (int i = 0; i < angList.size(); i++) {
        for (int j = 0; j < startPtr.size(); j++) {
            double thetaMax = std::fmod(stopPtr[j] - startPtr[j], twoPi);
            double theta = std::fmod(angList[i] - startPtr[j], twoPi);

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

const static Matrix3d makeEtaFrameRotMat(const Vector3d& b, const Vector3d& e) noexcept {
  const Vector3d yHat = e.cross(b).normalized();
  const Vector3d bHat = b.normalized();
  const Vector3d xHat = bHat.cross(yHat);
  Matrix3d r;
  r << xHat, yHat, -bHat;

  return r;
}

const static MatrixXd makeBinaryRotMat(const Vector3d& a) {
    Matrix3d r = 2.0 * a * a.transpose();
    r.diagonal() -= Vector3d::Ones();
    return r;
}

MatrixXd makeRotMatOfExpMap(const Vector3d& e) {
    AngleAxisd rotation(e.norm(), e.normalized());
    return rotation.toRotationMatrix();
}

const static MatrixXd makeOscillRotMat_internal(double chi, double ome) {
  Matrix3d r;
  r = AngleAxisd(chi, Vector3d::UnitX())
    * AngleAxisd(ome, Vector3d::UnitY());

  return r;
}

static MatrixX3d makeOscillRotMat(double chi, const VectorXd& ome) {
  MatrixX3d rots(3 * ome.size(), 3);
  const double cchi = cos(chi), schi = sin(chi);
  xsimd::batch<double, 2UL> come_v, some_v;

  const size_t batch_size = xsimd::simd_traits<double>::size;
  const size_t vec_size = ome.size();

  for (size_t i = 0; i < vec_size; i += batch_size) {
    auto ome_v = xsimd::load_unaligned<double>(&ome[i]);
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

const static MatrixXd makeOscillRotMatSingle(double chi, double ome) {
  return makeOscillRotMat_internal(chi, ome);
}

const static MatrixXd unitRowVectors(const MatrixXd& cIn) {
  return cIn.rowwise().normalized();
}

const static VectorXd unitRowVector(const VectorXd& cIn) {
  return cIn.normalized();
}

////////////////////////////////////////

MatrixXd anglesToGvecModified(const MatrixXd& angs, const Vector3d& bHat_l, const Vector3d& eHat_l, double chi, const Matrix3d& rMat_c) noexcept {
    MatrixXd angs_mod;
    if (angs.cols() == 2) {
        angs_mod = MatrixXd::Zero(angs.rows(), 3);
        angs_mod.block(0, 0, angs.rows(), 2) = angs;
    } else {
        angs_mod = angs;
    }
    const VectorXd cosHalfAngs = (0.5 * angs.col(0)).array().cos();
    const VectorXd sinHalfAngs = (0.5 * angs.col(0)).array().sin();
    const VectorXd cosAngs1 = angs.col(1).array().cos();
    const VectorXd sinAngs1 = angs.col(1).array().sin();
    MatrixXd gVec_l(angs.rows(), 3);
    gVec_l << cosHalfAngs.cwiseProduct(cosAngs1), cosHalfAngs.cwiseProduct(sinAngs1), sinHalfAngs;

    // Transform gVec_l to lab frame
    gVec_l = (makeEtaFrameRotMat(bHat_l, eHat_l) * gVec_l.transpose());
    MatrixXd gVec_c(angs.rows(), 3);

    const Matrix3d rMat_c_T = rMat_c.transpose();
    const Matrix3d A = AngleAxisd(chi, Vector3d::UnitX()).matrix();

    for (int i = 0; i < angs.rows(); i++) {
        gVec_c.row(i).noalias() = rMat_c_T * (A * AngleAxisd(angs(i, 2), Vector3d::UnitY())).matrix().transpose() * gVec_l.col(i);
    }

    return gVec_c;
}

MatrixXd anglesToGvecModified_atomic(const MatrixXd& angs, double chi, const Matrix3d& rMat_c) noexcept {
    Vector3d bHat_l(0, 0, -1); // default for beam_vec
    Vector3d eHat_l(1, 0, 0); // default for eta_vec
    return anglesToGvecModified(angs, bHat_l, eHat_l, chi, rMat_c);
}

////////////////////////////////////////

MatrixXd anglesToGvec(const MatrixXd& angs, const Vector3d& bHat_l, const Vector3d& eHat_l, double chi, const Matrix3d& rMat_c) noexcept {
    const VectorXd cosHalfAngs = (0.5 * angs.col(0)).array().cos();
    const VectorXd sinHalfAngs = (0.5 * angs.col(0)).array().sin();
    const VectorXd cosAngs1 = angs.col(1).array().cos();
    const VectorXd sinAngs1 = angs.col(1).array().sin();
    MatrixXd gVec_l(angs.rows(), 3);
    gVec_l << cosHalfAngs.cwiseProduct(cosAngs1), cosHalfAngs.cwiseProduct(sinAngs1), sinHalfAngs;

    // Transform gVec_l to lab frame
    gVec_l = (makeEtaFrameRotMat(bHat_l, eHat_l) * gVec_l.transpose());
    MatrixXd gVec_c(angs.rows(), 3);

    const Matrix3d rMat_c_T = rMat_c.transpose();
    const Matrix3d A = AngleAxisd(chi, Vector3d::UnitX()).matrix();

    for (int i = 0; i < angs.rows(); i++) {
        gVec_c.row(i).noalias() = rMat_c_T * (A * AngleAxisd(angs(i, 2), Vector3d::UnitY())).matrix().transpose() * gVec_l.col(i);
    }

    return gVec_c;
}

MatrixXd anglesToDvec(MatrixXd& angs, Vector3d& bHat_l, Vector3d& eHat_l, double chi, Matrix3d& rMat_c) {
    // Construct matrix where each row represents a gVec_l calculated for each angle
    MatrixXd sinAngs0 = angs.col(0).array().sin();
    MatrixXd cosAngs0 = angs.col(0).array().cos();
    MatrixXd cosAngs1 = angs.col(1).array().cos();
    MatrixXd sinAngs1 = angs.col(1).array().sin();
    MatrixXd gVec_l(angs.rows(), 3);
    gVec_l << sinAngs0.cwiseProduct(cosAngs1), sinAngs0.cwiseProduct(sinAngs1), -cosAngs0;

    // Make eta frame cob matrix and transform gVec_l to lab frame
    Matrix3d rMat_e = makeEtaFrameRotMat(bHat_l, eHat_l);
    gVec_l = (rMat_e * gVec_l.transpose()).transpose();

    // Calculate rotation matrices for each pair of angles chi and ome
    std::vector<Matrix3d> rotation_matrices;
    for (int i = 0; i < angs.rows(); i++) {
        rotation_matrices.push_back(makeOscillRotMat_internal(chi, angs(i, 2)));
    }

    // Multiply rotation matrices by rMat_c
    std::vector<Matrix3d> result_matrices;
    for (auto &rotMat_s : rotation_matrices) {
        result_matrices.push_back(rMat_c.transpose() * rotMat_s.transpose());
    }

    // Compute the dot product of result_matrices and gVec_l
    MatrixXd gVec_c(angs.rows(), 3);
    for (int i = 0; i < angs.rows(); i++) {
        gVec_c.row(i) = (result_matrices[i] * gVec_l.row(i).transpose()).transpose();
    }

    return gVec_c;
}

const static Vector2d gvecToDetectorXYOne(const Vector3d& gVec_c, const Matrix3d& rMat_d,
                                    const Matrix3d& rMat_sc, const Vector3d& tVec_d,
                                    const Vector3d& bHat_l, const Vector3d& nVec_l,
                                    double num, const Vector3d& P0_l)
{
  Vector3d gHat_c = gVec_c.normalized();
  Vector3d gVec_l = rMat_sc * gHat_c;
  double bDot = -bHat_l.dot(gVec_l);

  if ( bDot >= eps && bDot <= 1.0 - eps ) {
    Matrix3d brMat = makeBinaryRotMat(gVec_l);

    Vector3d dVec_l = -brMat * bHat_l;
    double denom = nVec_l.dot(dVec_l);

    if ( denom < -eps ) {
      Vector3d P2_l = P0_l + dVec_l * num / denom;
      Vector2d result;
      result[0] = (rMat_d.col(0).dot(P2_l - tVec_d));
      result[1] = (rMat_d.col(1).dot(P2_l - tVec_d));

      return result;
    }
  }

  return Vector2d(NAN, NAN);
}

const static MatrixXd gvecToDetectorXY(const MatrixXd& gVec_c, const Matrix3d& rMat_d,
                                 const MatrixXd& rMat_s, const Matrix3d& rMat_c,
                                 const Vector3d& tVec_d, const Vector3d& tVec_s,
                                 const Vector3d& tVec_c, const Vector3d& beamVec)
{
    int npts = gVec_c.rows();
    MatrixXd result(npts, 2);

    Vector3d bHat_l = beamVec.normalized();

    for (int i=0; i<npts; i++) {
        Vector3d nVec_l = rMat_d * Zl;
        Vector3d P0_l = tVec_s + rMat_s.block<3,3>(i*3, 0) * tVec_c;
        Vector3d P3_l = tVec_d;

        double num = nVec_l.dot(P3_l - P0_l);

        Matrix3d rMat_sc = rMat_s.block<3,3>(i*3, 0) * rMat_c;

        result.row(i) = gvecToDetectorXYOne(gVec_c.row(i), rMat_d, rMat_sc, tVec_d,
                                            bHat_l, nVec_l, num, P0_l).transpose();
    }

    return result;
}

const static MatrixXd gvecToDetectorXYArray(
    const MatrixXd& gVec,
    const Matrix3d& rMat_d, const MatrixXd& rMat_s, const Matrix3d& rMat_c,
    const Vector3d& tVec_d, const Vector3d& tVec_s, const Vector3d& tVec_c,
    const Vector3d& beamVec)
{
    MatrixXd result(gVec.rows(), 2);
    Vector3d bHat_l = beamVec.normalized();

    #pragma omp parallel for
    for (int i = 0; i < gVec.rows(); ++i) {
        // Replaced loop with matrix-vector operations
        Vector3d nVec_l = rMat_d * Zl;
        Vector3d P0_l = tVec_s + rMat_s.block<3, 3>(i * 3, 0) * tVec_c;

        result.row(i) = gvecToDetectorXYOne(gVec.row(i), rMat_d, rMat_s.block<3, 3>(i * 3, 0) * rMat_c,
                                            tVec_d, bHat_l, nVec_l, nVec_l.dot(tVec_d - P0_l), P0_l);
    }

    return result;
}

Vector3d rotateVecAboutAxis(const Vector3d& vec, const Vector3d& axis, double angle) {
    double c = cos(angle);
    double s = sin(angle);

    // Normalize the axis
    Vector3d axis_n = axis.normalized();

    // Compute the cross product of the axis with vec
    Vector3d aCrossV = axis_n.cross(vec);

    // Compute projection of vec along axis
    double proj = axis_n.dot(vec);

    // Combine the three terms to compute the rotated vector
    Vector3d rVec = c * vec + s * aCrossV + (1.0 - c) * proj * axis_n;
    return rVec;
}

std::tuple<double, double, Vector3d> detectorXYToGVecOne(
    const Vector2d& xy,
    const Matrix3d& rMat_d, const Matrix3d& rMat_e,
    const Vector3d& tVec1, const Vector3d& bVec)
{
    Vector3d xy_transformed = rMat_d.leftCols(2) * xy;
    Vector3d dHat_l = tVec1;
    for (int i = 0; i < xy_transformed.size(); ++i)
        dHat_l[i] += xy_transformed[i];
    dHat_l.normalize();

    // Compute tTh
    double b_dot_dHat_l = bVec.dot(dHat_l);
    double tTh = acos(b_dot_dHat_l);

    // Compute eta
    Vector3d tVec2 = rMat_e.transpose() * dHat_l;
    double eta = atan2(tVec2[1], tVec2[0]);

    // Compute n_g vector
    Vector3d n_g = bVec.cross(dHat_l);
    n_g.normalize();

    // Rotate dHat_l vector
    double phi = 0.5*(M_PI - tTh);
    Vector3d gVec_l_out = rotateVecAboutAxis(dHat_l, n_g, phi);

    return std::make_tuple(tTh, eta, gVec_l_out);
}


MatrixXd detectorXYToGvec(
    const MatrixXd& xy,
    const Matrix3d& rMat_d, const Matrix3d& rMat_s,
    const Vector3d& tVec_d, const Vector3d& tVec_s, const Vector3d& tVec_c,
    const Vector3d& beamVec, const Vector3d& etaVec)
{
    Matrix3d rMat_e = makeEtaFrameRotMat(beamVec, etaVec);
    Vector3d bVec = beamVec.normalized();
    Vector3d tVec1 = tVec_d - tVec_s - rMat_s * tVec_c;

    MatrixXd result(3, xy.cols());

    for (long i = 0; i < xy.cols(); ++i) {
        double tTh;
        double eta;
        Vector3d gVec_l;

        std::tie(tTh, eta, gVec_l) = detectorXYToGVecOne(xy.col(i), rMat_d, rMat_e, tVec1, bVec);
        result.col(i) = gVec_l;
    }

    return result;
}

std::pair<MatrixXd, MatrixXd> oscillAnglesOfHKLs(
    const MatrixXd& hkls, 
    double chi,
    const Matrix3d& rMat_c, 
    const Matrix3d& bMat, 
    double wavelength,
    const Matrix3d& vInv_s, 
    const Vector3d& beamVec, 
    const Vector3d& etaVec
){
    long int npts = hkls.rows();

    Vector3d gHat_c, gHat_s, bHat_l, eHat_l, tVec0, tmpVec, gVec_e;
    Vector2d oVec;
    Matrix3d rMat_e, rMat_s;
    double a, b, c, sintht, cchi, schi;
    double abMag, phaseAng, rhs, rhsAng;
    double nrm0;

    // output matrices
    MatrixXd oangs0(npts, 3), oangs1(npts, 3);

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
  m.def("anglesToGVecModified", &anglesToGvecModified, "Function that converts angles to g-vectors with additional checks");
  m.def("anglesToGVecModified_atomic", &anglesToGvecModified_atomic, "Function that converts angles to g-vectors with additional checks and defaults for eta_vec and beam_vec");
}