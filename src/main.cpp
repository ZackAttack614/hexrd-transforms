#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <cmath>
#include <iostream>

using namespace Eigen;

namespace py = pybind11;

const Eigen::Vector3d Zl = {0.0, 0.0, 1.0};
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

Matrix3d makeEtaFrameRotMat(const Vector3d& b, const Vector3d& e) {
    Vector3d yHat = e.cross(b).normalized();
    Vector3d bHat = b.normalized();
    Vector3d xHat = bHat.cross(yHat);
    Matrix3d r;
    r << xHat, yHat, -bHat;

    return r;
}

MatrixXd makeBinaryRotMat(const Vector3d& a) {
    Matrix3d r = 2.0 * a * a.transpose();
    r.diagonal() -= Vector3d::Ones();
    return r;
}

MatrixXd makeRotMatOfExpMap(const Vector3d& e) {
    AngleAxisd rotation(e.norm(), e.normalized());
    return rotation.toRotationMatrix();
}

MatrixXd makeOscillRotMat_internal(double chi, double ome) {
  Matrix3d r;
  r = AngleAxisd(chi, Vector3d::UnitX())
    * AngleAxisd(ome, Vector3d::UnitY());

  return r;
}

MatrixXd makeOscillRotMat(double chi, VectorXd ome) {
  MatrixXd rots(3 * ome.size(), 3);
  for (int i = 0; i < ome.size(); ++i) {
    rots.block<3,3>(3*i,0) = makeOscillRotMat_internal(chi, ome(i));
  }
  return rots;
}

MatrixXd unitRowVectors(const MatrixXd& cIn) {
  return cIn.rowwise().normalized();
}

VectorXd unitRowVector(const VectorXd& cIn) {
  return cIn.normalized();
}

MatrixXd anglesToGvec(MatrixXd& angs, Vector3d& bHat_l, Vector3d& eHat_l, double chi, Matrix3d& rMat_c) {
    // Construct matrix where each row represents a gVec_l calculated for each angle
    MatrixXd cosHalfAngs = (0.5 * angs.col(0)).array().cos();
    MatrixXd sinHalfAngs = (0.5 * angs.col(0)).array().sin();
    MatrixXd cosAngs1 = angs.col(1).array().cos();
    MatrixXd sinAngs1 = angs.col(1).array().sin();
    MatrixXd gVec_l(angs.rows(), 3);
    gVec_l << cosHalfAngs.cwiseProduct(cosAngs1), cosHalfAngs.cwiseProduct(sinAngs1), sinHalfAngs;

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

Eigen::Vector2d gvecToDetectorXYOne(const Eigen::Vector3d& gVec_c, const Eigen::Matrix3d& rMat_d,
                                    const Eigen::Matrix3d& rMat_sc, const Eigen::Vector3d& tVec_d,
                                    const Eigen::Vector3d& bHat_l, const Eigen::Vector3d& nVec_l,
                                    double num, const Eigen::Vector3d& P0_l)
{
  Eigen::Vector3d gHat_c = gVec_c.normalized();
  Eigen::Vector3d gVec_l = rMat_sc * gHat_c;
  double bDot = -bHat_l.dot(gVec_l);

  if ( bDot >= eps && bDot <= 1.0 - eps ) {
    Eigen::Matrix3d brMat = makeBinaryRotMat(gVec_l);

    Eigen::Vector3d dVec_l = -brMat * bHat_l;
    double denom = nVec_l.dot(dVec_l);

    if ( denom < -eps ) {
      Eigen::Vector3d P2_l = P0_l + dVec_l * num / denom;
      Eigen::Vector2d result;
      result[0] = (rMat_d.col(0).dot(P2_l - tVec_d));
      result[1] = (rMat_d.col(1).dot(P2_l - tVec_d));

      return result;
    }
  }

  return Eigen::Vector2d(NAN, NAN);
}

Eigen::MatrixXd gvecToDetectorXY(const Eigen::MatrixXd& gVec_c, const Eigen::Matrix3d& rMat_d,
                                 const Eigen::MatrixXd& rMat_s, const Eigen::Matrix3d& rMat_c,
                                 const Eigen::Vector3d& tVec_d, const Eigen::Vector3d& tVec_s,
                                 const Eigen::Vector3d& tVec_c, const Eigen::Vector3d& beamVec)
{
    int npts = gVec_c.rows();
    Eigen::MatrixXd result(npts, 2);

    Eigen::Vector3d bHat_l = beamVec.normalized();

    for (int i=0; i<npts; i++) {
        Eigen::Vector3d nVec_l = rMat_d * Zl;
        Eigen::Vector3d P0_l = tVec_s + rMat_s.block<3,3>(i*3, 0) * tVec_c;
        Eigen::Vector3d P3_l = tVec_d;

        double num = nVec_l.dot(P3_l - P0_l);

        Eigen::Matrix3d rMat_sc = rMat_s.block<3,3>(i*3, 0) * rMat_c;

        result.row(i) = gvecToDetectorXYOne(gVec_c.row(i), rMat_d, rMat_sc, tVec_d,
                                            bHat_l, nVec_l, num, P0_l).transpose();
    }

    return result;
}

Eigen::MatrixXd gvecToDetectorXYArray(
    const Eigen::MatrixXd& gVec,
    const Eigen::Matrix3d& rMat_d, const Eigen::MatrixXd& rMat_s, const Eigen::Matrix3d& rMat_c,
    const Eigen::Vector3d& tVec_d, const Eigen::Vector3d& tVec_s, const Eigen::Vector3d& tVec_c,
    const Eigen::Vector3d& beamVec)
{
    Eigen::MatrixXd result(gVec.rows(), 2);
    Eigen::Vector3d Zl(0, 0, 1);
    Eigen::Vector3d bHat_l = beamVec.normalized();

    for (int i = 0; i < gVec.rows(); ++i) {
        Eigen::Vector3d nVec_l = Eigen::Vector3d::Zero();
        Eigen::Vector3d P0_l = tVec_s;
        for (int j = 0; j < 3; ++j) {
            nVec_l += rMat_d.col(j) * Zl[j];
            P0_l += rMat_s.block<3, 3>(i * 3, 0).col(j) * tVec_c[j];
        }

        Eigen::Vector3d P3_l = tVec_d;
        double num = nVec_l.dot(P3_l - P0_l);

        Eigen::Matrix3d rMat_sc = rMat_s.block<3, 3>(i * 3, 0) * rMat_c;

        result.row(i) = gvecToDetectorXYOne(gVec.row(i), rMat_d, rMat_sc, tVec_d, bHat_l, nVec_l, num, P0_l);
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
  m.def("makeOscillRotMat", &makeOscillRotMat, "Function that generates a rotation matrix from two angles (chi, ome)");
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