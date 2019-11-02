#include "ICP.h"

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2){

}


void transform(MatrixXd &V1,const MatrixXd &V2){
  // find a mean value for two sets of points
  VectorXd d = V2.colwise().mean().transpose();
  VectorXd m = V1.colwise().mean().transpose();

  // calculation deltas
  MatrixXd delta_d = V2;
  MatrixXd delta_m = V1;

  delta_d.rowwise() -= d.transpose();
  delta_m.rowwise() -= m.transpose();

  // getting a correlation matrix
  MatrixXd corr = delta_m.transpose() * delta_d;

  // SVD and rotation
  JacobiSVD<MatrixXd> svd(corr, ComputeThinU | ComputeThinV);
  MatrixXd rot = svd.matrixV() * svd.matrixU().transpose();

  // translation
  VectorXd transl = d - rot * m;

  // change V1
  V1 = ((rot * V2.transpose()).colwise() + transl).transpose();
}
