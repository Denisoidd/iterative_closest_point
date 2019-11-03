#include "ICP.h"
#include <iostream>
#include <ostream>

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2){
  std::vector<std::vector<int > > O_PI;
  Eigen::MatrixXi O_CH;
  Eigen::MatrixXd O_CN;
  Eigen::VectorXd O_W;
  igl::octree(V2,O_PI,O_CH,O_CN,O_W);
  Eigen::MatrixXi I;

  igl::knn(V1, 2, O_PI, O_CH,O_CN, O_W, I);
  for (int i = 0; i < V1.rows(); i++){
      nn_V2.row(i) = V2.row(I(i, 1));
    }
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
  std::cout << "SVD V" << '\n';
  std::cout << svd.matrixV() << '\n';
  std::cout << "SVD U" << '\n';
  std::cout << svd.matrixU() << '\n';
  std::cout << "Rotation" << '\n';
  std::cout << rot << '\n';
  // translation
  VectorXd transl = d - rot * m;
  std::cout << "Translation" << '\n';
  std::cout << transl << '\n';

  // change V1
  V1 = ((rot * V1.transpose()).colwise() + transl).transpose();
}
