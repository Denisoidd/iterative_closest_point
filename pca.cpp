#include "pca.h"
#include <iostream>
#include <ostream>
#include <math.h>

void k_nearest_neighbour(const MatrixXd &V1,Eigen::MatrixXi &I, int k){
  // need to get k nearest_neighbour indexes
  // it will be in variable I
  std::vector<std::vector<int > > O_PI;
  Eigen::MatrixXi O_CH;
  Eigen::MatrixXd O_CN;
  Eigen::VectorXd O_W;

  igl::octree(V1,O_PI,O_CH,O_CN,O_W);
  igl::knn(V1, k, O_PI, O_CH,O_CN, O_W, I);
}



void compute_normals(const MatrixXd &V1,const Eigen::MatrixXi &I, int k, MatrixXd &normals){
    std::cout << "V1" << '\n';
    std::cout << V1 << '\n';
    // compute the normals using PCA
    for (int j = 0; j < V1.rows(); j++){
      // local variable for neighbours coordinates
      MatrixXd neighb_points = MatrixXd::Zero(3,k);
      // points in neighbourhood
      for (int i = 0; i < k; i++){
        neighb_points.col(i) = V1.transpose().col(I(j,i));
      }
      //find a mean for neighb_points (centroid)
      VectorXd m = neighb_points.rowwise().mean();

      //substruct mean from all values
      MatrixXd delta_m = neighb_points;
      delta_m.colwise() -= m;

      //scatter matrix S
      MatrixXd S = delta_m * delta_m.transpose();

      //SVD
      JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);

      //Getting plane normal (the last column in U)
      int last_index = svd.matrixU().cols();
      normals.row(j) = svd.matrixU().transpose().row(last_index - 1);

      //Normalisation
      double norm = sqrt(std::pow(normals(j,0),2) + std::pow(normals(j,1),2) + std::pow(normals(j,2),2));
      for (int i = 0; i < 3; i++){
        normals(j,i) /= norm;
      }
    }
    //alignment
    for (int i = 0; i < V1.rows(); i++){
      MatrixXd norm1 = normals.row(i);
      for (int t = 1; t < k; t++){

        MatrixXd norm2 = normals.row(I(i, t));
        double scalar_product = norm1(0,0) * norm2(0,0) + norm1(0,1) * norm2(0,1) + norm1(0,2) * norm2(0,2);
        std::cout << "scalar_product is " << scalar_product << '\n';
        if (scalar_product < 0.0d){
          std::cout << "Normal before" << '\n';
          std::cout << norm2 << '\n';
          for (int j = 0; j < 3; j++){
            std::cout << "What we will change?" << '\n';
            std::cout << normals(I(i, t),j) << '\n';
            normals(I(i, t),j) = -norm2(0,j);
          }
          std::cout << "Normal after" << '\n';
          std::cout << normals.row(I(i, t)) << '\n';
        }
      }
    }
  }
