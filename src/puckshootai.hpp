#ifndef PUCKETSHOOTAI_HPP
#define PUCKETSHOOTAI_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <Godot.hpp>
#include <Node.hpp>
#include <KinematicBody2D.hpp>
#include <RigidBody2D.hpp>
#pragma GCC diagnostic pop

#include <armadillo>

namespace godot
{

class PuckShootAI : public Node
{
  GODOT_CLASS(PuckShootAI, Node)

public:
  static const int16_t SPEED = 10;
  static const int16_t HANDICAP = 30;

  static void _register_methods();

  static void sigmoid(arma::mat *g, const arma::mat &z);
  static void normalize(arma::mat *normX, arma::rowvec *mu, arma::rowvec *sigma, const arma::mat &matX);

  PuckShootAI();
  ~PuckShootAI();

  void _init();

  void _process(float delta);

  uint16_t handicap();
  void set_handicap(const uint16_t handicap);
  uint16_t difficulty();
  void set_difficulty(const uint16_t difficulty);
  void set_window_size(const uint16_t width, const uint16_t height);

  int64_t get_y_left();  // triggers AI
  int64_t get_y_right(); // triggers AI

  double predict_direction(const arma::rowvec &mu, const arma::rowvec &sigma, const arma::colvec &theta, const int64_t refl_y);
  void train(arma::rowvec *mu, arma::rowvec *sigma, arma::colvec *theta);

protected:
  uint16_t _difficulty, speediculty, _handicap;
  int64_t hits_left = 0, hits_right = 0;
  Vector2 puck_position, puck_velocity, window_size;
  int64_t refl_left_y, refl_right_y;
  bool refl_left_dir = false, refl_right_dir = false;
  arma::rowvec logreg_mu, logreg_sigma;
  arma::colvec logreg_theta;
};

} // namespace godot

#endif
