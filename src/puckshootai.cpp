#include <cassert>
#include "puckshootai.hpp"

using namespace arma;
using namespace godot;

void PuckShootAI::_register_methods()
{
    register_method("_process", &PuckShootAI::_process);
    register_method("difficulty", &PuckShootAI::difficulty);
    register_method("set_difficulty", &PuckShootAI::set_difficulty);
    register_method("handicap", &PuckShootAI::handicap);
    register_method("set_handicap", &PuckShootAI::set_handicap);
    register_method("set_window_size", &PuckShootAI::set_window_size);
    register_method("get_y_left", &PuckShootAI::get_y_left);
    register_method("get_y_right", &PuckShootAI::get_y_right);

    register_property<PuckShootAI, int64_t>("hits_left", &PuckShootAI::hits_left, 0);
    register_property<PuckShootAI, int64_t>("hits_right", &PuckShootAI::hits_right, 0);
    register_property<PuckShootAI, Vector2>("puck_position", &PuckShootAI::puck_position, Vector2());
    register_property<PuckShootAI, Vector2>("puck_velocity", &PuckShootAI::puck_velocity, Vector2());
    register_property<PuckShootAI, Vector2>("window_size", &PuckShootAI::window_size, Vector2());
    register_property<PuckShootAI, int64_t>("reflector_left_y", &PuckShootAI::refl_left_y, 0);
    register_property<PuckShootAI, int64_t>("reflector_right_y", &PuckShootAI::refl_right_y, 0);

    // register_property<PuckShootAI, RigidBody2D>("puck", &PuckShootAI::set_puck, &PuckShootAI::get_puck, new RigidBody2D());
    // register_property<PuckShootAI, KinematicBody2D>("refl_left", &PuckShootAI::set_reflector_left, &PuckShootAI::get_reflector_left, new KinematicBody2D());
    // register_method("torch_rand", &PuckShootAI::torch_rand);
}

const int16_t PuckShootAI::SPEED;
const int16_t PuckShootAI::HANDICAP;

PuckShootAI::PuckShootAI()
{
    train(&logreg_mu, &logreg_sigma, &logreg_theta);
}

PuckShootAI::~PuckShootAI() {}

void PuckShootAI::_init() {}

void PuckShootAI::_process(float delta) {}

uint16_t PuckShootAI::difficulty()
{
    return this->_difficulty;
}

void PuckShootAI::set_difficulty(const uint16_t difficulty)
{
    this->_difficulty = difficulty;
    this->speediculty = difficulty / handicap();
}

uint16_t PuckShootAI::handicap()
{
    return this->_handicap > 0 ? this->_handicap : HANDICAP;
}

void PuckShootAI::set_handicap(const uint16_t handicap)
{
    this->_handicap = handicap;
    this->speediculty = _difficulty / _handicap;
}

void PuckShootAI::set_window_size(const uint16_t width, const uint16_t height)
{
    this->window_size = Vector2(width, height);
}

int64_t PuckShootAI::get_y_left()
{
    const auto *v = &puck_velocity;
    int64_t y = refl_left_y;
    if (v->x >= 0 && y < speediculty && y > -speediculty)
    {
        return 0;
    }
    const auto direction = predict_direction(logreg_mu, logreg_sigma, logreg_theta, y);
    if (direction == 1.0)
    {
        y = -speediculty;
        refl_left_dir = true;
    }
    else
    {
        y = speediculty;
        refl_left_dir = false;
    }

    return y;
}

int64_t PuckShootAI::get_y_right()
{
    const auto *v = &puck_velocity;
    int64_t y = refl_right_y;
    if (v->x <= 0 && y < speediculty && y > -speediculty)
    {
        return 0;
    }
    const auto direction = predict_direction(logreg_mu, logreg_sigma, logreg_theta, y);
    if (direction == 1.0)
    {
        y = -speediculty;
        refl_right_dir = true;
    }
    else
    {
        y = speediculty;
        refl_right_dir = false;
    }

    return y;
}

void PuckShootAI::sigmoid(mat *g, const mat &z)
{
    *g = 1 / (1 + exp(-z));
}

void PuckShootAI::normalize(mat *matNormX, rowvec *mu, rowvec *sigma, const mat &matX)
{
    *mu = mean(matX);
    *matNormX = matX;
    matNormX->each_row() -= *mu;
    *sigma = stddev(matX);
    matNormX->each_row() /= *sigma;
}

double PuckShootAI::predict_direction(const rowvec &mu, const rowvec &sigma, const colvec &theta, const int64_t refl_y)
{
    const auto wsize_half_y = window_size.y / 2.0;
    const double ws_y_step = 0.5 / wsize_half_y;
    const auto puck_abs_y = ws_y_step * (wsize_half_y - puck_position.y);
    const auto refl_abs_y = ws_y_step * (wsize_half_y - refl_y);

    const rowvec sample = {1, (0.5 - refl_abs_y + puck_abs_y - mu(0)) / sigma(0)}; // don't need vectorized
    const rowvec predict = sample * theta;
    colvec predict_g;
    PuckShootAI::sigmoid(&predict_g, predict);

    // Godot::print(String("puck: {0} - refl: {1} - {2}")
    //                  .format(Array::make(
    //                      String(std::to_string(puck_abs_y).c_str()),
    //                      String(std::to_string(refl_abs_y).c_str()),
    //                      String(std::to_string(predict_g(0)).c_str()))));

    return (predict_g(0) < 0.5 ? 0.0 : 1.0);
}

void PuckShootAI::train(rowvec *mu, rowvec *sigma, colvec *theta)
{
    const mat matX = join_vert(linspace<colvec>(0.0, 0.499999, 250),
                               join_vert(linspace<colvec>(0.5, 1.0, 250),
                                         join_vert(linspace<colvec>(0.40, 0.499999, 250),
                                                   linspace<colvec>(0.50, 0.599999, 250))));
    assert(matX.n_cols == 1);

    const colvec vec_y = join_vert(zeros(250, 1),
                                   join_vert(ones(250, 1),
                                             join_vert(zeros(250, 1),
                                                       ones(250, 1))));
    assert(vec_y.n_cols == 1);

    const auto m = vec_y.n_elem;

    mat normX;
    PuckShootAI::normalize(&normX, mu, sigma, matX);

    const mat matExtX = join_horiz(ones(m, 1), normX);
    assert(matExtX.n_cols == 2);

    const auto alpha = 0.9;
    const auto num_iters = 300;

    *theta = zeros(2, 1);

    for (int i = 0; i < num_iters; i++)
    {
        colvec vec_h;
        PuckShootAI::sigmoid(&vec_h, (matExtX * (*theta)));
        *theta = *theta - ((alpha / m) * matExtX.t() * (vec_h - vec_y));
    }
}
