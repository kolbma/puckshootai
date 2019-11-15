#include <catch.hpp>
#include "puckshootai.hpp"

using namespace arma;
using namespace godot;

TEST_CASE("Train", "[Train]")
{
    SECTION("train1")
    {
        colvec x = join_vert(ones(10, 1), zeros(10, 1));
        REQUIRE(x.n_rows > 1);
        REQUIRE(x.n_cols == 1);
        PuckShootAI ai;
        rowvec mu, sigma;
        colvec theta;
        ai.train(&mu, &sigma, &theta);
        REQUIRE(mu.n_elem == 1);
        REQUIRE(sigma.n_elem == 1);
        REQUIRE(theta.n_elem == 2);
    }

    SECTION("trainAndPredict")
    {
        PuckShootAI ai;
        rowvec mu, sigma;
        colvec theta;
        ai.train(&mu, &sigma, &theta);
        REQUIRE(mu.n_elem == 1);
        REQUIRE(sigma.n_elem == 1);
        REQUIRE(theta.n_elem == 2);

        rowvec sample = {1, (0.49 - mu(0)) / sigma(0)};
        rowvec predict = sample * theta;
        colvec predict_g;
        PuckShootAI::sigmoid(&predict_g, predict);

        REQUIRE(predict_g(0) < 0.5);

        sample = {1, (0.51 - mu(0)) / sigma(0)};
        predict = sample * theta;
        PuckShootAI::sigmoid(&predict_g, predict);

        REQUIRE(predict_g(0) >= 0.5);
    }
}
