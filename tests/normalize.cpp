#include <catch.hpp>
#include "puckshootai.hpp"
#include "util.hpp"

using namespace arma;
using namespace godot;

TEST_CASE("Normalize", "[Normalize]")
{
    SECTION("mat")
    {
        mat matX = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {4, 8, 1}};

        mat normX;
        rowvec mu, sigma;
        PuckShootAI::normalize(&normX, &mu, &sigma, matX);
        REQUIRE(normX.n_cols == 3);
        REQUIRE(normX.n_rows == 4);
        REQUIRE(mu.n_elem == matX.n_cols);
        REQUIRE(sigma.n_elem == matX.n_cols);
        REQUIRE(is_almost_equal(mu(0), 4.0000, 4));
        REQUIRE(is_almost_equal(mu(2), 4.7500, 4));
        REQUIRE(is_almost_equal(sigma(0), 2.4495, 4));
        REQUIRE(is_almost_equal(sigma(2), 3.5000, 4));
        REQUIRE(is_almost_equal(normX(0, 0), -1.22474, 5));
        REQUIRE(is_almost_equal(normX(2, 1), 0.78335, 5));
    }

    SECTION("vec")
    {
        vec v = {1, 2, 3, 4, 5, 6};
        vec norm;
        rowvec mu, sigma;
        PuckShootAI::normalize(&norm, &mu, &sigma, v);
        REQUIRE(norm.n_elem == v.n_elem);
        REQUIRE(norm.n_rows == v.n_rows);
        REQUIRE(mu.n_elem == v.n_cols);
        REQUIRE(sigma.n_elem == v.n_cols);
        REQUIRE(is_almost_equal(mu(0), 3.5000, 4));
        REQUIRE(is_almost_equal(sigma(0), 1.8708, 4));
        REQUIRE(is_almost_equal(norm(0), -1.33631, 5));
        REQUIRE(is_almost_equal(norm(3), 0.26726, 5));
    }
}
