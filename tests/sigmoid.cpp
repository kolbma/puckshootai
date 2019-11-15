#include <catch.hpp>
#include "puckshootai.hpp"
#include "util.hpp"

using namespace arma;
using namespace godot;

TEST_CASE("Sigmoid", "[Sigmoid]")
{
    SECTION("mat")
    {
        mat g;
        mat m1 = {{1, 2, 3}, {5, 6, 7}, {2, 8, 9}, {8, 7, 2}};
        PuckShootAI::sigmoid(&g, m1);
        REQUIRE(g.n_cols == 3);
        REQUIRE(g.n_rows == 4);
    }

    SECTION("vec")
    {
        mat g;
        mat m1 = {{1, 2, 3}};
        PuckShootAI::sigmoid(&g, m1);
        REQUIRE(g.n_cols == 3);
        REQUIRE(g.n_rows == 1);
        REQUIRE(is_almost_equal(g(0), 0.73106, 5));
    }
}
