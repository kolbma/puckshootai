#include <catch.hpp>
#include "helper.hpp"
#include "util.hpp"

using namespace arma;
using namespace godot;

TEST_CASE("Predict", "[Predict]")
{
    SECTION("center-to-abs")
    {
        const auto wsize_half_y = 600 / 2.0;
        const double ws_y_step = 0.5 / wsize_half_y;
        auto puck_abs_y = ws_y_step * (wsize_half_y - -100);
        auto refl_left_abs_y = ws_y_step * (wsize_half_y - 100);
        REQUIRE(is_almost_equal(puck_abs_y, 400.0 / 600.0, 5));
        REQUIRE(is_almost_equal(refl_left_abs_y, 200.0 / 600.0, 5));
        puck_abs_y = ws_y_step * (wsize_half_y - 100);
        refl_left_abs_y = ws_y_step * (wsize_half_y - -100);
        REQUIRE(is_almost_equal(puck_abs_y, 200.0 / 600.0, 5));
        REQUIRE(is_almost_equal(refl_left_abs_y, 400.0 / 600.0, 5));
        puck_abs_y = ws_y_step * (wsize_half_y - -100);
        refl_left_abs_y = ws_y_step * (wsize_half_y - 0);
        REQUIRE(is_almost_equal(puck_abs_y, 400.0 / 600.0, 5));
        REQUIRE(is_almost_equal(refl_left_abs_y, 300.0 / 600.0, 5));
    }

    SECTION("predict-left-0")
    {
        Helper ai;
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(-10, 5), 0, 0);
        auto speed = ai.get_y_left();
        REQUIRE(speed == Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(-10, 5), -80, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, 20), Vector2(-10, 5), 0, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(-10, 5), -1, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(-10, 5), 1, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == Helper::SPEED);
    }

    SECTION("predict-left-1")
    {
        Helper ai;
        ai.set_data(Vector2(1024, 600), Vector2(-100, -100), Vector2(-10, 5), 0, 0);
        auto speed = ai.get_y_left();
        REQUIRE(speed == -Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, -100), Vector2(-10, 5), 80, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == -Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(-10, 5), 120, 0);
        speed = ai.get_y_left();
        REQUIRE(speed == -Helper::SPEED);
    }

    SECTION("predict-right-0")
    {
        Helper ai;
        ai.set_data(Vector2(1024, 600), Vector2(-100, 100), Vector2(10, 5), 0, 0);
        auto speed = ai.get_y_right();
        REQUIRE(speed == Helper::SPEED);
        ai.set_data(Vector2(1024, 600), Vector2(0, 100), Vector2(10, 5), 0, -100);
        speed = ai.get_y_right();
        REQUIRE(speed == Helper::SPEED);
    }

    SECTION("predict-right-1")
    {
        Helper ai;
        ai.set_data(Vector2(1024, 600), Vector2(-100, -100), Vector2(10, 5), 0, 0);
        const auto speed = ai.get_y_right();
        REQUIRE(speed == -Helper::SPEED);
    }
}
