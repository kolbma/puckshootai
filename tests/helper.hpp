#ifndef HELPER_HPP
#define HELPER_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <Godot.hpp>
#pragma GCC diagnostic pop

#include "puckshootai.hpp"

class Helper : public godot::PuckShootAI
{
public:
    void set_data(const godot::Vector2 &window_size, const godot::Vector2 &puck_position,
                  const godot::Vector2 &puck_velocity,
                  int64_t refl_left_y, int64_t refl_right_y)
    {
        this->window_size = window_size;
        this->puck_position = puck_position;
        this->puck_velocity = puck_velocity;
        this->refl_left_y = refl_left_y;
        this->refl_right_y = refl_right_y;
    }
};

#endif // HELPER_HPP
