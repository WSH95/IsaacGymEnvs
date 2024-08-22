# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/5/27.
#

def custom_up_down_stairs_terrain(terrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        height += step_height * ((-1) ** (i))
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] = height

    return terrain


def custom_up_step_terrain(terrain, forward_distance, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        forward_distance (float):  the distance of the step from origin [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_distance = int(forward_distance / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    terrain.height_field_raw[:, :] = step_height
    terrain.height_field_raw[(terrain.width // 2)-step_distance: (terrain.width // 2)+step_distance, :] = 0

    return terrain
