"""Franka placement into a marked tabletop target region."""

from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Franka Emika robot.
Goal: place the primary object fully inside the marked target region, release it
gently, and open the gripper. Query the live target region and object extent.
The whole object footprint, not only its center, must fit inside the region.
Write only executable Python code. Import numpy explicitly when needed.
"""


class FrankaRegionPlaceCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT


__all__ = ["FrankaRegionPlaceCodeEnv"]
