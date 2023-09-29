from rtm.kinematics import kinematics

from rtm.utils import (
    ops,
)

profiler_kinematics = ops.Profile()  # count the time
# time the function
with profiler_kinematics:
    kinematics = kinematics.Kinematics(run_path="20230929-022531/Men 100m Heat-1.csv")
    kinematics(save=True, overwrite=True)
print(f"{profiler_kinematics.dt * 1E3:.1f}ms")
