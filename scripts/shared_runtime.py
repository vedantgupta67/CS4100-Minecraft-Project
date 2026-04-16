"""Shared runtime constants/helpers used across scripts.

This module keeps environment seed, GUI slot calibration defaults, and JVM
memory patching in one place so training/evaluation/calibration do not drift.
"""

WORLD_SEED = 175901257196164
ACTION_REPEAT = 4

# GUI slot positions as camera-degree offsets from screen center.
# Values are (pitch_delta, yaw_delta) from the cursor's center position.
SLOTS = {
    "recipe_book_btn": (-2, 4),
    "recipe_planks": (-6, -20),
    "recipe_table": (-6, -12),
    "output_slot": (-6, 23),
    "inv_slot": (0, 0),
}


def patch_jvm_memory(max_mem: str = "6G"):
    """Monkey-patch MinecraftInstance heap size before creating environments."""
    try:
        import minerl.env.malmo as _malmo

        _orig = _malmo.MinecraftInstance.__init__

        def _patched(self, port=None, existing=False, status_dir=None,
                     seed=None, instance_id=None, max_mem=None):
            _orig(
                self,
                port=port,
                existing=existing,
                status_dir=status_dir,
                seed=seed,
                instance_id=instance_id,
                max_mem=max_mem or max_mem_default,
            )

        max_mem_default = max_mem
        _malmo.MinecraftInstance.__init__ = _patched
        print(f"[make_env] JVM heap set to {max_mem}")
    except Exception as e:
        print(f"[make_env] Warning: could not patch JVM memory: {e}")
