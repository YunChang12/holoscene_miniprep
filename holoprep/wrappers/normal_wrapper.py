"""Placeholder normal model integration."""

from __future__ import annotations


class NormalWrapper:
    """External normal model wrapper placeholder."""

    def run(self, *args, **kwargs):
        """Generate normals with an external model."""

        raise RuntimeError(
            "Normal model is not integrated in holoscene_miniprep. Provide normal.provided_dir, "
            "use normal.mode=depth_to_normal/dummy, or implement NormalWrapper.run()."
        )
