"""Placeholder SAM2/Seg2Track integration."""

from __future__ import annotations


class SAM2Wrapper:
    """External SAM2 wrapper placeholder."""

    def run(self, *args, **kwargs):
        """Generate masks with SAM2 or Seg2Track.

        This minimal project does not call the deployed service directly.
        Use mask.mode=provided or mask.mode=dummy, or implement this method to
        call Zaiwu services.seg2track_sam2 and convert mask_rle to label masks.
        """

        raise RuntimeError(
            "SAM2/Seg2Track is not integrated in holoscene_miniprep. Provide mask.provided_dir "
            "or implement SAM2Wrapper.run() to call Zaiwu services.seg2track_sam2."
        )
