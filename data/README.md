# LineCount AI — Video Data Guidelines

## Overview

LineCount AI is evaluated on industrial conveyor videos and public benchmark tracking datasets (such as MOT17/MOT20 conveyor sequences or simulated manufacturing line footage).

## Sample Test Videos

To run testing with your own video streams or public benchmark videos:
1. Place conveyor MP4/AVI videos inside `data/` (e.g. `data/conveyor_sample.mp4`)
2. Run tracking and counting:
   ```powershell
   python scripts/detect_video.py --video data/conveyor_sample.mp4 --output outputs/tracked_conveyor.mp4
   ```

## Ground Truth Annotation Format

For quantitative counting verification, test video ground truth counts are maintained in simple JSON format:

```json
{
  "video_name": "conveyor_sample.mp4",
  "line_gate": "main_line",
  "ground_truth_counts": {
    "bottle": 48,
    "can": 32
  }
}
```
