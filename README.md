# HoloScene MiniPrep

`holoscene_miniprep` 是一个最小可运行的 HoloScene 数据预处理项目，用于把普通视频或已有图片序列整理成 HoloScene 可读取的 capture 目录。

当前版本的目标不是做一个复杂通用的数据集转换框架，而是先把稳定的数据格式链路跑通。VGGT、SAM2、Depth Anything、Marigold、Omnidata 等外部模型本体不会直接内置在本项目中，而是通过 wrapper 预留接口接入。如果配置选择了外部模型模式，但 wrapper 尚未实现，程序会给出明确错误，不会静默生成错误数据。

## 输出结构

目标输出目录：

```text
data_dir/custom/<scene_name>/
  images/
    frame000000.jpg
    frame000001.jpg
  instance_mask/
    frame000000.png
    frame000001.png
  depth/
    frame000000.npy
    frame000001.npy
  normal/
    frame000000.png
    frame000001.png
  transforms.json
  graph.json
  meta/
  review/
```

HoloScene 关键输入包括：

- `images/`
- `instance_mask/`
- `depth/`
- `normal/`
- `transforms.json`
- `graph.json`

逐帧对齐要求：

- `images`、`instance_mask`、`depth`、`normal` 和 `transforms.json.frames` 的帧数必须一致。
- 文件 basename 必须一致，例如 `frame000123.jpg`、`frame000123.png`、`frame000123.npy`。
- 所有图像、mask、depth、normal 的分辨率应一致。
- 推荐分辨率为 `512x512`，由配置项 `frame.resolution` 控制。

## Mask 约定

`instance_mask/*.png` 是单通道 label mask：

- 背景固定为 `255`。
- 前景物体 raw id 连续编号为 `0, 1, 2, ...`。
- HoloScene loader 通常会把背景 `255` 映射成内部 `node_id=0`。
- HoloScene loader 通常会把前景 raw id `N` 映射成内部 `node_id=N+1`。
- 因此 `graph.json` 里的物体节点应使用 `raw_mask_value + 1`。

## 运行环境

在当前服务器上，直接复用已有的 `sam3d` conda 环境即可：

```bash
cd /root/autodl-fs/Chengpeng/holoscene_miniprep
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py --config configs/example_images.yaml
```

该环境已经包含：

- `numpy`
- `Pillow`
- `PyYAML`
- `opencv-python`

因此它可以处理图片序列，也可以处理 mp4 视频。

如果换到全新环境，再安装依赖：

```bash
cd /root/autodl-fs/Chengpeng/holoscene_miniprep
pip install -r requirements.txt
```

如果没有 `opencv-python`，仍然可以处理 `input_type: images`；但处理 mp4 视频需要 OpenCV。

## 快速使用

运行图片序列示例：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py --config configs/example_images.yaml
```

运行视频示例：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py --config configs/example_video.yaml
```

如果希望按固定步长抽帧，例如每 2 帧取 1 帧，在配置的 `frame` 段设置：

```yaml
frame:
  stride: 2
```

单独验证已生成的 scene：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/validate_scene.py --scene_dir data_dir/custom/my_image_scene
```

单独生成 review 可视化：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/visualize_scene.py --scene_dir data_dir/custom/my_image_scene
```

选择部分阶段运行：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/example_images.yaml \
  --stages frames,camera,vlm,mask,depth,camera_scale,normal,geometry,graph,validate,review
```

复用已有阶段产物：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py --config configs/example_images.yaml --resume
```

## 配置说明

`scene`：

- `name`：场景名称。
- `input_type`：输入类型，可选 `video` 或 `images`。
- `input_path`：视频文件路径或图片目录路径。
- `output_dir`：最终 HoloScene capture 输出目录。

`frame`：

- `target_fps`：视频抽帧 fps。
- `stride`：抽帧/抽图步长，例如 `2` 表示每 2 帧/张取 1 帧/张；视频输入设置后优先于 `target_fps`。也可写作 `frame_stride`。
- `max_frames`：最多输出帧数。
- `resolution`：输出分辨率，格式为 `[width, height]`。
- `overwrite`：是否覆盖已有 `images/`。

`camera.mode`：

- `fixed`：生成静态相机，适合格式 smoke test。
- `provided`：读取 `camera.provided_transforms`。
- `provided_or_vggt`：如果存在 provided transforms 就读取，否则调用 VGGT wrapper。
- `vggt`：调用 `holoprep/wrappers/vggt_wrapper.py`，当前为占位实现。
- `zaiwu_vggt`：调用 Zaiwu VGGT 服务，生成 `transforms.json`，并可与官方 `fallback_transforms` 对比。
- `long_sequence_vggt`：调用外部 `vggt_long_sequence_pose` 项目做分段 VGGT、Sim(3) 对齐、窗口质量门控和融合，然后转换为 MiniPrep/HoloScene 的 `transforms.json`。推荐用于几百帧到上千帧的长序列。

长序列 VGGT 示例：

```yaml
camera:
  mode: long_sequence_vggt
  service_url: http://127.0.0.1:20008/sse
  long_sequence_project_dir: /autodl-fs/data/Chengpeng/vggt_long_sequence_pose
  input_convention: auto
  max_image_size: 518
  keyframe_interval: 5
  key_window_size: 80
  key_overlap: 20
  all_window_size: 80
  all_overlap: 30
  max_anchor_translation_rmse: 0.15
  max_anchor_rotation_deg: 10.0
  low_quality_fallback_weight: 0.25
  resume: true
```

该模式必须在 `frames` 阶段之后运行，会对 MiniPrep 已经规范化后的
`images/frameXXXXXX.jpg` 进行分段 VGGT。分段项目原始输出保存在
`raw_outputs/vggt_long_sequence/`，MiniPrep 会把其中的
`final_transforms.json` 转换为 `images/frameXXXXXX.jpg` 路径格式，并把
窗口质量统计写入 `meta/camera_source_report.json`。如果不在 YAML 中写
`long_sequence_project_dir`，也可以通过环境变量
`HOLOSCENE_LONG_SEQUENCE_VGGT_DIR` 指定。

`camera_scale`：

- `enabled`：默认 `false`。设为 `true` 后，在 `depth` 阶段之后用 `images/`、`depth/*.npy` 和当前 `transforms.json` 估计一个全局尺度。
- `method`：当前支持 `depth_correspondence`。
- `pair_strategy`：默认 `multi_gap`，会从多个时间间隔采样帧对并优先使用更可靠的 baseline；如需旧行为可设为 `single_gap`。
- `pair_stride` 或 `pair_gap`：基础间隔，默认 `5`。在 `multi_gap` 下会自动扩展为 `pair_stride * [1,2,4,8,16]`。
- `pair_gap_multipliers`：自动扩展倍数，默认 `[1, 2, 4, 8, 16]`。
- `pair_gaps`：显式指定多间隔列表，例如 `[5, 10, 20, 40, 80]`，优先级高于 `pair_stride`。
- `max_pair_gap`：限制自动扩展出的最大间隔。
- `max_pairs`：最多用于估计的图像对数量，默认 `80`。`multi_gap` 会在多个 gap 内均匀覆盖时间轴，再用较大 baseline 候选补齐。
- `min_baseline`：VGGT 相机中心最小间距，过小的帧对会跳过。
- `selection_min_baseline`：帧对选择阶段的最小 baseline；默认复用 `min_baseline`，用于避免把名额浪费在明显短基线帧对上。
- `min_depth`、`max_depth`：参与尺度估计的 depth 范围。
- `min_matches`、`min_observations`：每对/全局最少有效匹配数。
- `min_used_pairs`：全局最少可用帧对数。`multi_gap` 默认 `3`，防止只靠单一帧对完成尺度对齐；`single_gap` 默认 `1`。
- `max_pair_scale_spread_ratio`：多帧对尺度候选一致性门控，默认 `3.0`；若 pair-level 的 p90/p10 超过该值，则拒绝写入尺度，避免 DA3 等深度源跨帧尺度漂移时误对齐。设为 `0` 可关闭。
- `ransac_threshold`：深度三维残差阈值。
- `anchor`：缩放相机中心时保持不动的锚点，默认 `first`。
- `write_backup`：默认 `true`，会写出 `meta/transforms_before_scale_align.json`。
- `overwrite`：默认 `false`，已有 `meta/camera_scale_alignment_report.json` 时不重复缩放；需要重新估计时设为 `true`。

示例：

```yaml
camera_scale:
  enabled: true
  method: depth_correspondence
  pair_strategy: multi_gap
  pair_stride: 5
  pair_gaps: [5, 10, 20, 40, 80]
  max_pairs: 80
  min_baseline: 0.02
  min_used_pairs: 3
  min_depth: 0.05
  max_depth: 20.0
  min_matches: 50
  min_observations: 200
  max_pair_scale_spread_ratio: 3.0
  ransac_threshold: 0.15
  anchor: first
  write_backup: true
  overwrite: false
```

该阶段只修改 `frames[*].transform_matrix` 的平移列，不改变 `transforms.json`
的字段格式。尺度因子、匹配统计和失败原因写入
`meta/camera_scale_alignment_report.json`。

`vlm`：

- 默认全流程阶段为 `frames,camera,vlm,mask,depth,camera_scale,normal,geometry,graph,validate,review`。
- 对 `sam2`、`provided_or_sam2`、`zaiwu_seg2track_sam2` 这类开放词汇 mask 模式，MiniPrep 默认先用 VLM 生成 `meta/vlm_object_prompt.json`，再把其中的 `prompt` 传给 mask 阶段。
- `mask.text_prompt: auto`、`vlm`、`from_vlm`，或 `mask.use_vlm_prompt: true` 都会启用 VLM prompt。
- 如果需要完全手写词表，设置明确的 `mask.text_prompt` 并配置 `mask.use_vlm_prompt: false`。
- VLM 默认读取 `configs/vlm_local.yaml`，也可在 `vlm.config_path` 中指定 OpenAI-compatible chat completion 配置。

`mask.mode`：

- `dummy`：生成一个居中的前景物体 mask。
- `provided`：从 `mask.provided_dir` 读取已有 mask。
- `provided_or_sam2`：如果存在 provided masks 就读取，否则调用 SAM2/Seg2Track wrapper。
- `sam2`：调用 `holoprep/wrappers/sam2_wrapper.py`，当前为占位实现。
- `zaiwu_seg2track_sam2`：调用 Zaiwu Seg2Track-SAM2 服务，输出跨帧稳定 `instance_mask` 和 `meta/id_mapping.json`。

`depth.mode`：

- `dummy`：生成常数深度图，适合格式测试。
- `provided`：读取已有 `.npy` 或图像深度。
- `provided_or_model`、`da3`、`marigold`、`model`：调用 `holoprep/wrappers/depth_wrapper.py`，当前为占位实现。
- `zaiwu_da3`：调用 Zaiwu Depth Anything 3 服务，把返回的 stacked depth `.npy` 拆成逐帧 `depth/frameXXXXXX.npy`。

`normal.mode`：

- `depth_to_normal`：根据 depth 和相机内参估计 normal。
- `dummy`：生成 `[0, 0, 1]` 法线。
- `provided`：读取已有 RGB normal map。
- `model`、`omnidata`、`marigold`：调用 `holoprep/wrappers/normal_wrapper.py`，当前为占位实现。

`graph.mode`：

- `auto_simple`：根据实例点云 bbox 简单推断支撑图，低置信物体默认挂到 root。
- `manual`：复制 `graph.manual_graph`。

## 使用已有数据

使用已有 `transforms.json`：

```yaml
camera:
  mode: provided
  provided_transforms: /path/to/transforms.json
```

使用已有 mask：

```yaml
mask:
  mode: provided
  provided_dir: /path/to/instance_mask
  background_value: 255
```

如果外部 mask 的背景值不是 `255`，请把 `mask.background_value` 设置为外部 mask 的背景值。最终写出的 HoloScene mask 仍会固定使用背景 `255`。

使用已有 depth：

```yaml
depth:
  mode: provided
  provided_dir: /path/to/depth
```

使用已有 normal：

```yaml
normal:
  mode: provided
  provided_dir: /path/to/normal
```

提供的 transforms 会被规范化：

- 每帧 `file_path` 改为 `images/frameXXXXXX.jpg`。
- 全局内参会按配置中的目标分辨率进行缩放。

## 外部模型接入位置

wrapper 文件位置：

```text
holoprep/wrappers/vggt_wrapper.py
holoprep/wrappers/sam2_wrapper.py
holoprep/wrappers/depth_wrapper.py
holoprep/wrappers/normal_wrapper.py
```

推荐与 Zaiwu 服务的映射关系：

- VLM：调用 OpenAI-compatible chat completion 接口，从抽样帧生成开放词汇 `text_prompt`。
- VGGT：调用 `services.vggt.reconstruct_scene_from_dir`，再把返回的 intrinsics/extrinsics 转成 HoloScene `transforms.json`。
- Seg2Track-SAM2：调用 `services.seg2track_sam2.seg2track_parse_video`，再把 `mask_rle` 转成逐帧 `instance_mask/*.png`。
- HoloScene Stage 0 或 Marigold：生成 `depth/*.npy` 和 `normal/*.png`。

当前最小项目不会对外部模型模式做静默 fallback。如果 wrapper 未实现，命令会明确失败，方便定位问题。

## Replica room_0 真实模型接入测试

这一组流程使用 HoloScene 官方 Replica `room_0` 的少量帧作为稳定测试源，逐步验证 Zaiwu 上部署的 Depth Anything 3、Seg2Track-SAM2、VGGT 是否能生成 HoloScene 可读取的数据。

当前仓库配置默认使用本机可用服务：

```text
Depth Anything 3:  http://127.0.0.1:8443   # Gateway job 已验证可用
Seg2Track-SAM2:   http://127.0.0.1:20010  # direct worker /sse
VGGT:             http://127.0.0.1:20008  # direct worker /sse
```

wrapper 会优先尝试 Gateway `/api/v1/jobs`；如果 `service_url` 指向直接 worker 端口，也会尝试 `/sse` 直接调用。

常见直接 worker 端口示例：

```text
Depth Anything 3:  http://127.0.0.1:20001
Seg2Track-SAM2:   http://127.0.0.1:20010
VGGT:             http://127.0.0.1:20008
```

如果当前 Zaiwu Gateway 或服务端口不同，只需要修改 `configs/room0_mini_*.yaml` 里的 `service_url`。

### T0：制作 room0 mini scene

```bash
cd /root/autodl-fs/Chengpeng/holoscene_miniprep
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/make_room0_mini_scene.py \
  --source_scene /root/autodl-fs/Zaiwu/third_party/HoloScene/data_dir/replica/room_0 \
  --output_scene tmp_tests/room0_mini \
  --num_frames 30 \
  --stride 5 \
  --resolution 512 512
```

输出：

```text
tmp_tests/room0_mini/
  images/frame000000.jpg
  transforms_official.json
  meta/source_frame_mapping.json
```

验收标准：

- 不修改官方 `room_0`。
- `images/`、`transforms_official.json.frames` 数量一致。
- `transforms_official.json` 的 `file_path` 是 `images/frameXXXXXX.jpg`。
- resize 后 `fl_x/fl_y/cx/cy/h/w` 已同步缩放。

### T1：官方 camera + DA3

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/room0_mini_official_camera_da3.yaml \
  --stages frames,camera,vlm,mask,depth,camera_scale,normal,geometry,graph,validate,review \
  --resume

/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir data_dir/custom/room0_mini_da3
```

验收标准：

- `validate` 无 error。
- `depth/frame000000.npy` 到最后一帧数量完整，`meta/depth_report.json` 无 NaN/Inf。
- `review/depth_vis/` 和 `review/normal_vis/` 视觉合理。
- loader smoke test `ok=true`。

### T2：官方 camera + DA3 + Seg2Track-SAM2

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/room0_mini_zaiwu_depth_s2t.yaml \
  --stages frames,camera,depth,camera_scale,normal,vlm,mask,geometry,graph,validate,review \
  --resume

/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir data_dir/custom/room0_mini_s2t
```

验收标准：

- `review/mask_overlay.mp4` 中实例边界大致合理。
- 同一物体跨帧 ID 稳定。
- `meta/id_mapping.json` 中 `raw_mask_value` 和 `holoscene_node_id` 对应正确。
- `graph.json` 覆盖所有 `holoscene_node_id`。
- loader smoke test `ok=true`。

### T3：VGGT + DA3 + Seg2Track-SAM2

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/room0_mini_zaiwu_full.yaml \
  --stages frames,depth,vlm,mask,camera,camera_scale,normal,geometry,graph,validate,review \
  --resume

/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir data_dir/custom/room0_mini_zaiwu_full
```

验收标准：

- `transforms.json` 能通过 validate。
- `meta/camera_report.json` 记录 VGGT 坐标系解析结果。
- 如果配置了 `fallback_transforms`，会生成 `meta/camera_compare_report.json`。
- `review/camera_trajectory.ply` 轨迹合理，`review/instance_clouds/` 不明显散开。
- loader smoke test `ok=true`。

### T4：可选 HoloScene Stage 1 debug

在 T3 通过后生成短迭代 debug conf：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/make_holoscene_debug_conf.py \
  --scene_dir data_dir/custom/room0_mini_zaiwu_full \
  --template_conf /root/autodl-fs/Zaiwu/third_party/HoloScene/confs/replica/room_0/replica_room_0.conf \
  --output_conf /root/autodl-fs/Zaiwu/third_party/HoloScene/confs/custom/room0_mini_zaiwu_full_debug.conf
```

然后到 HoloScene 根目录运行：

```bash
cd /root/autodl-fs/Zaiwu/third_party/HoloScene
python training/exp_runner.py \
  --conf confs/custom/room0_mini_zaiwu_full_debug.conf \
  --none_wandb
```

debug 只建议跑 10 到 50 step，目标是验证 dataset loader、loss 和训练入口，不追求重建质量。

## Review 输出

`review/` 目录可能包含：

- `mask_overlay.mp4`：OpenCV 可用时生成。
- `mask_overlay_frames/*.png`：逐帧 overlay，可作为视频生成失败时的 fallback。
- `depth_vis/*.png`：depth percentile 彩色可视化。
- `normal_vis/*.png`：normal RGB 可视化。
- `camera_trajectory.ply`：相机中心轨迹。
- `graph_vis.png`：简单 graph 预览。
- `instance_clouds/object_001.ply`：每个物体的粗点云，文件名使用 HoloScene loaded id。

## Validation 报告

`validate_scene.py` 不只检查文件是否存在，还会检查 HoloScene loader 读取时容易出错的细节。

会写出：

```text
meta/frame_consistency_report.json
meta/camera_report.json
meta/camera_scale_alignment_report.json
meta/transforms_before_scale_align.json
meta/mask_report.json
meta/id_mapping_check.json
meta/depth_report.json
meta/normal_report.json
meta/graph_report.json
meta/validation_report.json
meta/validation_report.md
```

返回状态：

- `pass`：无 error、无 warning。
- `warning`：格式可读取，但存在弱数据，例如 fixed camera、dummy normal、所有物体都挂 root。
- `fail`：存在会导致 HoloScene 读取失败或明显错位的问题。

严重错误会让命令返回非零状态码；warning 不会阻断 pipeline。

## Smoke Test

项目内置一个最小 smoke test，位于 `tmp_demo/`：

```bash
cd /root/autodl-fs/Chengpeng/holoscene_miniprep
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py --config tmp_demo/config_smoke.yaml
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/validate_scene.py --scene_dir tmp_demo/data_dir/custom/smoke_scene
```

预期验证结果：

```text
ok: true
images: 4
instance_mask: 4
depth: 4
normal: 4
```

该最小测试会产生 warning，这是预期行为，因为它使用 fixed camera、dummy depth/normal 和简单 root graph。

## HoloScene Loader Smoke Test

轻量模拟 HoloScene dataset loader 读取：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir tmp_demo/data_dir/custom/smoke_scene
```

如果想顺便尝试 import 官方 HoloScene dataset loader：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir tmp_demo/data_dir/custom/smoke_scene \
  --holoscene_root /root/autodl-fs/Zaiwu/third_party/HoloScene
```

注意：官方 loader import 依赖 HoloScene 自己的环境。如果当前 Python 环境缺少 `nvdiffrast` 等依赖，轻量 loader 仍可通过，但官方 import 会在 `meta/holoscene_loader_test.json` 中记录失败原因。

## HoloScene Debug Conf

生成短迭代 debug conf：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/make_holoscene_debug_conf.py \
  --scene_dir tmp_demo/data_dir/custom/smoke_scene \
  --template_conf /root/autodl-fs/Zaiwu/third_party/HoloScene/confs/replica/room_0/replica_room_0.conf \
  --output_conf tmp_demo/confs/custom/smoke_scene/smoke_scene_debug.conf
```

脚本会自动修改：

- `expname`
- `data_root_dir`
- `data_dir`
- `img_res`
- `max_total_iters`
- `stop_iter`
- `plot_freq`
- `checkpoint_freq`

然后到 HoloScene 根目录运行：

```bash
python training/exp_runner.py \
  --conf /root/autodl-fs/Chengpeng/holoscene_miniprep/tmp_demo/confs/custom/smoke_scene/smoke_scene_debug.conf \
  --none_wandb
```

## 逐步替换 Dummy 数据

替换 provided depth，然后从 depth 重新生成 normal：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/example_video.yaml \
  --stages depth,camera_scale,normal,geometry,graph,validate,review \
  --resume
```

替换 provided mask：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/example_video.yaml \
  --stages vlm,mask,geometry,graph,validate,review \
  --resume
```

替换 provided transforms：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/example_video.yaml \
  --stages camera,geometry,graph,validate,review \
  --resume
```

替换 manual graph：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/run_pipeline.py \
  --config configs/example_video.yaml \
  --stages graph,validate,review \
  --resume
```

每次替换后建议都运行：

```bash
/root/autodl-tmp/conda-envs/sam3d/bin/python scripts/test_holoscene_loader.py \
  --scene_dir <scene_dir>
```

## 常见问题

`Video input requires opencv-python`：

说明当前环境缺少 OpenCV。可以安装 `opencv-python`，或者先把视频抽成图片，再使用 `input_type: images`。

`VGGT/SAM2/Depth model is not integrated`：

说明配置选择了外部模型模式，但对应 wrapper 仍是占位。可以先使用 `fixed`、`dummy` 或 `provided` 模式跑通格式，或者实现对应 wrapper。

`Frame counts differ`：

检查 `images`、`instance_mask`、`depth`、`normal` 的文件数量是否完全一致，并确认 basename 是否逐帧对应。

`Mask labels are not continuous`：

说明源 mask 可能存在跳号、异常高前景值，或背景值配置不一致。使用 `mask.mode: provided` 时请正确设置 `mask.background_value`，让 MiniPrep 重新 remap label。

`graph missing nodes from id_mapping`：

说明 `graph.json` 没有覆盖所有物体节点。可以使用 `graph.mode: auto_simple` 重新生成，或手工补齐缺失节点。
