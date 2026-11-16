# First-release scope

Ray360 serves developers applying computer vision to stitched equirectangular
images. A video is a sequence of application-decoded panorama arrays.

The supported workflow is view extraction, explicit preprocessing, point/box
mapping, seam-aware rendering, and conversion between equirectangular images and
six named cube faces. The same faces can be rendered as a single cross image.
NumPy remains the core backend. Pillow adds loading, saving, and outline overlays.

The earlier development implementation included camera adapters, external process
bridges, PyAV sources/writers, live buffers, Torch sampling, model wrappers, COCO/YOLO
converters, fisheye conversions, and a CLI. Those modules and install extras
have been removed. There are no compatibility aliases. The old dataset conversion
and filename-generation paths are gone, including their lossy metadata handling
and dotted-prefix collision. Applications write their own format-specific exports.

Cubemap conversion uses the existing perspective geometry, with explicit face
orientation and adjacent-face filtering. It adds no backend dependency. Model and
decoder integrations remain application examples with opt-in end-to-end tests.

## Experimental merging

`ray360.experimental.spherical_iou` estimates solid-angle overlap using
deterministic equal-area sphere samples. `spherical_nms` uses that estimate to
suppress lower-scoring same-label regions. It uses score 1 for missing scores;
provide comparable confidence values or avoid suppression. `class_agnostic=True`
allows suppression across labels. Suppression uses overlap strictly above the
threshold. No suppression runs automatically during extraction or mapping.

The tests include overlapping detections from two different camera directions,
class-aware suppression, and an analytic overlap reference. They do not measure
detector accuracy on a labeled multi-view dataset. Small objects can receive zero
sample hits. NMS does not join partial detections, perform tracking, or combine
segmentation masks. Treat these operations as experiments requiring application
validation; they are excluded from the stable root API.

## Criteria for expanding scope

Before adding another backend or annotation type, use the existing workflow in
an independent consuming project and record the missing behavior. Evaluate real
stitched scenes with small objects, seams, poles, and overlapping views. Measure
mapping error, missed objects, duplicate predictions, full-panorama latency, and
memory. A passing geometry suite does not establish detector recall.

Future segmentation, tensor, or dataset support should have a specific consumer,
independent correctness checks, and a maintained compatibility matrix. Raw-camera
stitching and production streaming remain separate application concerns.
