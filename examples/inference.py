from rtmdet import RTMDet

model = RTMDet.from_preset("small", pretrained=True)  # tiny / small / medium / large
model.cfg.score_threshold = 0.6
image_path = "examples/assets/city.png"
bboxes, scores, classes = model(image_path)
print(f"Detections: {len(bboxes)}")
for bbox, score, cls in zip(bboxes, scores, classes):
    x1, y1, x2, y2 = bbox.tolist()
    print(f"  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]  score={score:.3f}  class={cls.item()}")
img = model.draw_detections(image_path, bboxes, scores, classes)
img.save("detections.png")
print("Saved to detections.png")
