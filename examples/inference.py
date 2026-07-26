from rtmdet import RTMDet

model = RTMDet.from_preset("small")  # tiny / small / medium / large
image_path = "examples/assets/highway.png"
bboxes, scores, classes = model(image_path)
print(f"Detections: {len(bboxes)}")
img = model.draw_detections(image_path, bboxes, scores, classes)
img.save("detections.png")
print("Saved to detections.png")
