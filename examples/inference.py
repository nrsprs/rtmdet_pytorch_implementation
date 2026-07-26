from rtmdet import RTMDet

model = RTMDet.from_preset("small")  # tiny / small / medium / large
bboxes, scores, classes = model("examples/assets/city.png")
print(bboxes, scores, classes)
