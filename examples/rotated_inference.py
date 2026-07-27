from rtmdet import RotRTMDet


def main() -> None:
    model = RotRTMDet.from_preset("large", pretrained=True)
    model.cfg.score_threshold = 0.05

    image_path = "examples/assets/city.png"
    bboxes, scores, classes = model.predict(image_path)

    print(f"Detections: {len(bboxes)}")
    for bbox, score, cls in zip(bboxes, scores, classes):
        cx, cy, w, h, theta = bbox.tolist()
        print(f"  [{cx:.1f}, {cy:.1f}, {w:.1f}, {h:.1f}, {theta:.3f}]  score={score:.3f}  class={cls.item()}")

    img = model.draw_detections(image_path, bboxes, scores, classes)
    output_path = "examples/out/rotated_detections.png"
    img.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
