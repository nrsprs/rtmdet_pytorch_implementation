from rtmdet import RTMDet


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = RTMDet.from_preset("small", pretrained=True).to(device)
    model.cfg.score_threshold = 0.6

    image_path = "examples/assets/city.png"
    bboxes, scores, classes = model(image_path)

    print(f"Detections: {len(bboxes)}")

    for bbox, score, cls in zip(bboxes, scores, classes):
        x1, y1, x2, y2 = bbox.tolist()
        print(f"  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]  score={score:.3f}  class={cls.item()}")

    img = model.draw_detections(image_path, bboxes, scores, classes)
    output_path = "examples/out/detections.png"
    img.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
