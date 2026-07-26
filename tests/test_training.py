import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, complete_box_iou_loss

from rtmdet import RTMDet


class ShapesDataset(Dataset):
    """Minimal synthetic dataset: random rectangles with random labels."""

    def __init__(self, n, img_size, num_classes, seed=42):
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes
        self.rng = torch.manual_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img = torch.rand(3, self.img_size, self.img_size)
        s = self.img_size // 3
        x1 = torch.randint(0, self.img_size - s, (1,)).item()
        y1 = torch.randint(0, self.img_size - s, (1,)).item()
        x2 = x1 + s
        y2 = y1 + s
        bboxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        label = torch.randint(0, self.num_classes, (1,))
        return img, {"bboxes": bboxes, "labels": label}


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


def best_iou_idx(pred_boxes, gt_boxes):
    B, _, _ = pred_boxes.shape
    out = []
    for b in range(B):
        ious = box_iou(pred_boxes[b], gt_boxes[b].view(1, 4)).squeeze(-1)
        out.append(torch.argmax(ious))
    return torch.stack(out)


def _make_model():
    return RTMDet.from_preset("tiny", img_size=128, num_classes=2, pretrained=False).to("cpu")


class TestTraining:
    def test_single_step(self):
        model = _make_model()
        model.train()
        imgs = torch.rand(4, 3, 128, 128)
        gt_boxes = torch.rand(4, 4) * 128
        gt_boxes[:, 2:] = gt_boxes[:, :2] + 10
        gt_labels = torch.randint(0, 2, (4,))

        pred_boxes, _, _, pred_logits = model(imgs, return_logits=True)
        best_idx = best_iou_idx(pred_boxes, gt_boxes)

        pred_best = pred_boxes[torch.arange(4), best_idx]
        logits_best = pred_logits[torch.arange(4), best_idx]

        ciou = complete_box_iou_loss(pred_best, gt_boxes, reduction="mean")
        ce = nn.CrossEntropyLoss()(logits_best, gt_labels)
        loss = 2.0 * ciou + ce

        assert loss.dim() == 0
        assert loss.shape == ()  # scalar

    def test_gradients_exist(self):
        model = _make_model()
        model.train()
        imgs = torch.rand(4, 3, 128, 128)
        gt_boxes = torch.rand(4, 4) * 128
        gt_boxes[:, 2:] = gt_boxes[:, :2] + 10
        gt_labels = torch.randint(0, 2, (4,))

        pred_boxes, _, _, pred_logits = model(imgs, return_logits=True)
        best_idx = best_iou_idx(pred_boxes, gt_boxes)
        pred_best = pred_boxes[torch.arange(4), best_idx]
        logits_best = pred_logits[torch.arange(4), best_idx]

        ciou = complete_box_iou_loss(pred_best, gt_boxes, reduction="mean")
        ce = nn.CrossEntropyLoss()(logits_best, gt_labels)
        loss = 2.0 * ciou + ce
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_parameters_change(self):
        model = _make_model()
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=4e-3)
        original = {k: v.clone() for k, v in model.state_dict().items()}

        imgs = torch.rand(4, 3, 128, 128)
        gt_boxes = torch.rand(4, 4) * 128
        gt_boxes[:, 2:] = gt_boxes[:, :2] + 10
        gt_labels = torch.randint(0, 2, (4,))

        pred_boxes, _, _, pred_logits = model(imgs, return_logits=True)
        best_idx = best_iou_idx(pred_boxes, gt_boxes)
        pred_best = pred_boxes[torch.arange(4), best_idx]
        logits_best = pred_logits[torch.arange(4), best_idx]

        ciou = complete_box_iou_loss(pred_best, gt_boxes, reduction="mean")
        ce = nn.CrossEntropyLoss()(logits_best, gt_labels)
        loss = 2.0 * ciou + ce
        loss.backward()
        optim.step()

        changed = False
        for k in original:  # noqa: PLC0206
            if not torch.allclose(original[k], model.state_dict()[k]):
                changed = True
                break
        assert changed, "No parameters changed after optimizer step"

    def test_loss_decreases(self):
        model = _make_model()
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=4e-3)
        losses = []
        dataset = ShapesDataset(200, 128, 2, seed=42)
        loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

        for imgs, targets in loader:
            gt_boxes = torch.stack([t["bboxes"] for t in targets]).squeeze(1)
            gt_labels = torch.cat([t["labels"] for t in targets]).squeeze(-1)

            pred_boxes, _, _, pred_logits = model(imgs, return_logits=True)
            best_idx = best_iou_idx(pred_boxes, gt_boxes)
            pred_best = pred_boxes[torch.arange(len(imgs)), best_idx]
            logits_best = pred_logits[torch.arange(len(imgs)), best_idx]

            ciou = complete_box_iou_loss(pred_best, gt_boxes, reduction="mean")
            ce = nn.CrossEntropyLoss()(logits_best, gt_labels.squeeze(-1))
            loss = 2.0 * ciou + ce

            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())

        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"

    def test_batch_size_1(self):
        model = _make_model()
        model.train()
        imgs = torch.rand(1, 3, 128, 128)
        pred_boxes, _, _, _ = model(imgs, return_logits=True)
        assert pred_boxes.shape[0] == 1

    def test_return_logits_4_tuple(self):
        model = _make_model()
        model.train()
        imgs = torch.rand(4, 3, 128, 128)
        result = model(imgs, return_logits=True)
        bboxes, _, _, cls = result
        assert len(result) == 4
        assert bboxes.shape == (4, bboxes.shape[1], 4)
        assert cls.shape == (4, cls.shape[1], 2)


class TestSyntheticData:
    def test_shapes_dataset(self):
        ds = ShapesDataset(50, 128, 2, seed=99)
        assert len(ds) == 50
        img, target = ds[0]
        assert img.shape == (3, 128, 128)
        assert target["bboxes"].shape == (1, 4)
        assert target["labels"].shape == (1,)

    def test_collate_fn(self):
        ds = ShapesDataset(50, 128, 2, seed=99)
        batch = [ds[i] for i in range(8)]
        imgs, targets = collate_fn(batch)
        assert imgs.shape == (8, 3, 128, 128)
        assert len(targets) == 8

    def test_best_iou_idx(self):
        pred_boxes = torch.tensor([
            [[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 60.0, 60.0]],
            [[5.0, 5.0, 15.0, 15.0], [80.0, 80.0, 90.0, 90.0]],
        ])
        gt_boxes = torch.tensor([[12.0, 12.0, 22.0, 22.0], [82.0, 82.0, 92.0, 92.0]])
        idx = best_iou_idx(pred_boxes, gt_boxes)
        assert idx[0] == 0
        assert idx[1] == 1
