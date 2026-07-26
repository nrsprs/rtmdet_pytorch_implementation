import torch

from rtmdet.utils import apply_factor, distance2bbox, generate_grid_centers


class TestApplyFactor:
    def test_single_int(self):
        assert apply_factor(128, 0.5) == 64
        assert apply_factor(10, 2.0) == 20

    def test_single_int_rounding(self):
        assert apply_factor(7, 0.375) == 3  # round(2.625) = 3

    def test_min_one_floor(self):
        assert apply_factor(1, 0.001) == 1

    def test_identity_factor(self):
        assert apply_factor(128, 1.0) == 128

    def test_list_ints(self):
        assert apply_factor([32, 64, 128], 0.5) == [16, 32, 64]

    def test_list_min_one(self):
        assert apply_factor([1, 2, 3], 0.1) == [1, 1, 1]


class TestGenerateGridCenters:
    def test_grid_centers_length(self):
        centers = generate_grid_centers(640, [8, 16, 32], torch.device("cpu"))
        assert len(centers) == 3

    def test_grid_centers_shape_stride8(self):
        centers = generate_grid_centers(640, [8], torch.device("cpu"))
        cx, cy = centers[0]
        assert cx.shape == (80, 80)
        assert cy.shape == (80, 80)

    def test_grid_centers_shape_stride16(self):
        centers = generate_grid_centers(640, [16], torch.device("cpu"))
        cx, _ = centers[0]
        assert cx.shape == (40, 40)

    def test_grid_centers_shape_stride32(self):
        centers = generate_grid_centers(640, [32], torch.device("cpu"))
        cx, _ = centers[0]
        assert cx.shape == (20, 20)

    def test_grid_centers_values(self):
        centers = generate_grid_centers(640, [8], torch.device("cpu"))
        cx, cy = centers[0]
        assert cx[0, 0] == 4.0  # 0.5 * 8
        assert cx[0, 1] == 12.0  # 1.5 * 8
        assert cy[1, 0] == 12.0  # 1.5 * 8

    def test_grid_centers_device(self):
        centers = generate_grid_centers(640, [8], torch.device("cpu"))
        cx, _ = centers[0]
        assert cx.device == torch.device(torch.device("cpu"))

    def test_smaller_image(self):
        centers = generate_grid_centers(256, [8], torch.device("cpu"))
        cx, _ = centers[0]
        assert cx.shape == (32, 32)


class TestDistance2bbox:
    def test_basic_1d(self):
        points = torch.tensor([[10.0, 10.0]])  # [1, 2]
        distance = torch.tensor([[5.0, 3.0, 7.0, 9.0]])  # [1, 4]
        bbox = distance2bbox(points, distance)
        # x1=10-5=5, y1=10-3=7, x2=10+7=17, y2=10+9=19
        expected = torch.tensor([[5.0, 7.0, 17.0, 19.0]])
        assert torch.allclose(bbox, expected)

    def test_multiple_points(self):
        points = torch.tensor([[10.0, 10.0], [20.0, 20.0]])  # [2, 2]
        distance = torch.tensor([[5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 10.0, 10.0]])  # [2, 4]
        bbox = distance2bbox(points, distance)
        expected = torch.tensor([[5.0, 5.0, 15.0, 15.0], [10.0, 10.0, 30.0, 30.0]])
        assert torch.allclose(bbox, expected)

    def test_batched_2d(self):
        points = torch.zeros(2, 4, 4, 2)  # [B, H, W, 2]
        distance = torch.ones(2, 4, 4, 4)  # [B, H, W, 4]
        bbox = distance2bbox(points, distance)
        assert bbox.shape == (2, 4, 4, 4)

    def test_no_clamping(self):
        points = torch.tensor([[0.0, 0.0]])
        distance = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        bbox = distance2bbox(points, distance)
        # x1 = -10, y1 = -10 (no clamping)
        assert bbox[0, 0] < 0
        assert bbox[0, 1] < 0

    def test_clamping_x(self):
        points = torch.tensor([[650.0, 320.0]])
        distance = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        bbox = distance2bbox(points, distance, (640, 640))
        # x2 = 660, clamped to 640
        assert bbox[0, 2] == 640.0

    def test_clamping_y(self):
        points = torch.tensor([[320.0, 650.0]])
        distance = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        bbox = distance2bbox(points, distance, (640, 640))
        # y2 = 660, clamped to 640
        assert bbox[0, 3] == 640.0

    def test_asymmetric_max_shape(self):
        points = torch.tensor([[400.0, 300.0]])
        distance = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        bbox = distance2bbox(points, distance, (480, 640))
        # x clamped to [0, 640], y clamped to [0, 480]
        assert bbox[0, 2] == 600.0  # 400+200, within 640
        assert bbox[0, 3] == 480.0  # 300+200=500, clamped to 480
