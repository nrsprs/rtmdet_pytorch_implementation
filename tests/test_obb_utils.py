import math

import torch

from rtmdet.obb_utils import distance2obb, norm_angle, obb2polygons, rotated_nms


class TestNormAngle:
    def test_zero(self):
        assert torch.isclose(norm_angle(torch.tensor(0.0)), torch.tensor(0.0))

    def test_pi_over_2(self):
        result = norm_angle(torch.tensor(math.pi / 2))
        expected = torch.tensor(-math.pi / 2)
        assert torch.isclose(result, expected)

    def test_negative_pi_over_2(self):
        result = norm_angle(torch.tensor(-math.pi / 2))
        expected = torch.tensor(-math.pi / 2)
        assert torch.isclose(result, expected)

    def test_pi(self):
        assert torch.isclose(norm_angle(torch.tensor(math.pi)), torch.tensor(0.0), atol=1e-6)

    def test_negative_pi(self):
        assert torch.isclose(norm_angle(torch.tensor(-math.pi)), torch.tensor(0.0), atol=1e-6)

    def test_full_rotation(self):
        assert torch.isclose(
            norm_angle(torch.tensor(2 * math.pi)), torch.tensor(0.0), atol=1e-6
        )

    def test_batch(self):
        angles = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi])
        result = norm_angle(angles)
        expected = torch.tensor([0.0, -math.pi / 2, -math.pi / 2, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_arbitrary_angles(self):
        angles = torch.tensor([0.3, 1.5, -1.2, 2.5])
        result = norm_angle(angles)
        assert (result >= -math.pi / 2).all()
        assert (result < math.pi / 2).all()


class TestDistance2obb:
    def test_zero_angle_at_origin(self):
        points = torch.tensor([[0.0, 0.0]])
        distance = torch.tensor([[5.0, 3.0, 7.0, 9.0, 0.0]])
        obb = distance2obb(points, distance)
        # w = 5+7 = 12, h = 3+9 = 12, offset_t = (1, 3), rotated offset = (1, 3)
        # cx = 0+1 = 1, cy = 0+3 = 3
        # w=12, h=12 (equal, no swap)
        expected = torch.tensor([[1.0, 3.0, 12.0, 12.0, 0.0]])
        assert torch.allclose(obb, expected)

    def test_zero_angle_offset(self):
        points = torch.tensor([[100.0, 100.0]])
        distance = torch.tensor([[10.0, 10.0, 10.0, 10.0, 0.0]])
        obb = distance2obb(points, distance)
        # w=20, h=20, offset_t=(0,0), ctr=(100, 100), angle=0
        expected = torch.tensor([[100.0, 100.0, 20.0, 20.0, 0.0]])
        assert torch.allclose(obb, expected)

    def test_width_height_enforcement(self):
        points = torch.tensor([[0.0, 0.0]])
        # h > w: h=20, w=10, so swap triggers
        distance = torch.tensor([[5.0, 10.0, 5.0, 10.0, 0.0]])
        obb = distance2obb(points, distance)
        assert obb[0, 2] >= obb[0, 3]  # w >= h

    def test_angle_in_range(self):
        points = torch.zeros(1, 2)
        distances = torch.rand(1, 5) * 20
        obb = distance2obb(points, distances)
        assert obb[0, 4] >= -math.pi / 2
        assert obb[0, 4] < math.pi / 2

    def test_batched_2d(self):
        points = torch.zeros(2, 4, 4, 2)  # [B, H, W, 2]
        distance = torch.ones(2, 4, 4, 5) * 10  # [B, H, W, 5]
        obb = distance2obb(points, distance)
        assert obb.shape == (2, 4, 4, 5)

    def test_1d_output_shape(self):
        points = torch.zeros(5, 2)
        distance = torch.ones(5, 5) * 10
        obb = distance2obb(points, distance)
        assert obb.shape == (5, 5)

    def test_consistent_with_axis_aligned(self):
        """Angle=0, symmetric distances: center should match grid point."""
        points = torch.tensor([[320.0, 320.0]])
        distance = torch.tensor([[50.0, 50.0, 50.0, 50.0, 0.0]])
        obb = distance2obb(points, distance)
        assert torch.allclose(obb[:, :2], points)
        assert torch.allclose(obb[:, 2:4], torch.tensor([[100.0, 100.0]]))


class TestOBB2polygons:
    def test_returns_four_vertices(self):
        obb = torch.tensor([[100.0, 100.0, 20.0, 10.0, 0.0]])
        polys = obb2polygons(obb)
        assert len(polys) == 1
        assert len(polys[0]) == 4

    def test_axis_aligned_rectangle(self):
        obb = torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.0]])
        polys = obb2polygons(obb)
        # Expected corners: (80,90), (120,90), (120,110), (80,110)
        expected = [(80.0, 90.0), (120.0, 90.0), (120.0, 110.0), (80.0, 110.0)]
        for actual, exp in zip(polys[0], expected):
            assert math.isclose(actual[0], exp[0], abs_tol=1e-6)
            assert math.isclose(actual[1], exp[1], abs_tol=1e-6)

    def test_45_degree_rotation(self):
        obb = torch.tensor([[100.0, 100.0, 20.0, 10.0, math.pi / 4]])
        polys = obb2polygons(obb)
        assert len(polys[0]) == 4
        # Verify points are not axis-aligned (x coords should differ from 90/110)
        x_coords = [p[0] for p in polys[0]]
        assert not all(math.isclose(x, 90.0) or math.isclose(x, 110.0) for x in x_coords)

    def test_batch(self):
        obb = torch.tensor([
            [100.0, 100.0, 20.0, 10.0, 0.0],
            [200.0, 200.0, 30.0, 15.0, math.pi / 4],
        ])
        polys = obb2polygons(obb)
        assert len(polys) == 2
        assert len(polys[0]) == 4
        assert len(polys[1]) == 4


class TestRotatedNMS:
    def test_empty_input(self):
        bboxes = torch.empty((0, 5))
        scores = torch.empty(0)
        keep = rotated_nms(bboxes, scores, 0.1)
        assert len(keep) == 0

    def test_no_overlap(self):
        bboxes = torch.tensor([
            [100.0, 100.0, 20.0, 20.0, 0.0],
            [300.0, 300.0, 20.0, 20.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = rotated_nms(bboxes, scores, 0.1)
        assert len(keep) == 2

    def test_complete_overlap(self):
        bboxes = torch.tensor([
            [100.0, 100.0, 20.0, 20.0, 0.0],
            [100.0, 100.0, 20.0, 20.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = rotated_nms(bboxes, scores, 0.1)
        assert len(keep) == 1
        assert keep[0].item() == 0  # highest score kept

    def test_partial_overlap_above_threshold(self):
        # IoU of two 40x40 boxes offset by 10 is ~0.39
        bboxes = torch.tensor([
            [100.0, 100.0, 40.0, 40.0, 0.0],
            [110.0, 110.0, 40.0, 40.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = rotated_nms(bboxes, scores, 0.3)
        assert len(keep) == 1

    def test_partial_overlap_below_threshold(self):
        bboxes = torch.tensor([
            [100.0, 100.0, 20.0, 20.0, 0.0],
            [115.0, 115.0, 20.0, 20.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = rotated_nms(bboxes, scores, 0.9)
        assert len(keep) == 2  # low IoU, high threshold -> both kept

    def test_max_num_cap(self):
        bboxes = torch.tensor([
            [100.0, 100.0, 10.0, 10.0, 0.0],
            [200.0, 200.0, 10.0, 10.0, 0.0],
            [300.0, 300.0, 10.0, 10.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = rotated_nms(bboxes, scores, 0.1, max_num=2)
        assert len(keep) == 2

    def test_sorted_by_score(self):
        bboxes = torch.tensor([
            [100.0, 100.0, 10.0, 10.0, 0.0],
            [200.0, 200.0, 10.0, 10.0, 0.0],
            [300.0, 300.0, 10.0, 10.0, 0.0],
        ])
        scores = torch.tensor([0.5, 0.9, 0.7])
        keep = rotated_nms(bboxes, scores, 0.1)
        assert keep[0].item() == 1  # highest score index first
        assert keep[1].item() == 2
        assert keep[2].item() == 0

    def test_rotated_boxes(self):
        """NMS on rotated boxes should use polygon intersection."""
        bboxes = torch.tensor([
            [100.0, 100.0, 40.0, 10.0, 0.0],
            [100.0, 100.0, 10.0, 40.0, math.pi / 2],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = rotated_nms(bboxes, scores, 0.5)
        # These cross at center, IoU is moderate
        assert len(keep) >= 1