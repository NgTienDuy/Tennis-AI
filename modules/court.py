import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import Line
from itertools import combinations

class CourtReference:
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)

        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1], self.right_inner_line[1]],
            3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1], self.right_court_line[1]],
            4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1], self.right_inner_line[1]],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1], self.right_court_line[1]],
            9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1], self.left_inner_line[1]],
            10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0], self.middle_line[1]],
            11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1], self.bottom_inner_line[1]],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]
        }
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        
        self.court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        cv2.line(self.court, *self.baseline_top, 1, 1)
        cv2.line(self.court, *self.baseline_bottom, 1, 1)
        cv2.line(self.court, *self.top_inner_line, 1, 1)
        cv2.line(self.court, *self.bottom_inner_line, 1, 1)
        cv2.line(self.court, *self.left_court_line, 1, 1)
        cv2.line(self.court, *self.right_court_line, 1, 1)
        cv2.line(self.court, *self.left_inner_line, 1, 1)
        cv2.line(self.court, *self.right_inner_line, 1, 1)
        cv2.line(self.court, *self.middle_line, 1, 1)
        self.court = cv2.dilate(self.court, np.ones((5, 5), dtype=np.uint8))

    def get_court_mask(self, mask_type=0):
        mask = np.ones_like(self.court)
        if mask_type == 1: mask[:self.net[0][1] - 1000, :] = 0
        elif mask_type == 2: mask[self.net[0][1]:, :] = 0
        return mask

    def get_important_lines(self):
        return [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                *self.left_inner_line, *self.right_inner_line, *self.middle_line,
                *self.top_inner_line, *self.bottom_inner_line]

    def get_extra_parts(self):
        return [self.top_extra_part, self.bottom_extra_part]

class CourtDetector:
    def __init__(self, config=None, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        
        if config:
            self.colour_threshold = config['court']['colour_threshold']
            self.dist_tau = config['court']['dist_tau']
            self.intensity_threshold = config['court']['intensity_threshold']

        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.best_conf = None
        self.frame_points = None
        self.dist = 5
        self.success_accuracy = 80
        self.success_score = 1000
        self.court_accuracy = 0

    def detect(self, frame):
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.threshold(self.gray, self.colour_threshold, 255, cv2.THRESH_BINARY)[1]

        filtered = self._filter_pixels(self.gray)
        horizontal_lines, vertical_lines = self._detect_lines(filtered)
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal_lines, vertical_lines)
        
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        
        self.court_accuracy = self._get_court_accuracy()
        if self.verbose:
            print(f"Court Accuracy: {self.court_accuracy:.2f}, Score: {self.court_score}")

    def _filter_pixels(self, gray):
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0: continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and
                        gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue
                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and
                        gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        minLineLength = 100
        maxLineGap = 20
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is None: return [], []
        lines = np.squeeze(lines)
        horizontal, vertical = self._classify_lines(lines)
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        return horizontal, vertical

    def _classify_lines(self, lines):
        horizontal, vertical = [], []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        if len(lines.shape) == 1: lines = [lines]
        for line in lines:
            x1, y1, x2, y2 = line
            dx, dy = abs(x1 - x2), abs(y1 - y2)
            if dx > 2 * dy: horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((0, self.v_height * 6/7), (self.v_width, self.v_height * 6/7)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((0, self.v_height * 6/7), (self.v_width, self.v_height * 6/7)))
                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.court_reference.court_conf.items():
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    if matrix is None: continue
                    inv_matrix = cv2.invert(matrix)[1]
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i

        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        court = cv2.warpPerspective(self.court_reference.court, matrix, (self.v_width, self.v_height))
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def _get_court_accuracy(self):
        if len(self.court_warp_matrix) == 0: return 0
        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], (self.v_width, self.v_height))
        court[court > 0] = 1
        gray = cv2.dilate(self.gray, np.ones((9, 9), dtype=np.uint8))
        gray[gray > 0] = 1
        # [FIX OVERFLOW] Sử dụng np.sum thay vì sum(sum()) để tránh tràn số int32
        total_white_pixels = np.sum(court)
        sub = court.copy()
        sub[gray == 1] = 0
        if total_white_pixels == 0: return 0
        return 100 - (np.sum(sub) / total_white_pixels) * 100

    def add_court_overlay(self, frame, overlay_color=(255, 255, 255), frame_num=-1):
        if len(self.court_warp_matrix) == 0: return frame
        if frame_num >= len(self.court_warp_matrix): frame_num = -1
        homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, (frame.shape[1], frame.shape[0]))
        frame[court > 0, :] = overlay_color
        return frame

    def delete_extra_parts(self, frame, frame_num=-1):
        if len(self.court_warp_matrix) == 0: return frame
        if frame_num >= len(self.court_warp_matrix): frame_num = -1
        img = frame.copy()
        parts = np.array(self.court_reference.get_extra_parts(), dtype=np.float32).reshape((-1, 1, 2))
        parts = cv2.perspectiveTransform(parts, self.court_warp_matrix[frame_num]).reshape(-1)
        top, bottom = parts[:2], parts[2:]
        img[int(bottom[1]-10):int(bottom[1]+10), int(bottom[0]-15):int(bottom[0]+15), :] = (0,0,0)
        img[int(top[1]-10):int(top[1]+10), int(top[0]-15):int(top[0]+15), :] = (0,0,0)
        return img

    def track_court(self, frame):
        if len(self.court_warp_matrix) > 0:
            self.court_warp_matrix.append(self.court_warp_matrix[-1])
            self.game_warp_matrix.append(self.game_warp_matrix[-1])
        else:
            self.detect(frame)

def sort_intersection_points(intersections):
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34

def line_intersection(line1, line2):
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])
    intersection = l1.intersection(l2)
    if intersection:
        return intersection[0].coordinates
    return (0, 0)