#!/usr/bin/env python
# coding: utf-8

import time
from pathlib import Path
import numpy as np
import cv2
import PySimpleGUI as sg


def file_read():
    fp = ""
    layout = [
        [
            sg.FileBrowse(key="file"),
            sg.Text("File"),
            sg.InputText()
        ],
        [sg.Submit(key="submit"), sg.Cancel("Exit")]
    ]
    window = sg.Window("Select Files", layout)

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values[0] == "":
                sg.popup("No input.")
                event = ""
            else:
                fp = values[0]
                break
    window.close()
    return Path(fp)


def hsv(frame, H_max, H_min, S_max, S_min, V_max, V_min, reverse=False):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if reverse:
        lower1 = np.array([0, int(S_min), int(V_min)])
        upper1 = np.array([int(H_min), int(S_max), int(V_max)])
        mask1 = cv2.inRange(frame_hsv, lower1, upper1)
        lower2 = np.array([int(H_max), int(S_min), int(V_min)])
        upper2 = np.array([255, int(S_max), int(V_max)])
        mask2 = cv2.inRange(frame_hsv, lower2, upper2)
        mask = mask1 + mask2
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        # mask = cv2.bitwise_and(frame, mask, mask=mask)
    else:
        lower = np.array([int(H_min), int(S_min), int(V_min)])
        upper = np.array([int(H_max), int(S_max), int(V_max)])
        mask = cv2.inRange(frame_hsv, lower, upper)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame


def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x / (b))
    avg_y = int(avg_y / (b))
    avg_r = int(avg_r / (b))
    return avg_x, avg_y, avg_r


def dist_2_pts(x1, y1, x2, y2):
    # print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calibrate_gauge(img):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array(
        []), 100, 50, int(height*0.35), int(height*0.48))
    a, b, c = circles.shape
    x, y, r = avg_circles(circles, b)

    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)

    separation = 10.0
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p1[i][j] = x + 0.9 * r * \
                    np.cos(separation * i * 3.14 / 180)  # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * \
                    np.cos((separation) * (i+9) * 3.14 / 180)
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2 * r * \
                    np.sin((separation) * (i+9) * 3.14 / 180)
    for i in range(0, interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])),
                 (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (int(i*separation)), (int(p_text[i][0]), int(
            p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    # min_angle = input( 'Min angle (lowest possible angle of dial) - in degrees: ')
    # max_angle = input('Max angle (highest possible angle) - in degrees: ')
    # min_value = input('Min value: ')
    # max_value = input('Max value: ')
    # units = input('Enter units: ')
    min_angle = 45
    max_angle = 320
    min_value = 0
    max_value = 200
    units = "PSI"

    # get_current_value
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 175
    maxValue = 255
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    minLineLength = 10
    maxLineGap = 0

    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180,
                            threshold=100, minLineLength=minLineLength, maxLineGap=0)
    final_line_list = []

    diff1LowerBound = 0.15
    diff1UpperBound = 0.25
    diff2LowerBound = 0.5
    diff2UpperBound = 1.0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)
            diff2 = dist_2_pts(x, y, x2, y2)

            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp

            if (((diff1 < diff1UpperBound*r) and (diff1 > diff1LowerBound*r) and (diff2 < diff2UpperBound*r)) and (diff2 > diff2LowerBound*r)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                final_line_list.append([x1, y1, x2, y2])

    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    # np.rad2deg(res) #coverts to degrees

    # these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  # in 四分円 I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res

    # print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


class Main:
    def __init__(self):
        self.fp = file_read()
        self.cap = cv2.VideoCapture(str(self.fp))
        self.rec_flg = False
        self.ret, self.f_frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.org_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.width = self.org_width
            self.org_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.height = self.org_height
            self.total_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.mask = np.zeros_like(self.f_frame[:, :, 0])
            self.frame_count = 0
            self.s_frame = 0
            self.e_frame = self.total_count
            self.stop_flg = False
            self.event = ""
            cv2.namedWindow("Movie")
            cv2.setMouseCallback("Movie", self.onMouse)
        else:
            sg.Popup("Failed to read the file.")
            return

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            hsv = cv2.cvtColor(
                self.frame[y:y + 1, x:x + 1, :], cv2.COLOR_BGR2HSV)
            h = int(hsv[:, :, 0])
            h_min = max(h - 20, 0)
            h_max = min(255, h + 20)
            s = int(hsv[:, :, 1])
            s_min = max(s - 20, 0)
            s_max = min(255, s + 20)
            v = int(hsv[:, :, 2])
            v_min = max(v - 20, 0)
            v_max = min(255, v + 20)
            self.window['-H_MIN SLIDER_MASK-'].update(h_min)
            self.window['-H_MAX SLIDER_MASK-'].update(h_max)
            self.window['-S_MIN SLIDER_MASK-'].update(s_min)
            self.window['-S_MAX SLIDER_MASK-'].update(s_max)
            self.window['-V_MIN SLIDER_MASK-'].update(v_min)
            self.window['-V_MAX SLIDER_MASK-'].update(v_max)
            self.window['-Hue Reverse_MASK-'].update(False)

    def run(self):
        layout = [
            [
                sg.Text("Start", size=(8, 1)),
                sg.Slider(
                    (0, self.total_count - 1),
                    0,
                    1,
                    orientation='h',
                    size=(45, 15),
                    key='-START FRAME SLIDER-',
                    enable_events=True
                )
            ],
            [sg.Slider(
                (0, self.total_count - 1),
                0,
                1,
                orientation='h',
                size=(50, 15),
                key='-PROGRESS SLIDER-',
                enable_events=True
            )],
            [sg.HorizontalSeparator()],
            [
                sg.Text("Resize     ", size=(13, 1)),
                sg.Slider(
                    (0.1, 4),
                    1,
                    0.01,
                    orientation='h',
                    size=(40, 15),
                    key='-RESIZE SLIDER-',
                    enable_events=True
                )
            ],
            [
                sg.Checkbox(
                    "Inpaint",
                    size=(10, 1),
                    default=False,
                    key='-INPAINT-',
                    enable_events=True
                )
            ],
            [
                sg.Checkbox(
                    "Opening",
                    size=(20, 1),
                    default=False,
                    key='-OPENING-',
                    enable_events=True
                ),
                sg.Checkbox(
                    "Closing",
                    size=(20, 1),
                    default=False,
                    key='-CLOSING-',
                    enable_events=True
                ),
                sg.Slider(
                    (3, 31),
                    5,
                    2,
                    orientation='h',
                    size=(15, 15),
                    key='-OPENING SLIDER-',
                    enable_events=True
                )
            ],
            [
                sg.Checkbox(
                    "Dilation",
                    size=(10, 1),
                    default=False,
                    key='-DILATION-',
                    enable_events=True
                ),
                sg.Slider(
                    (1, 31),
                    4,
                    2,
                    orientation='h',
                    size=(15, 15),
                    key='-DILATION SLIDER-',
                    enable_events=True
                )
            ],
            [
                sg.Checkbox(
                    'blur',
                    size=(10, 1),
                    key='-BLUR-',
                    enable_events=True
                ),
                sg.Slider(
                    (1, 10),
                    1,
                    1,
                    orientation='h',
                    size=(40, 15),
                    key='-BLUR SLIDER-',
                    enable_events=True
                )
            ],
            [
                sg.Text(
                    'hsv',
                    size=(10, 1),
                    key='-HSV_MASK-',
                    enable_events=True
                ),
                sg.Button('Blue', size=(10, 1)),
                sg.Button('Green', size=(10, 1)),
                sg.Button('Red', size=(10, 1))
            ],
            [
                sg.Checkbox(
                    'Hue Reverse',
                    size=(10, 1),
                    key='-Hue Reverse_MASK-',
                    enable_events=True
                )
            ],
            [
                sg.Text('Hue', size=(10, 1), key='-Hue_MASK-'),
                sg.Slider(
                    (0, 255),
                    0,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-H_MIN SLIDER_MASK-',
                    enable_events=True
                ),
                sg.Slider(
                    (1, 255),
                    125,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-H_MAX SLIDER_MASK-',
                    enable_events=True
                )
            ],
            [
                sg.Text('Saturation', size=(10, 1), key='-Saturation_MASK-'),
                sg.Slider(
                    (0, 255),
                    50,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-S_MIN SLIDER_MASK-',
                    enable_events=True
                ),
                sg.Slider(
                    (1, 255),
                    255,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-S_MAX SLIDER_MASK-',
                    enable_events=True
                )
            ],
            [
                sg.Text('Value', size=(10, 1), key='-Value_MASK-'),
                sg.Slider(
                    (0, 255),
                    50,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-V_MIN SLIDER_MASK-',
                    enable_events=True
                ),
                sg.Slider(
                    (1, 255),
                    255,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-V_MAX SLIDER_MASK-',
                    enable_events=True
                )
            ],
            [sg.Output(size=(65, 5), key='-OUTPUT-')],
            [sg.Button('Clear')]
        ]
        self.window = sg.Window('OpenCV Integration', layout, location=(0, 0))
        self.event, values = self.window.read(timeout=0)
        print("The file has been read.")
        print("File Path: " + str(self.fp))
        print("fps: " + str(int(self.fps)))
        print("width: " + str(self.width))
        print("height: " + str(self.height))
        print("frame count: " + str(int(self.total_count)))

        try:
            while True:
                self.event, values = self.window.read(
                    timeout=0
                )
                if self.event != "__TIMEOUT__":
                    print(self.event)
                if self.event in ('Exit', sg.WIN_CLOSED, None):
                    break
                if self.event == 'Reset':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    self.window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.video_stabilization_flg = False
                    self.stab_prepare_flg = False
                    continue
                if self.event == '-PROGRESS SLIDER-':
                    self.frame_count = int(values['-PROGRESS SLIDER-'])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                if self.event == '-START FRAME SLIDER-':
                    self.s_frame = int(values['-START FRAME SLIDER-'])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    self.window['-PROGRESS SLIDER-'].update(self.frame_count)
                if self.frame_count >= self.e_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    self.window['-PROGRESS SLIDER-'].update(self.frame_count)
                    continue
                if self.event == 'Play / Stop':
                    self.stop_flg = not self.stop_flg
                if(
                    (
                        self.stop_flg
                        and self.event == "__TIMEOUT__"
                    )
                ):
                    self.window['-PROGRESS SLIDER-'].update(self.frame_count)
                    continue
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    continue
                self.width = int(self.org_width * values['-RESIZE SLIDER-'])
                self.height = int(self.org_height * values['-RESIZE SLIDER-'])
                self.frame = cv2.resize(self.frame, (self.width, self.height))

                if self.event == 'Blue':
                    self.window['-H_MIN SLIDER_MASK-'].update(70)
                    self.window['-H_MAX SLIDER_MASK-'].update(110)
                    self.window['-S_MIN SLIDER_MASK-'].update(70)
                    self.window['-S_MAX SLIDER_MASK-'].update(255)
                    self.window['-V_MIN SLIDER_MASK-'].update(0)
                    self.window['-V_MAX SLIDER_MASK-'].update(255)
                    self.window['-Hue Reverse_MASK-'].update(False)
                if self.event == 'Green':
                    self.window['-H_MIN SLIDER_MASK-'].update(20)
                    self.window['-H_MAX SLIDER_MASK-'].update(70)
                    self.window['-S_MIN SLIDER_MASK-'].update(70)
                    self.window['-S_MAX SLIDER_MASK-'].update(255)
                    self.window['-V_MIN SLIDER_MASK-'].update(0)
                    self.window['-V_MAX SLIDER_MASK-'].update(255)
                    self.window['-Hue Reverse_MASK-'].update(False)
                if self.event == 'Red':
                    self.window['-H_MIN SLIDER_MASK-'].update(20)
                    self.window['-H_MAX SLIDER_MASK-'].update(110)
                    self.window['-S_MIN SLIDER_MASK-'].update(70)
                    self.window['-S_MAX SLIDER_MASK-'].update(255)
                    self.window['-V_MIN SLIDER_MASK-'].update(0)
                    self.window['-V_MAX SLIDER_MASK-'].update(255)
                    self.window['-Hue Reverse_MASK-'].update(True)

                self.mask = self.frame
                self.mask = hsv(
                    self.mask,
                    values['-H_MAX SLIDER_MASK-'],
                    values['-H_MIN SLIDER_MASK-'],
                    values['-S_MAX SLIDER_MASK-'],
                    values['-S_MIN SLIDER_MASK-'],
                    values['-V_MAX SLIDER_MASK-'],
                    values['-V_MIN SLIDER_MASK-'],
                    values['-Hue Reverse_MASK-']
                )
                self.mask = cv2.cvtColor(
                    self.mask,
                    cv2.COLOR_BGR2GRAY
                )
                if values['-OPENING-']:
                    self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN,
                                                 np.ones((int(values['-OPENING SLIDER-']), int(values['-OPENING SLIDER-'])), np.uint8))
                if values['-CLOSING-']:
                    self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE,
                                                 np.ones((int(values['-OPENING SLIDER-']), int(values['-OPENING SLIDER-'])), np.uint8))
                if values['-DILATION-']:
                    self.mask = cv2.dilate(self.mask,
                                           np.ones((int(values['-DILATION SLIDER-']), int(values['-DILATION SLIDER-'])), np.uint8), iterations=1)
                if values['-INPAINT-']:
                    self.frame = cv2.inpaint(
                        self.frame,
                        self.mask,
                        2,
                        cv2.INPAINT_TELEA
                    )
                if values['-BLUR-']:
                    self.frame_roi = cv2.GaussianBlur(
                        self.frame, (21, 21), values['-BLUR SLIDER-']
                    )

                    # frame内にマスクを適用
                    # マスク処理部のみ.frameに変える
                    self.frame = cv2.bitwise_not(
                        cv2.bitwise_not(self.frame_roi),
                        self.frame,
                        mask=self.mask
                    )
                cv2.putText(self.frame,
                            str("framecount: {0:.0f}".format(
                                self.frame_count)),
                            (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 230, 0), 1, cv2.LINE_AA)
                cv2.putText(self.frame,
                            str("time: {0:.1f} sec".format(
                                self.frame_count / self.fps)),
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 230, 0), 1, cv2.LINE_AA)
                cv2.putText(self.frame, str("Gauge Value: {0:.3f} Units".format(calibrate_gauge(self.frame))), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 230, 0), 1, cv2.LINE_AA)

                cv2.imshow("Movie", self.frame)
                cv2.imshow("Mask", cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR))

                if self.stop_flg:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                else:
                    self.frame_count += 1
                    self.window['-PROGRESS SLIDER-'].update(
                        self.frame_count + 1)
                if self.event == 'Clear':
                    self.window['-OUTPUT-'].update('')
        finally:
            cv2.destroyWindow("Movie")
            cv2.destroyWindow("Mask")
            self.cap.release()
            self.window.close()


if __name__ == '__main__':
    Main().run()
