#!/usr/bin/env python

from AR10_class import AR10
import time
import sys

def calibrate_joint(hand, joint):
    n_points = 0

    sum_x  = 0.0
    sum_y  = 0.0
    sum_xy = 0.0
    sum_xx = 0.0

    if joint == 9:
        hand.move(8, 5500)

    for target in range(4500, 8000, 500):
        hand.move(joint, target)
        hand.wait_for_hand()
        time.sleep(2.0)

        position = hand.get_read_position(joint)

        n_points = n_points + 1

        sum_x  = sum_x + position
        sum_y  = sum_y + target
        sum_xy = sum_xy + (position * target)
        sum_xx = sum_xx + (position * position)

    slope       = ((sum_x * sum_y) - (n_points * sum_xy)) / ((sum_x * sum_x) - (n_points * sum_xx))
    y_intercept = (sum_y - (slope * sum_x)) / n_points

    hand.move(joint, 7950)
    hand.wait_for_hand()

    if joint == 9:
        hand.move(8, 7950)
        hand.wait_for_hand()

    return y_intercept, slope

def main():
    # create hand object
    hand = AR10()

    # open calibration file
    cal_file = open("calibration_file", "w")

    hand.open_hand()

    for joint in range(0, 10):
        y_intercept, slope = calibrate_joint(hand, joint)
        print(("joint = " + str(joint) + " y intercept = " + str(y_intercept) + " slope = " + str(slope)))
        cal_file.write(str(joint) + "\t" + str(y_intercept) + "\t" + str(slope) + "\n")

    # close calibration file
    cal_file.close()

    # destroy hand object
    hand.close()

if __name__ == "__main__":
    main()



