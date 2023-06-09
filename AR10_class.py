
"""
Active8 Robots, AR10 hand class.

Includes pololu serial communication, speed, acceleration
and movement together with demonstration facilities.
"""

import time
import sys
import random
import serial
import csv

class AR10:
    def __init__(self):
        self.speed        = 50
        self.acceleration = 30

        self.intercept = []
        self.slope     = []

        # When connected via USB, the Maestro creates two virtual serial ports
        # /dev/ttyACM0 for commands and /dev/ttyACM1 for communications.
        # Be sure the Maestro is configured for "USB Dual Port" serial mode.
        # "USB Chained Mode" may work as well, but hasn't been tested.

        # Pololu protocol allows for multiple Maestros to be connected. A device
        # number is used to index each connected unit.  This code currently is statically
        # configured to work with the default device 0x0C (or 12 in decimal).

        # Open the command port
        self.usb = serial.Serial('/dev/ttyACM1', baudrate=9600)
        # Command lead-in and device 12 are sent for each Pololu serial commands.
        self.pololu_command = chr(0xaa) + chr(0xc)

        # Read the calibration file
        try:
            cal_file = csv.reader(open("calibration_file"), delimiter='\t')
            for row in cal_file:
                self.intercept.append(float(row[1]))
                self.slope.append(float(row[2]))
        except IOError:
            print("Calibration file missing")
            print("Please run AR10_calibrate.py")

    # Cleanup by closing USB serial port
    def close(self):
        self.usb.close()

    # Change speed setting
    def change_speed(self, speed):
        self.speed = speed

    # Set speed of channel
    def set_speed(self, channel):
        lsb = self.speed & 0x7f                      #7 bits for least significant byte
        msb = (self.speed >> 7) & 0x7f               #shift 7 and take next 7 bits for msb
        # Send Pololu intro, device number, command, channel, speed lsb, speed msb
        command = self.pololu_command + chr(0x07) + chr(channel) + chr(lsb) + chr(msb)
        self.usb.write(command.encode())

    # Change acceleration setting
    def change_acceleration(self, acceleration):
        self.acceleration = acceleration

    # Set acceleration of channel
    # This provide soft starts and finishes when servo moves to target position.
    def set_acceleration(self, channel, acceleration):
        lsb = acceleration & 0x7f                      # 7 bits for least significant byte
        msb = (acceleration >> 7) & 0x7f               # shift 7 and take next 7 bits for msb
        # Send Pololu intro, device number, command, channel, acceleration lsb, acceleration msb
        command = self.pololu_command + chr(0x09) + chr(channel) + chr(lsb) + chr(msb)
        self.usb.write(command.encode())

    # Set channel to a specified target value
    def set_target(self, channel, target):
        lsb = target & 0x7f                      # 7 bits for least significant byte
        msb = (target >> 7) & 0x7f               # shift 7 and take next 7 bits for msb
        # Send Pololu intro, device number, command, channel, and target lsb/msb
        command = self.pololu_command + chr(0x04) + chr(channel) + chr(lsb) + chr(msb)
        self.usb.write(command.encode())

    # convert joint number to channel number
    def joint_to_channel(self, joint):
        channel = joint + 10
        return channel

    # Get the current position of the device on the specified channel
    # The result is returned in a measure of quarter-microseconds, which mirrors
    # the Target parameter of set_target.
    # This is not reading the true servo position, but the last target position sent
    # to the servo.  If the Speed is set to below the top speed of the servo, then
    # the position result will align well with the acutal servo position, assuming
    # it is not stalled or slowed.
    def get_set_position(self, joint):
        # convert joint to channel
        channel = self.joint_to_channel(joint)

        command = self.pololu_command + chr(0x10) + chr(channel)
        self.usb.write(command.encode('utf-8'))
        lsb = ord(self.usb.read())
        msb = ord(self.usb.read())

        return (msb * 256) + lsb

    # Have servo outputs reached their targets? This is useful only if Speed and/or
    # Acceleration have been set on one or more of the channels.  Returns True or False.
    def get_read_position(self, channel):
        command = self.pololu_command + chr(0x90) + chr(channel)
        self.usb.write(command.encode('utf-8'))
        lsb = ord(self.usb.read())
        msb = ord(self.usb.read())
        read_position = (256 * msb) + lsb

        return read_position

    # Have servo outputs reached their targets? This is useful only if Speed and/or
    # Acceleration have been set on one or more of the channels.  Returns True or False.
    def get_position(self, channel):
        read_position = self.get_read_position(channel)
        position      = self.intercept[channel] + (self.slope[channel] * read_position)

        return int(position)

    # Have servo outputs reached their targets? This is useful only if Speed and/or
    # Acceleration have been set on one or more of the channels.  Returns True or False.
    def get_moving_state(self):
        command = self.pololu_command + chr(0x13)# + chr(0x01)
        self.usb.write(command.encode())
        res = self.usb.read().decode()
        print(ord(res[0]))
        if res[0] == chr(0x00):
            return False
        else:
            return True

    # Run a Maestro Script subroutine in the currently active script.  Scripts can
    # have multiple subroutines, which get numbered sequentially from 0 on up.  Code your
    # Maestro subroutine to either infinitely loop, or just end (return is not valid).
    def run_script(self, subNumber):
        command = self.pololu_command + chr(0x27) + chr(subNumber)
        # can pass a param with comman 0x28
        #  command = self.pololu_command + chr(0x28) + chr(subNumber) + chr(lsb) + chr(msb)
        self.usb.write(command.encode())

    # Stop the current Maestro Script
    def stop_script(self):
        command = self.pololu_command + chr(0x24)
        self.usb.write(command.encode())

    # move joint to target position
    def move(self, joint, target):
        # convert joint to channel
        channel = self.joint_to_channel(joint)

        # check target position is in range
        if target > 8000:
            target = 8000
        elif target < 4000:
            target = 4000

        # a speed of 1 will take 1 minute
        # a speed of 60 would take 1 second.
        # Speed of 0 is unlimited
        # self.set_speed(channel)
        # time.sleep(0.25)

        # Valid values are from 0 to 255. 0=unlimited, 1 is slowest start. 
        # A value of 1 will take the servo about 3s to move between 1ms to 2ms range.
        # self.set_acceleration(channel, self.acceleration)
        # time.sleep(0.25)

        # Valid servo range is 256 to 3904
        self.set_target(channel, target)
        time.sleep(0.005)

    # wait for joints to stop moving
    def wait_for_hand(self):
        # while self.get_moving_state():
        #     time.sleep(0.25)
        time.sleep(0.1)

    # open hand
    def open_hand(self):
        self.move(0, 8000)
        self.move(1, 8000)

        time.sleep(1.0)

        for joint in range(2, 10):
            self.move(joint, 8000)
            # time.sleep(0.25)	
        self.wait_for_hand()

    # close hand
    def close_hand(self):
        # move fingers
        self.move(2, 4000)
        self.move(3, 4000)
        self.move(4, 4000)
        self.move(5, 4000)
        self.move(6, 4000)
        self.move(7, 4000)
        self.move(8, 4000)
        self.move(9, 4000)
        time.sleep(2.0)

        self.move(0, 5000)

        time.sleep(1.0)

        self.move(1, 6500)

        self.wait_for_hand()

    def set_angle(self, joint, angle):
        if joint == 0:
            angle = angle
        if angle > 180:
            angle = 180
        if angle < 90:
            angle = 90
        
        angle -= 90

        target = int((8000-4000)* angle / 90 + 4200)
        self.move(joint, target)

# hold golf ball
    def hold_golf_ball(self):
       # move thumb, second finger and third finger
        self.move(0, 5700)
        self.move(1, 8000)
        self.move(6, 4500)
        self.move(7, 7700)
        self.move(8, 4500)
        self.move(9, 7900)
       

        self.wait_for_hand()

    # hold tennis ball
    def hold_tennis_ball(self):
         # move fingers
        self.move(0, 5500)	
        self.move(1, 8000)        
        self.move(2, 5000)
        self.move(3, 5000)
        self.move(4, 5500)
        self.move(5, 7400)
        self.move(6, 5300)
        self.move(7, 7400)
        self.move(8, 5200)
        self.move(9, 7500)

        self.wait_for_hand()

    # test
    def test(self):
        for pos in range(1500, 2000, 100):
            print(pos)
            self.move(6, pos)
            self.wait_for_hand()
            time.sleep(2.0)

	    # flex a finger
    def flex_finger(self, finger):
        if finger < 0 or finger > 4:
            print("ERROR in flex_finger: finger =", finger)
            sys.exit()

        # if thumb
        if finger == 4:
            self.move(2 * finger, 300)
            self.move(2 * finger + 1, 800)
        else:
            self.move(2 * finger, 300)
            self.move(2 * finger + 1, 300)

        self.wait_for_hand()
        time.sleep(1.0)

        self.move(2 * finger, 3850)
        self.move(2 * finger + 1, 3850)

        self.wait_for_hand()

		#Demo
    def demo(self):#move fingers
        self.move(0, 2500)
        self.move(1, 2500)
        time.sleep(3.0)
        self.move(0, 8000)
        self.move(1, 8000)
        time.sleep(3.0)
        self.move(9, 2500)
        self.move(8, 2500)
        time.sleep(3.0)
        self.move(7, 2500)
        self.move(6, 2500)
        time.sleep(3.0)
        self.move(5, 2500)
        self.move(4, 2500)
        time.sleep(3.0)
        self.move(3, 2500)
        self.move(2, 2500)
        time.sleep(3.0)
        self.move(2, 8000)
        self.move(3, 8000)
        time.sleep(3.0)
        self.move(4, 8000)
        self.move(5, 8000)
        time.sleep(3.0)
        self.move(6, 8000)
        self.move(7, 8000)
        time.sleep(3.0)
        self.move(8, 8000)
        self.move(9, 8000)
        time.sleep(3.0)	
        self.wait_for_hand()




