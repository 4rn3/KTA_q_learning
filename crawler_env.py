import sys
import tkinter
import numpy as np

import math
from math import pi as PI
import random

from gymnasium import utils
from gymnasium import Env, spaces
from gymnasium.utils import seeding

import platform, subprocess, os
import pickle
envs_list = []

class KTAICrawlerEnv(Env):

    def close_gui(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def __init__(self, horizon=np.inf, render=False):
        if render:
            for env in envs_list:
                env.close_gui()
            envs_list.clear()
            envs_list.append(self)

            root = tkinter.Tk()
            root.title('Crawler GUI')
            root.resizable(0, 0)

            self.root = root
            canvas = tkinter.Canvas(root, height=200, width=1000)
            canvas.grid(row=2, columnspan=10)

            def close():
                if self.root is not None:
                    self.root.destroy()
                    self.root = None
            root.protocol('WM_DELETE_WINDOW', lambda: close)
            root.lift()
 
            
        else:
            canvas = None
            self.root = None
            
        robot = CrawlingRobot(canvas)
        self.crawlingRobot = robot

        self._stepCount = 0
        self.horizon = horizon

        self.state = None

        self.nArmStates = 9
        self.nHandStates = 13

        minArmAngle, maxArmAngle = self.crawlingRobot.getMinAndMaxArmAngles()
        minHandAngle, maxHandAngle = self.crawlingRobot.getMinAndMaxHandAngles()
        
        armIncrement = (maxArmAngle - minArmAngle) / (self.nArmStates-1)
        handIncrement = (maxHandAngle - minHandAngle) / (self.nHandStates-1)
        
        self.armBuckets = [minArmAngle+(armIncrement*i) for i in range(self.nArmStates)]
        self.handBuckets = [minHandAngle+(handIncrement*i) for i in range(self.nHandStates)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(self.nArmStates), spaces.Discrete(self.nHandStates)]
        )

        self._reset()

    @property
    def stepCount(self):
        return self._stepCount

    @stepCount.setter
    def stepCount(self, val):
        self._stepCount = val
        self.crawlingRobot.draw(val, self.root)

    def _legal_actions(self, state):
        actions = list()

        currArmBucket,currHandBucket = state
        if currArmBucket > 0: actions.append(0)
        if currArmBucket < self.nArmStates-1: actions.append(1)
        if currHandBucket > 0: actions.append(2)
        if currHandBucket < self.nHandStates-1: actions.append(3)

        return actions

    def step(self, a):
        if self.stepCount >= self.horizon:
            raise Exception("Horizon reached")
        nextState, reward = None, None

        oldX, oldY = self.crawlingRobot.getRobotPosition()
        armBucket, handBucket = self.state

        if a in self._legal_actions(self.state):
            if a == 0:
                newArmAngle = self.armBuckets[armBucket-1]
                self.crawlingRobot.moveArm(newArmAngle)
                nextState = (armBucket-1,handBucket)
            elif a == 1:
                newArmAngle = self.armBuckets[armBucket+1]
                self.crawlingRobot.moveArm(newArmAngle)
                nextState = (armBucket+1,handBucket)
            elif a == 2:
                newHandAngle = self.handBuckets[handBucket-1]
                self.crawlingRobot.moveHand(newHandAngle)
                nextState = (armBucket,handBucket-1)
            elif a == 3:
                newHandAngle = self.handBuckets[handBucket+1]
                self.crawlingRobot.moveHand(newHandAngle)
                nextState = (armBucket,handBucket+1)
            else:
                raise Exception("action out of range")
        else:
            nextState = self.state

        newX, newY = self.crawlingRobot.getRobotPosition()

        reward = newX - oldX

        self.state = nextState
        self.stepCount += 1

        return tuple(nextState), reward, self.stepCount >= self.horizon, {}


    def _reset(self):
        armState = self.nArmStates // 2
        handState = self.nHandStates // 2
        self.state = armState, handState
        self.crawlingRobot.setAngles(self.armBuckets[armState], self.handBuckets[handState])
        self.crawlingRobot.positions = [20, self.crawlingRobot.getRobotPosition()[0]]

        self.stepCount = 0

class CrawlingRobot:

    def __init__(self, canvas):
        self.canvas = canvas
        self.velAvg = 0
        self.lastStep = 0

        self.armAngle = self.oldArmDegree = 0.0
        self.handAngle = self.oldHandDegree = -PI/6

        self.maxArmAngle = PI/6
        self.minArmAngle = -PI/6

        self.maxHandAngle = 0
        self.minHandAngle = -(5.0/6.0) * PI

        self.robotWidth = 80
        self.robotHeight = 40
        self.armLength = 60
        self.handLength = 40
        self.positions = [0,0]

        if canvas is not None:
            self.totWidth = canvas.winfo_reqwidth()
            self.totHeight = canvas.winfo_reqheight()
            self.groundHeight = 40
            self.groundY = self.totHeight - self.groundHeight

            self.ground = canvas.create_rectangle(
                0,
                self.groundY,self.totWidth,self.totHeight, fill='blue'
            )

            self.robotPos = (self.totWidth / 5 * 2, self.groundY)
            self.robotBody = canvas.create_polygon(0,0,0,0,0,0,0,0, fill='green')
            self.robotArm = canvas.create_line(0,0,0,0,fill='orange',width=5)
            self.robotHand = canvas.create_line(0,0,0,0,fill='red',width=3)
        else:
            self.robotPos = (20, 0)

    def setAngles(self, armAngle, handAngle):
        self.armAngle = armAngle
        self.handAngle = handAngle

    def getAngles(self):
        return self.armAngle, self.handAngle

    def getRobotPosition(self):
        return self.robotPos

    def moveArm(self, newArmAngle):
        oldArmAngle = self.armAngle
        if newArmAngle > self.maxArmAngle:
            raise 'Crawling Robot: Arm Raised too high. Careful!'
        if newArmAngle < self.minArmAngle:
            raise 'Crawling Robot: Arm Raised too low. Careful!'
        disp = self.displacement(self.armAngle, self.handAngle,
                                 newArmAngle, self.handAngle)
        curXPos = self.robotPos[0]
        self.robotPos = (curXPos+disp, self.robotPos[1])
        self.armAngle = newArmAngle

        self.positions.append(self.getRobotPosition()[0])
        if len(self.positions) > 100:
            self.positions.pop(0)

    def moveHand(self, newHandAngle):
        oldHandAngle = self.handAngle

        if newHandAngle > self.maxHandAngle:
            raise 'Crawling Robot: Hand Raised too high. Careful!'
        if newHandAngle < self.minHandAngle:
            raise 'Crawling Robot: Hand Raised too low. Careful!'

        disp = self.displacement(self.armAngle, self.handAngle, self.armAngle, newHandAngle)
        curXPos = self.robotPos[0]
        self.robotPos = (curXPos+disp, self.robotPos[1])
        self.handAngle = newHandAngle

        self.positions.append(self.getRobotPosition()[0])
        if len(self.positions) > 100:
            self.positions.pop(0)
    def getRotationAngle(self):
        armCos, armSin = self.cos_sin(self.armAngle)
        handCos, handSin = self.cos_sin(self.handAngle)
        x = self.armLength * armCos + self.handLength * handCos + self.robotWidth
        y = self.armLength * armSin + self.handLength * handSin + self.robotHeight
        if y < 0:
            return math.atan(-y/x)
        return 0.0

    def getMinAndMaxArmAngles(self):
        return self.minArmAngle, self.maxArmAngle


    def getMinAndMaxHandAngles(self):
        return self.minHandAngle, self.maxHandAngle

    def cos_sin(self, angle):
        return math.cos(angle), math.sin(angle)

    def displacement(self, oldArmDegree, oldHandDegree, armDegree, handDegree):

        oldArmCos, oldArmSin = self.cos_sin(oldArmDegree)
        armCos, armSin = self.cos_sin(armDegree)
        oldHandCos, oldHandSin = self.cos_sin(oldHandDegree)
        handCos, handSin = self.cos_sin(handDegree)

        xOld = self.armLength * oldArmCos + self.handLength * oldHandCos + self.robotWidth
        yOld = self.armLength * oldArmSin + self.handLength * oldHandSin + self.robotHeight

        x = self.armLength * armCos + self.handLength * handCos + self.robotWidth
        y = self.armLength * armSin + self.handLength * handSin + self.robotHeight

        if y < 0:
            if yOld <= 0:
                return math.sqrt(xOld*xOld + yOld*yOld) - math.sqrt(x*x + y*y)
            return (xOld - yOld*(x-xOld) / (y - yOld)) - math.sqrt(x*x + y*y)
        else:
            if yOld  >= 0:
                return 0.0
            return -(x - y * (xOld-x)/(yOld-y)) + math.sqrt(xOld*xOld + yOld*yOld)

        raise 'Never Should See This!'

    def draw(self, stepCount, root):
        if self.canvas is None or root is None:
            return
        x1, y1 = self.getRobotPosition()
        x1 = x1 % self.totWidth

        if y1 != self.groundY:
            raise 'Flying Robot!!'

        rotationAngle = self.getRotationAngle()
        cosRot, sinRot = self.cos_sin(rotationAngle)

        x2 = x1 + self.robotWidth * cosRot
        y2 = y1 - self.robotWidth * sinRot

        x3 = x1 - self.robotHeight * sinRot
        y3 = y1 - self.robotHeight * cosRot

        x4 = x3 + cosRot*self.robotWidth
        y4 = y3 - sinRot*self.robotWidth

        self.canvas.coords(self.robotBody,x1,y1,x2,y2,x4,y4,x3,y3)

        armCos, armSin = self.cos_sin(rotationAngle+self.armAngle)
        xArm = x4 + self.armLength * armCos
        yArm = y4 - self.armLength * armSin

        self.canvas.coords(self.robotArm,x4,y4,xArm,yArm)

        handCos, handSin = self.cos_sin(self.handAngle+rotationAngle)
        xHand = xArm + self.handLength * handCos
        yHand = yArm - self.handLength * handSin

        self.canvas.coords(self.robotHand,xArm,yArm,xHand,yHand)

        steps = (stepCount - self.lastStep)

        pos = self.positions[-1]
        velocity = pos - self.positions[-2]
        vel2 = (pos - self.positions[0]) / len(self.positions)
        self.velAvg = .9 * self.velAvg + .1 * vel2
        velMsg = '100-step Avg Velocity: %.2f' % self.velAvg
        velocityMsg = 'Velocity: %.2f' % velocity
        positionMsg = 'Position: %2.f' % pos
        stepMsg = 'Step: %d' % stepCount
        if 'vel_msg' in dir(self):
            self.canvas.delete(self.vel_msg)
            self.canvas.delete(self.pos_msg)
            self.canvas.delete(self.step_msg)
            self.canvas.delete(self.velavg_msg)

        self.velavg_msg = self.canvas.create_text(650,190,text=velMsg)
        self.vel_msg = self.canvas.create_text(450,190,text=velocityMsg)
        self.pos_msg = self.canvas.create_text(250,190,text=positionMsg)
        self.step_msg = self.canvas.create_text(50,190,text=stepMsg)

        self.lastStep = stepCount
        root.update()