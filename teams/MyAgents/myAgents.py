# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
import time

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class MyAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    return OffensiveReflexAgent(index)


##########
# Agents #
##########

class OffensiveReflexAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        start_time = time.time()
        "Calculations that are only required before the game starts."
        CaptureAgent.registerInitialState(self, gameState)
        grid = gameState.data.food
        self.halfway = grid.width/2
        self.halfheight = grid.height/2
        self.enemies = self.getOpponents(gameState)
        #Setup legal actions and legal positions
        self.legalPositions = list()
        self.legalActions = self.calculateLegalAction(gameState)
        #Setup particles
        self.particles = dict()
        self.distributions = list()
        self.numParticles = 500
        #Set all particles at the start position of this agent
        for enemy in self.enemies:
            self.setParticles(gameState.getInitialAgentPosition(enemy), enemy)
        end_time = time.time()
        print "total pre-compute time:", end_time-start_time



    def initializeUniformly(self, gameState, enemy):
        "Begin with a uniform distribution over initial positions."
        uniform = self.numParticles/len(self.legalPositions)
        for position in self.legalPositions:
            for i in range(uniform):
                self.particles[enemy].append(position)

    def setParticles(self, Position, enemy):
        "Set all particles of a certain enemy to a certain position"
        self.particles[enemy] = list()
        for i in range(self.numParticles):
            self.particles[enemy].append(Position)

    def observe(self, gameState):
        "Based on current observation, update particles."
        noisyDistances = self.getCurrentObservation().getAgentDistances()
        for enemy in self.enemies:
            newPossible = util.Counter()
            noisyDistance = noisyDistances[enemy]
            #If we can observe the enemy, set all particles to that location.
            if gameState.getAgentPosition(enemy) != None:
                self.setParticles(gameState.getAgentPosition(enemy), enemy)
                continue
            #Otherwise calculate the possibility
            for p in self.particles[enemy]:
                trueDistance = util.manhattanDistance(p, self.pos)
                newPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)

            #if the weight of all particles are 0, uniform initialize
            if newPossible.totalCount() == 0:
                self.initializeUniformly(gameState, enemy)
                continue

            self.particles[enemy] = list()
            for new_p in range(self.numParticles):
                self.particles[enemy].append(util.sample(newPossible))

    def elapseTime(self, gameState):
        for enemy in self.enemies:
            new_particles = list()
            if gameState.getAgentPosition(enemy) != None:
                self.setParticles(gameState.getAgentPosition(enemy), enemy)
                continue
            for x, y in self.particles[enemy]:
                newPos = list()
                x = int(x)
                y = int(y)
                newPos.append((x, y))
                if not gameState.hasWall(x, y+1):
                    newPos.append((x, y+1))
                if not gameState.hasWall(x, y-1):
                    newPos.append((x, y-1))
                if not gameState.hasWall(x-1, y):
                    newPos.append((x-1, y))
                if not gameState.hasWall(x+1, y):
                    newPos.append((x+1, y))
                new_particles.append(random.choice(newPos))
            self.particles[enemy] = new_particles

    def getBeliefDistribution(self):
        self.distributions = list()
        for enemy in self.enemies:
            allPossible = util.Counter()
            for p in self.particles[enemy]:
                allPossible[p] += 1
            allPossible.normalize()
            self.distributions.append(allPossible)

    def chooseAction(self, gameState):
        #overview of the board
        self.pos = gameState.getAgentPosition(self.index)
        self.foods = self.findMyFoods(gameState)

        self.GhostInference(gameState)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        features['numFoodLeft'] = len(self.getFood(successor).asList())
        # Compute distance to the nearest food
        foodList = self.foods
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        closestEnemy = 100
        for enemy in self.enemies:
            if gameState.getAgentPosition(enemy) != None:
                distance = self.getMazeDistance(gameState.getAgentPosition(enemy), myPos)
                if distance < closestEnemy:
                    closestEnemy = distance
        if closestEnemy<4:
            features['distanceToGhost'] = 5-closestEnemy
        #No observation available, need inference.
        if closestEnemy == 100:
            for enemy in self.enemies:
                distribution = self.distributions[enemy/2]
                if distribution[myPos]>0.1:
                    features['distanceToGhost'] = 0.5
        #if our agents are ghost and our ghost is not scared:
        if not successor.getAgentState(self.index).isPacman and not self.isScared(gameState):
            features['distanceToGhost'] = 0

        #if our agents are pacman, and the enemy ghost is scared:
        for enemy in self.enemies:
            if successor.getAgentState(self.index).isPacman and gameState.getAgentState(enemy).scaredTimer > 0:
               features['distanceToGhost'] = 0

        #if our agents are ghost,
        #STOP penalty
        if action == Directions.STOP: features['stop'] = 1
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': -5,
                'stop': -100, 'numFoodLeft': -2}


    def GhostInference(self, gameState):
        self.observe(gameState)
        self.elapseTime(gameState)
        self.getBeliefDistribution()
        self.displayDistributionsOverPositions(self.distributions)
        for distribution in self.distributions:
            for pos in distribution:
                pass

    def findMyFoods(self, gameState):
        """
        Divide the food board into 2 parts: top and bottom. One agent will
        focus on the top and another agent will focus on the bottom.
        """
        foods = self.getFood(gameState)
        myFoods = list()
        if self.index>=2:
            for food_pos in foods.asList():
                x, y = food_pos
                if y>=self.halfheight:
                    myFoods.append(food_pos)
        else:
            for food_pos in foods.asList():
                x, y = food_pos
                if y<self.halfheight:
                    myFoods.append(food_pos)
        if len(myFoods) == 0:
            return self.getFood(gameState).asList()
        else:
            return myFoods

    def calculateLegalAction(self, gameState):
        "Use the wall to calculate legal action of every position."
        legalActions = dict()
        grid = gameState.getWalls()
        for x in range(grid.width):
            for y in range(grid.height):
                if not gameState.hasWall(x, y):
                    self.legalPositions.append((x, y))
                    actions = list()
                    actions.append(Directions.STOP)
                    if not gameState.hasWall(x, y+1):
                        actions.append(Directions.NORTH)
                    if not gameState.hasWall(x, y-1):
                        actions.append(Directions.SOUTH)
                    if not gameState.hasWall(x-1, y):
                        actions.append(Directions.WEST)
                    if not gameState.hasWall(x+1, y):
                        actions.append(Directions.EAST)
                    legalActions[(x, y)] = actions
                    #print "position:", (x, y)
                    #print "legal actions:", actions
        return legalActions

    def scaredTimeRemaining(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    def isScared(self, gameState):
        return self.scaredTimeRemaining(gameState)>0
