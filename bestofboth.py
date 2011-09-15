import math
import operator
import optparse
import os
import Queue
import random
import re
import shutil
import sys
import textwrap
import zlib

from pymclevel import materials
from pymclevel import mclevel
from pymclevel.mclevelbase import ChunkNotPresent

VERSION_STRING = "0.1"

# Sea level is 62 in 1.8 and 63 in 1.7 and previous.
WATER_HEIGHT = 62
MAX_HEIGHT   = 128

# The six orthogonal directions: up, down, left, right, front, and back.
ORTHOGONAL_NEIGHBOR_POSITIONS = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
    ]

ALL_NEIGHBOR_POSITIONS = []
for x in range(-1, 2):
    for z in range(-1, 2):
        for y in range(-1, 2):
            if x or z or y:
                ALL_NEIGHBOR_POSITIONS.append((x, z, y))

class ErosionTask:
    @staticmethod
    def fromString(string):
        (erosionType, direction, posX, posZ) = re.split("\s+", string)
        if erosionType == "corner":
            return CornerErosionTask(Erode.Map.index(direction), int(posX), int(posZ))
        elif erosionType == "edge":
            return EdgeErosionTask(Erode.Map.index(direction), int(posX), int(posZ))
        else:
            raise Exception("unrecognized erosion type '%s'" % (erosionType))
        #    (chunkX, chunkZ, erodeType, xPos, zPos, xMin, xMax, zMin, zMax) = [int(x) for x in re.split("\s+", line)]
        #      if type
        
    def __init__(self, posX, posZ):
        self.posX = posX
        self.posZ = posZ
    
    def removeOrphanLeaf(self, level, x, z, y):
        # This is basically a block-fill algorithm: find all leaf blocks
        # connected to this one. If no log block is found on the way,
        # delete the whole group of leaf blocks.
        
        trunkFound = False
        frontier = []
        visited = set()
        frontier.append((x, z, y))
        visited.add((x, z, y))

        while frontier:
            fPos = frontier.pop()
            #print("fPos: %d,%d,%d; frontier size: %d" % (x, z, y, len(frontier)))
            (fx, fz, fy) = fPos
            chunk = level.getChunk(fx / 16, fz / 16)
            blockID = chunk.Blocks[fx % 16, fz % 16, fy]
            if blockID == leafID:
                for nPos in ORTHOGONAL_NEIGHBOR_POSITIONS:
                    (nx, nz, ny) = map(operator.add, (fx, fz, fy), nPos)
                    if (nx, nz, ny) not in visited:
                        visited.add((nx, nz, ny))
                        frontier.append((nx, nz, ny))
            elif blockID == logID:
                trunkFound = True
                break
        
        if not trunkFound:
            for (vx, vz, vy) in visited:
                # Some of these visited blocks aren't leaf blocks, but
                # that's okay, because removeLeafBlock() will ignore
                # them.
                chunk = level.getChunk(vx / 16, vz / 16)
                remove_tree_block(chunk, vx % 16, vz % 16, vy)
    
    # Searches a column of a chunk for leaves. If leaves are found that are not
    # connected to a tree trunk, they will be turned into air.
    def removeOrphanLeaves(self, level, startChunk, relX, relZ):
        
        x = relX + startChunk.chunkPosition[0] * 16
        z = relZ + startChunk.chunkPosition[1] * 16
        
        #print("removing orphan leaves from column %d,%d" % (x, z))
        
        # There shouldn't be any orphaned leaves below sea level.
        for y in range(WATER_HEIGHT, 128):
            if startChunk.Blocks[relX, relZ, y] == leafID:
                self.removeOrphanLeaf(level, x, z, y)
    
    # Returns the altitude at which the water column starts (assuming it
    # ends at sea level).
    def waterDepth(self, chunk, x, z):
        h = WATER_HEIGHT
        while chunk.Blocks[x, z, h] in [iceID, waterID]:
            h -= 1
        return h
    
    def getChunksAndWaterDepth(self, level):
        chunksToEdit = {}
        chunkWaterDepths = []
        
        for cx in range(-1, 1):
            for cz in range(-1, 1):
                chunk = level.getChunk(self.posX / 16 + cx, self.posZ / 16 + cz)
                chunksToEdit[(cx, cz)] = chunk
                # Find the average water depth along the two borders of this
                # chunk that contact the other chunks involved in this
                # erosion task. For example, if this is the top-left chunk,
                # find the average water depth along its right and bottom
                # edges.
                rowX = 0 if cx == 0 else 15
                rowZ = 0 if cz == 0 else 15
                sumWaterDepths = 0
                for x in range(0, 16):
                    sumWaterDepths += self.waterDepth(chunk, x, rowZ)
                for z in range(0, 16):
                    sumWaterDepths += self.waterDepth(chunk, rowX, z)
            
                chunkWaterDepths.append(sumWaterDepths / 32)
        
        deepestWaterDepth = WATER_HEIGHT - 1
        for wd in chunkWaterDepths:
            if wd < deepestWaterDepth:
                deepestWaterDepth = wd
        
        return (chunksToEdit, deepestWaterDepth)
    
    # Erodes a column of terrain 1 block wide and 1 block long.
    #
    # relativeDistance: ranges from -1 (on the far edge of the erosion
    #           area) to 1 (on the near edge of the erosion area), with
    #           0 meaning the point is in the middle of the river
    # waterWidth: the width of the river, relative to the width of the erosion
    #           area
    def erode(self, chunk, treeDecayList, x, z, relativeDistance, waterWidth, deepestWaterDepth):
        #print("setting (%d,%d) to h=%d" % (x, z, h))
        
        if deepestWaterDepth < WATER_HEIGHT and abs(relativeDistance) < waterWidth:
            # We're in the water and need to slope downward to the ocean
            # floor at deepestWaterDepth
            
            # relativeDistance is on the interval [-waterWidth..waterWidth]
            # (with -waterWidth corresponding to the ocean side and waterWidth
            # corresponding to the ocean side).
            # distanceAcross will be on the interval [0..1] (with 0 being the
            # near side and 1 being the ocean side).
            distanceAcross = (relativeDistance - waterWidth) / (-2 * waterWidth)
            
            # h should range from WATER_HEIGHT to deepestWaterDepth, depending on
            # how far across the river we are.
            h = (deepestWaterDepth - WATER_HEIGHT) * distanceAcross
            h += WATER_HEIGHT
            
            # Make the river bed slope upward slightly toward the edges.
            h += 0.3 * abs(relativeDistance) / waterWidth
            
        else:
            # We're on land. Make relativeDistance positive for simplicity.
            if relativeDistance < 0:
                relativeDistance *= -1
            relativeRiverDistance = (relativeDistance - waterWidth) / (1 - relativeDistance)
            
            # relativeRiverDistance measures the relative distance from the
            # river's edge to the high point (instead of from the river's center
            # to the high point).
            
            currentTerrainHeight = MAX_HEIGHT - 1
            while chunk.Blocks[x, z, currentTerrainHeight] in leafAndAirIDs:
                currentTerrainHeight -= 1
            
            # relativeDistance is on the interval [0..1],
            # so h will be on the interval [2^0..2^1], which is [1..2].
            #h = 2 ** relativeDistance
            
            # Shift h to the interval [0..1].
            #h -= 1
            #h *= (currentTerrainHeight - WATER_HEIGHT)
            
            h = (currentTerrainHeight - WATER_HEIGHT) * relativeRiverDistance
            if (h < 0):
                h = 0
            if (h < 1):
                h += 0.5
            h += WATER_HEIGHT
        
        h = int(h)

        #print("(%d,%d) %d" % (x, z, h))
        
        chunkChanged = False
        
        if h < MAX_HEIGHT:
        
            blockWasIce = (chunk.Blocks[x, z, WATER_HEIGHT] == iceID)
            
            # The height at which air will begin.
            airHeight = max(h, WATER_HEIGHT + 1)
            
            # If a tree is standing on terrain that will be preserved,
            # preserve the tree, too.
            while chunk.Blocks[x, z, airHeight] == logID:
                airHeight += 1
            
            for logHeight in range(airHeight, MAX_HEIGHT):
                if chunk.Blocks[x, z, logHeight] == logID:
                    treeDecayList.append((chunk, x, z, logHeight))
            
            surfaceHeight = 127
            
            erodeDest = airHeight - 1
            while surfaceHeight > airHeight - 1 and erodeDest > WATER_HEIGHT:
                block = chunk.Blocks[x, z, surfaceHeight]
                data  = chunk.Data  [x, z, surfaceHeight]
                
                # If this block is a log that's above dirt, pretend it was a
                # sapling so that we can be a responsible citizen and replant
                # the tree.
                if block == logID:
                    if chunk.Blocks[x, z, surfaceHeight - 1] in [dirtID, grassID]:
                        # Make sure there's a dirt block below the sapling (so
                        # it can grow) and an air block above it. (It's possible
                        # that the block above is a snow layer.)
                        chunk.Blocks[x, z, erodeDest - 1] = grassID
                        chunk.Blocks[x, z, erodeDest]     = saplingID
                        chunk.Data  [x, z, erodeDest]     = saplingData[data]
                        chunk.Blocks[x, z, erodeDest + 1] = airID
                        chunkChanged = True
                        erodeDest -= 2
                
                # During erosion, skip air, leaves, logs and vines.
                elif block not in [airID, leafID, logID, vinesID]:
                    chunk.Blocks[x, z, erodeDest] = block
                    chunk.Data  [x, z, erodeDest] = data
                    chunkChanged = True
                    erodeDest -= 1
                surfaceHeight -= 1
            
            # Turn everything in this vertical column into air, but leave
            # trees alone (to avoid weird-looking half-trees). Trees will be
            # decayed elsewhere.
            # TODO: Allow vines to persist if they're attached to a tree that
            # persists.
            for ah in range(airHeight, 128):
                if chunk.Blocks[x, z, ah] not in [leafID, logID]:
                    chunk.Blocks[x, z, ah] = airID
                    chunk.Data  [x, z, ah] = 0
            if h <= WATER_HEIGHT + 1:
                if h <= WATER_HEIGHT:
                    chunk.Blocks[x, z, h : WATER_HEIGHT + 1] = waterID
                # Turn non-water, non-ice blocks along the shoreline, or under the water, into sand.
                if chunk.Blocks[x, z, h - 1] not in [iceID, waterID]:
                    chunk.Blocks[x, z, h - 1] = sandID
            
            if blockWasIce:
                # Restore ice that was far from the center of the river.
                # A larger relative distance from the center of the river should
                # result in a greater chance of restoring the block of ice.
                if random.random() < abs(relativeDistance):
                    chunk.Blocks[x, z, WATER_HEIGHT] = iceID
            
            chunkChanged = True
        
        return chunkChanged
        
class CornerErosionTask(ErosionTask):
    def __init__(self, cornerDirection, cornerPosX, cornerPosZ):
        ErosionTask.__init__(self, cornerPosX, cornerPosZ)
        self.cornerDirection = cornerDirection
        
    def __repr__(self):
        return "corner %-2s %d %d" % (Erode.Map[self.cornerDirection], self.posX, self.posZ)
        
    def run(self, level, decayList, erosionWidth = 8, waterWidth = 3):
        try:
            (chunksToEdit, deepestWaterDepth) = self.getChunksAndWaterDepth(level)
        except ChunkNotPresent:
            return False
            
        if self.cornerDirection == Erode.TL:
            highPoint = (8, 8)
        elif self.cornerDirection == Erode.TR:
            highPoint = (-8, 8)
        elif self.cornerDirection == Erode.BR:
            highPoint = (-8, -8)
        elif self.cornerDirection == Erode.BL:
            highPoint = (8, -8)
            
        for cx in range(-1, 1):
            for cz in range(-1, 1):
                chunkChanged = False
                chunk = chunksToEdit[(cx, cz)]
                highPointX = highPoint[0] - (cx * 16)
                highPointZ = highPoint[1] - (cz * 16)
                for x in range(-8 * cx, -8 * (cx - 1)):
                    for z in range(-8 * cz, -8 * (cz - 1)):
                        # This is the edge of a chunk. Leaves on the edge of the
                        # world may not have a trunk attached to them (since the
                        # terrain which was supposed to have the trunk was never
                        # generated). These leaves should be examined and
                        # possibly removed.
                        if (cx == -1 and x == 15) or (cx == 0 and x == 0) \
                                or (cz == -1 and z == 15) or (cz == 0 and z == 0):
                            self.removeOrphanLeaves(level, chunk, x, z)
                        dx = x - (highPointX - 0.5)
                        dz = z - (highPointZ - 0.5)
                        
                        distanceFromCenter = math.sqrt(dx * dx + dz * dz)
                        distanceFromEdge = 8 - distanceFromCenter
                        
                        if abs(distanceFromEdge) < erosionWidth:
                            relativeDistance = float(distanceFromEdge) / float(erosionWidth)
                            chunkChanged |= self.erode(chunk, decayList, x, z, relativeDistance, .375, deepestWaterDepth)

                if chunkChanged:
                    chunk.chunkChanged()
        return True
        
class EdgeErosionTask(ErosionTask):
    def __init__(self, edgeDirection, edgePosX, edgePosZ):
        ErosionTask.__init__(self, edgePosX, edgePosZ)
        self.edgeDirection = edgeDirection
        
    def __repr__(self):
        return "edge   %-2s %d %d" % (Erode.Map[self.edgeDirection], self.posX, self.posZ)
        
    def run(self, level, decayList, erosionWidth = 8, waterWidth = 3):
        try:
            (chunksToEdit, deepestWaterDepth) = self.getChunksAndWaterDepth(level)
        except ChunkNotPresent:
            return False
        
        # Let straight sections of the river bend slightly. (When wiggleRoom is
        # 0 the river will be perfectly straight.)
        wiggleRoom = random.random() - 0.5
        
        for cx in range(-1, 1):
            for cz in range(-1, 1):
                chunkChanged = False
                xMin = -8 * cx
                xMax = -8 * (cx - 1)
                zMin = -8 * cz
                zMax = -8 * (cz - 1)
                chunk = chunksToEdit[(cx, cz)]
                
                # This may be the edge of a chunk. Leaves on the edge of the
                # world may not have a trunk attached to them (since the terrain
                # which was supposed to have the trunk was never generated).
                # These leaves should be examined and possibly removed.
                if self.edgeDirection == Erode.HE:
                    z = 15 if cz == -1 else 0
                    for x in range(0, 16):
                        self.removeOrphanLeaves(level, chunk, x, z)
                elif self.edgeDirection == Erode.VE:
                    x = 15 if cx == -1 else 0
                    for z in range(0, 16):
                        self.removeOrphanLeaves(level, chunk, x, z)
                                
                for x in range(xMin, xMax):
                    for z in range(zMin, zMax):
                        if self.edgeDirection == Erode.HE:
                            # horizontal edge
                            distanceFromCenter = abs(7.5 - z)
                            distanceFromCenter += wiggleRoom * (abs(x - 7.5)) * (1 if z > 7.5 else -1)
                        elif self.edgeDirection == Erode.VE:
                            # vertical edge
                            distanceFromCenter = abs(7.5 - x)
                            distanceFromCenter += wiggleRoom * (abs(z - 7.5)) * (1 if x > 7.5 else -1)
                        else:
                            raise Exception("unrecognized edge direction %d (%s)" % (self.edgeDirection, Erode.Map[self.edgeDirection]))
                            
                        distanceFromEdge = 8 - distanceFromCenter
                        
                        if abs(distanceFromEdge) < erosionWidth:
                            relativeDistance = float(distanceFromEdge) / float(erosionWidth)
                            chunkChanged |= self.erode(chunk, decayList, x, z, relativeDistance, .375, deepestWaterDepth)
                            
                if chunkChanged:
                    chunk.chunkChanged()
        return True


# These values indicate the shape a chunk should have after erosion.
# T, B, L, and R mean "top", "bottom", "left" and "right", respectively.
# TL, TR, BL, and BR refer to the four kinds of corners: "top left",
# "top right", "bottom left", and "bottom right", respectively.
class Erode:
    Map = [ "TL", "TR", "BL", "BR", "VE", "HE" ]
    TL  = 0 # top-left corner
    TR  = 1 # top-right corner
    BL  = 2 # bottom-left corner
    BR  = 3 # bottom-right corner
    VE  = 4 # vertical edge
    HE  = 5 # horizontal edge

airID       = materials.alphaMaterials.Air.ID
dirtID      = materials.alphaMaterials.Dirt.ID
grassID     = materials.alphaMaterials.Grass.ID
iceID       = materials.alphaMaterials.Ice.ID
leafID      = materials.alphaMaterials.Leaves.ID
logID       = materials.alphaMaterials.Wood.ID
sandID      = materials.alphaMaterials.Sand.ID
saplingID   = materials.alphaMaterials.Sapling.ID
snowLayerID = materials.alphaMaterials.SnowLayer.ID
vinesID     = materials.alphaMaterials.Vines.ID
waterID     = materials.alphaMaterials.WaterStill.ID

# Map log IDs to the corresponding sapling types. This is used when replanting
# trees on eroded slopes. Since normal trees are the only ones that can grow in
# a 1x1 column, we'll only use normal saplings. (Other saplings might not be
# able to grow.)
saplingData = {
    materials.alphaMaterials.Wood.blockData:         materials.alphaMaterials.Sapling.blockData,
    materials.alphaMaterials.Ironwood.blockData:     materials.alphaMaterials.Sapling.blockData,
    materials.alphaMaterials.BirchWood.blockData:    materials.alphaMaterials.Sapling.blockData,
}

leafAndAirIDs = [
    airID,
    leafID,
    vinesID,
    ]

def find_edges(worldDir, edgeFilename):
    level = mclevel.fromFile(worldDir)
    edgeFile = open(edgeFilename, "w")
    sys.stdout.write("finding edges...")
    
    chunks = []
    
    for chunk in level.allChunks:
        chunks.append(chunk)
    
    erodeTasks = []
    
    examined = 0
    lastProgress = 0
    numChunks = len(chunks)
    
    for chunk in chunks:
        checkChunk(level, chunk, erodeTasks)
        examined += 1
        progress = examined * 100 / numChunks
        if progress != lastProgress:
            lastProgress = progress
            sys.stdout.write("\rfinding edges (%d%%)..." % (progress))
    print("")
    
    edgeFile.write("# erodeType erodeDirection posX posZ\n")
    
    numEdgeChunks = 0
    for task in erodeTasks:
        edgeFile.write("%s\n" % (task))
        numEdgeChunks += 1
    edgeFile.close()
    print("found %d edge(s)" % (numEdgeChunks))

def ice_wall(worldDir):
    level = mclevel.fromFile(worldDir)
    print("making an ice wall around the world...")
    
    chunks = []
    
    for chunk in level.allChunks:
        chunks.append(chunk)
    
    erodeTasks = []
    
    aroundMe = [(-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1)]
    i = 0
    for chunkPos in chunks:
        i += 1
        sys.stdout.write("\r   chunk %d of %d" % (i, len(chunks)))
        (cx, cz) = chunkPos
        for (dx, dz) in aroundMe:
            if (cx + dx, cz + dz) not in chunks:
                # It's an edge chunk! Make a GIANT ICE WALL.
                chunk = level.getChunk(cx, cz)
                chunk.Blocks[:, :, 1:] = iceID
                chunk.chunkChanged()
                break
    print("")
    level.saveInPlace()

# Remove this leaf or log block (and collapse the snow above it). Also works for
# vines: if this is a vine block, it and any vines directly beneath it will be
# removed.
def remove_tree_block(chunk, relX, relZ, relY, distance = 1):
    blockID = chunk.Blocks[relX, relZ, relY]
    
    turnToAir = []
    
    if blockID == leafID:
        chunk.Blocks[relX, relZ, relY] = airID #materials.alphaMaterials.WhiteWool.ID
        chunk.Data  [relX, relZ, relY] = 0 #distance
    
        # If this leaf had a snow layer above it, make the snow fall to the
        # next-lowest block.
        if chunk.Blocks[relX, relZ, relY + 1] == snowLayerID:
            snowY = relY + 1
            while chunk.Blocks[relX, relZ, snowY - 1] == airID:
                snowY -= 1
            
            # If the snow layer has fallen, turn its former location to air and
            # its new location to snow.
            if snowY != relY:
                chunk.Blocks[relX, relZ, relY + 1] = airID
                chunk.Data  [relX, relZ, relY + 1] = 0
                chunk.Blocks[relX, relZ, snowY] = snowLayerID
        
        chunk.chunkChanged()
    
    elif blockID == vinesID:
        # Destroy this vine block and any vines beneath it.
        vineY = relY
        while chunk.Blocks[relX, relZ, vineY] == vinesID:
            chunk.Blocks[relX, relZ, vineY] = airID
            chunk.Data  [relX, relZ, vineY] = 0
            vineY -= 1
        chunk.chunkChanged()
    
    # Some logs might be found at nonzero distances. This means they were't part
    # of the tree being eroded, so we shouldn't remove them.
    elif blockID == logID and distance == 0:
        chunk.Blocks[relX, relZ, relY] = airID #materials.alphaMaterials.LavaStill.ID
        chunk.Data  [relX, relZ, relY] = 0
        chunk.chunkChanged()

def decay_trees(level, decayList):
    # decayList is a list of locations of logs which have been removed.
    # Leaves further than 4 blocks from one of these removed logs should be
    # decayed.
    logQueue = Queue.PriorityQueue()
    for (chunk, x, z, y) in decayList:
        # x and z are relative to the chunk. Let's make them universal.
        x += chunk.chunkPosition[0] * 16
        z += chunk.chunkPosition[1] * 16
        #print("%s %d,%d,%d" % (chunk, x, y, z))
        logQueue.put((0, x, z, y))
    
    # Leaves can be up to 6 blocks from a trunk.
    LEAF_DISTANCE_LIMIT = 6
    
    # First, find all logs that are attached to this tree.
    logs = set()
    maxSize = logQueue.qsize()
    i = 0
    while not logQueue.empty():
        i += 1
        (distance, x, z, y) = logQueue.get()
        sys.stdout.write("\rexamining tree %d of %d on the erosion boundary..." % (i, maxSize))
        treeLogQueue = Queue.PriorityQueue()
        treeLogQueue.put((x, z, y, False, x, z, y))
        treeLogs = set()
        rootBlock = None
        while not (treeLogQueue.empty() or rootBlock):
            # (px, pz, py) are the coordinates of the "parent" block -- the one
            # that led to the block at (x, z, y).
            (x, z, y, lookForGround, px, pz, py) = treeLogQueue.get()
            if (x, z, y) not in treeLogs:
                isLog = False
                
                (relX, relZ, relY) = (x % 16, z % 16, y)
                chunk = level.getChunk(x / 16, z / 16)
                blockID = chunk.Blocks[relX, relZ, relY]
                if blockID == logID:
                    isLog = True
                    #chunk.Blocks[relX, relZ, relY] = airID #materials.alphaMaterials.LavaStill.ID
                    #chunk.Data  [relX, relZ, relY] = 0
                    #chunk.chunkChanged()
                elif lookForGround and blockID in [dirtID, grassID]:
                    rootBlock = (px, pz, py)
                
                if isLog:
                    treeLogs.add((x, z, y))
                    for nPos in ALL_NEIGHBOR_POSITIONS:
                        # Check nearby blocks for dirt: below, and each of the
                        # four blocks next to the block below.
                        lookForGround = (nPos[2] == -1 and (nPos[0] == 0 or nPos[1] == 0))
                        
                        (nx, nz, ny) = map(operator.add, (x, z, y), nPos)
                        treeLogQueue.put((nx, nz, ny, lookForGround, x, z, y))
        if not rootBlock:
            logs |= treeLogs
        else:
            # print("found the ground!")
            # (px, pz, py) are the coordinates of the tree's lowest trunk block.
            # Make sure there are no air blocks beneath it.
            chunk = level.getChunk(px / 16, pz / 16)
            (relX, relZ, relY) = (px % 16, pz % 16, py)
            airY = relY - 1
            while chunk.Blocks[relX, relZ, airY] in [airID, grassID, saplingID]:
                chunk.Blocks[relX, relZ, airY] = dirtID
                chunk.Data  [relX, relZ, airY] = 0
                airY -= 1
    
    decayQueue = Queue.PriorityQueue()
    for (x, z, y) in logs:
        decayQueue.put((0, x, z, y))
    print("")
    
    maxSize = 0
    while not decayQueue.empty():
        maxSize = max(decayQueue.qsize(), maxSize)
        (distance, x, z, y) = decayQueue.get()
        sys.stdout.write("\rdecaying trees (decay queue size: %d; max: %d)...%s" % (decayQueue.qsize(), maxSize, " " * 10))
        
        (relX, relZ, relY) = (x % 16, z % 16, y)
        chunk = level.getChunk(x / 16, z / 16)
        blockID = chunk.Blocks[relX, relZ, relY]
        
        # Vines may be attached to leaves, so we have to allow extra search
        # distance for vines.
        distanceLimit = LEAF_DISTANCE_LIMIT
        if blockID == vinesID:
            distanceLimit += 1
        if distance > distanceLimit:
            continue
        
        neighborPositions = []
        
        # A list of chunk-relative Y-coordinates (at relX, relZ) which should be
        # turned into air blocks.
        turnToAir = []
        
        # distance = 0 corresponds to logs which were removed during erosion.
        if distance == 0:
            neighborPositions = ORTHOGONAL_NEIGHBOR_POSITIONS
        elif blockID == leafID:
            neighborPositions = ORTHOGONAL_NEIGHBOR_POSITIONS
        
        remove_tree_block(chunk, relX, relZ, relY, distance)
        
        for nPos in neighborPositions:
            (nx, nz, ny) = map(operator.add, (x, z, y), nPos)
            decayQueue.put((distance + 1, nx, nz, ny))
    
    print("")

def smooth(worldDir, edgeFilename, width = 16):
    level = mclevel.fromFile(worldDir)
    newEdgeFile = open(edgeFilename + ".tmp", "w")
    edgeFile = open(edgeFilename, "r")
    
    width = int(width) / 2
    
    erosionTasks = []
    
    for line in edgeFile.readlines():
        originalLine = line
        line = line.strip()
        # Preserve comments
        if line.startswith("#"):
            newEdgeFile.write(originalLine)
        else:
            task = ErosionTask.fromString(line)
            erosionTasks.append(task)
    
    edgeFile.close()
    
    numTasks = len(erosionTasks)
    
    skipped = 0
    smoothed = 0
    
    treeDecayList = []
    
    if erosionTasks:
        examined = 0
        for erosionTask in erosionTasks:
            examined += 1
            sys.stdout.write("\rexamining edge %d of %d..." % (examined, numTasks))
            
            # If the task didn't run (because it requires chunks that
            # haven't been generated yet), write it back to edges.txt.
            if erosionTask.run(level, treeDecayList, width):
                smoothed += 1
            else:
                skipped += 1
                newEdgeFile.write("%s\n" % (task))
        print("")
        
        decay_trees(level, treeDecayList)
        
        print("saving changes...")
        level.saveInPlace()
    
    newEdgeFile.close()
    
    if smoothed:
        print("smoothed %d edge(s)" % (smoothed))
        shutil.move(newEdgeFile.name, edgeFilename)
    else:
        os.remove(newEdgeFile.name)
    
    if skipped:
        print("%d edge(s) can't be smoothed yet, since they're not fully explored" % (skipped))
    elif smoothed == numTasks:
        print("the map is perfectly smoothed -- nothing to do!")

def fix_sea_level(worldDir, edgeFilename):
    level = mclevel.fromFile(worldDir)
    
    waterBlocks = 0
    blocks = 0
    
    allChunks = [x for x in level.allChunks]
    numChunks = len(allChunks)
    i = 0
    for chunkPosition in allChunks:
        i += 1
        sys.stdout.write("\rdetecting current sea level (chunk %d of %d)..." % (i, numChunks))
        chunk = level.getChunk(chunkPosition[0], chunkPosition[1])
        # Shift everything down by one block, deleting whatever is at layer 1.
        # (Layer 0 should remain solid bedrock.)
        for x in range(0, 16):
            for z in range(0, 16):
                if chunk.Blocks[x, z, WATER_HEIGHT] in [waterID, iceID]:
                    waterBlocks += 1
                blocks += 1
    print("")
    waterCoverage = float(waterBlocks) / blocks
    print("water coverage at y=%d: %.2f%%" % (WATER_HEIGHT, (waterCoverage * 100)))
    
    if waterCoverage < 0.1:
        print("... but that seems too low!")
        sys.exit(1)
    
    i = 0
    for chunkPosition in allChunks:
        i += 1
        chunk = level.getChunk(chunkPosition[0], chunkPosition[1])
        # Shift everything down by one block, deleting whatever is at layer 1.
        # (Layer 0 should remain solid bedrock.)
        sys.stdout.write("\rreducing sea level (chunk %d of %d)..." % (i, numChunks))
        newBlocks = []
        newData   = []
        for x in range(0, 16):
            for z in range(0, 16):
                newBlocks[:] = chunk.Blocks[x, z, 2 : 127]
                newData  [:] = chunk.Data  [x, z, 2 : 127]
                chunk.Blocks[x, z, 1 : 126] = newBlocks
                chunk.Data  [x, z, 1 : 126] = newData
                """
                for y in range (1, 127):
                    chunk.Blocks[x, z, y] = chunk.Blocks[x, z, y + 1]
                    chunk.Data  [x, z, y] = chunk.Data  [x, z, y + 1]
                """
                chunk.Blocks[x, z, 127] = airID
                chunk.Data  [x, z, 127] = 0
        chunk.chunkChanged()
    
    level.saveInPlace()
    
    # TODO: Make sure edges.txt hasn't been modified. (How? There could have
    # been multiple "islands" of land in the original world, so the presence
    # of one completely-enclosed group of chunks doesn't tell us anything.
    
    # TODO: Autodetect the current sea level instead of assuming it's 64 (1.7).

def addCorner(chunkPos, erodeList, erodeType):
    (chunkX, chunkZ) = chunkPos
    erodeList.append(
        CornerErosionTask(
            erodeType,
            chunkX * 16,
            chunkZ * 16
        )
    )

def addEdge(chunkPos, erodeList, erodeType):
    (chunkX, chunkZ) = chunkPos
    erodeList.append(
        EdgeErosionTask(
            erodeType,
            chunkX * 16,
            chunkZ * 16
        )
    )

# Examine a chunk in a level. For each edge that's found, add a
# (chunk, direction) pair to erodeQueue. Return the number of pairs
# added to the queue.
def checkChunk(level, coords, toErode):
    
    aroundMe = [(-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1)]
    
    (TL, T, TR, L, R, BL, B, BR) = range(0, 8)
    
    neighbors = [True] * 8
    
    onPerimeter = False
    
    for i in range(len(aroundMe)):
        a = aroundMe[i]
        if (coords[0] + a[0], coords[1] + a[1]) not in level.allChunks:
            onPerimeter = True
            neighbors[i] = False
    
    if onPerimeter:
    
        # Top-left corner
        if not (neighbors[TL] or neighbors[T] or neighbors[L]):
            addCorner(coords, toErode, Erode.TL)
        
        # Top-right corner
        if not (neighbors[T] or neighbors[TR] or neighbors[R]):
            coordsRight = (coords[0] + 1, coords[1])
            addCorner(coordsRight, toErode, Erode.TR)
        
        # Bottom-right corner
        if not (neighbors[R] or neighbors[BR] or neighbors[B]):
            coordsBelowAndRight = (coords[0] + 1, coords[1] + 1)
            addCorner(coordsBelowAndRight, toErode, Erode.BR)
        
        # Bottom-left corner
        if not (neighbors[B] or neighbors[BL] or neighbors[L]):
            coordsBelow = (coords[0], coords[1] + 1)
            addCorner(coordsBelow, toErode, Erode.BL)
        
        # Top-left corner (inverted)
        if neighbors[L] and neighbors[T] and not neighbors[TL]:
            addCorner(coords, toErode, Erode.BR)
        
        # Top-right corner (inverted)
        if neighbors[T] and neighbors[R] and not neighbors[TR]:
            coordsRight = (coords[0] + 1, coords[1])
            addCorner(coordsRight, toErode, Erode.BL)
        
        # Bottom-right corner (inverted)
        if neighbors[R] and neighbors[B] and not neighbors[BR]:
            coordsBelowAndRight = (coords[0] + 1, coords[1] + 1)
            addCorner(coordsBelowAndRight, toErode, Erode.TL)
        
        # Bottom-left corner (inverted)
        if neighbors[B] and neighbors[L] and not neighbors[BL]:
            coordsBelow = (coords[0], coords[1] + 1)
            addCorner(coordsBelow, toErode, Erode.TR)
        
        if neighbors[L]:
            if not (neighbors[T] or neighbors[TL]):
                addEdge(coords, toErode, Erode.HE)
            
            if not (neighbors[B] or neighbors[BL]):
                coordsBelow = (coords[0], coords[1] + 1)
                addEdge(coordsBelow, toErode, Erode.HE)
        
        if neighbors[T]:
            if not (neighbors[L] or neighbors[TL]):
                addEdge(coords, toErode, Erode.VE)
        
            if not (neighbors[R] or neighbors[TR]):
                coordsRight = (coords[0] + 1, coords[1])
                addEdge(coordsRight, toErode, Erode.VE)
                
        # These "checkerboard" cases are kind of weird.
        if neighbors[TL] and not (neighbors[L] or neighbors[T]):
            addEdge(coords, toErode, Erode.HE)
            addEdge(coords, toErode, Erode.VE)
        
        if neighbors[TR] and not (neighbors[R] or neighbors[T]):
            coordsRight = (coords[0] + 1, coords[1])
            addEdge(coordsRight, toErode, Erode.HE)
            addEdge(coordsRight, toErode, Erode.VE)
    
    return len(toErode)

def get_info_text():
    return "\n".join([
        "bestofboth version %s" % VERSION_STRING,
        "(a tool for smoothing terrain discontinuities in Minecraft worlds)",
        "http://github.com/gmcnew/bestofboth",
        ""])

def get_usage_text():
    usage = """
    bestofboth --find-edges <path_to_world>
    bestofboth --smooth <path_to_world>"""

    usageWithSmooth = """
    bestofboth --find-edges <path_to_world>
    bestofboth --smooth <path_to_world> [--width <1-16>]"""
    
    # A paragraph is a list of lines.
    paragraphs = [[usage]]
    
    paragraphs.append(textwrap.wrap(
        "This script must be run in two steps. The first is the " \
        "--find-edges step, which examines a world and finds its edges. " \
        "Next is the --smooth step, which smooths edges by carving a river " \
        "between old chunks and newly-generated ones."))
    
    paragraphs.append(textwrap.wrap(
        "You can run the --smooth step multiple times as players explore " \
        "edges and cause new chunks to be generated along them. Eventually, " \
        "if all chunks next to edges have been generated, the script will " \
        "report that the map is perfectly smoothed. At this point further use " \
        "of the script is unnecessary."))
    
    paragraphs.append(
        ["Typical use:"] +
        [("    %s" % x) for x in [
            "bestofboth --find-edges <path_to_world>",
            "<upgrade Minecraft to a version with new terrain generation code>",
            "...",
            "<play Minecraft, explore, and cause new terrain to be generated>",
            "bestofboth --smooth <path_to_world>",
            "...",
            "<more exploration and edge discovery>",
            "bestofboth --smooth <path_to_world>",
            "...",
            "<finish exploring edges and new terrain along them>",
            "bestofboth --smooth <path_to_world>",
            ]
        ]
    )
    
    return "\n\n".join(["\n".join(p) for p in paragraphs])
    
def main():
    
    random.seed(0)
    
    parser = optparse.OptionParser(usage = get_usage_text())
    parser.add_option("--ice-wall",
                    dest="ice_wall",
                    metavar = "path",
                    help="path to the world to make an ice wall around")
    parser.add_option("--find-edges",
                    dest="find_edges",
                    metavar = "path",
                    help="path to the world to examine")
    parser.add_option("--smooth",
                    dest="smooth",
                    metavar = "path",
                    help="path to the world to smooth")
    parser.add_option("--fix-sea-level",
                    dest="fix_sea_level",
                    metavar = "path",
                    help="path to the world to be given a 1.8+ sea level")
    """
    parser.add_option("--width", dest="width", 
                    default = "16",
                    help="width of the river")
    """
    
    print(get_info_text())

    (options, args) = parser.parse_args()
    
    worldDir = options.find_edges or options.smooth or options.fix_sea_level or options.ice_wall
    if worldDir:
        edgeFilePath = os.path.join(worldDir, "edges.txt")
    
    errorText = None
    if options.find_edges and options.smooth:
        errorText = "--find-edges and --smooth can't be specified " \
            "at the same time. Please run with --find-edges first, " \
            "then run with --smooth."
    elif not (options.find_edges or options.smooth or options.fix_sea_level or options.ice_wall):
        parser.print_help()
    elif not os.path.exists(os.path.join(worldDir, "level.dat")):
        errorText = "'%s' is not a Minecraft world directory (no " \
            "level.dat file was found)." % (worldDir)
    elif options.smooth and not os.path.exists(edgeFilePath):
        errorText = "Edge file '%s' does not exist. Run with " \
            "--find-edges to create the edge file, which must exist when " \
            "--smooth is specified." \
            % (edgeFilePath)
    elif options.find_edges and os.path.exists(edgeFilePath):
        errorText = "Edge file '%s' already exists. Did you mean " \
            "to specify --smooth instead?" \
            % (edgeFilePath)
    
    
    if errorText:
        parser.error("\n" + "\n".join(textwrap.wrap(errorText)))
    
    """
    elif options.width and (int(options.width) < 1 or int(options.width) > 16):
        parser.error("--width must be between 1 and 16 (inclusive)")
    """
    
    # Phew! Now that the arguments have been validated...
    if options.ice_wall:
        ice_wall(worldDir)
    elif options.find_edges:
        find_edges(worldDir, edgeFilePath)
    elif options.smooth:
        # TODO: Fix the "--width" argument.
        #smooth(options.smooth, edgeFilePath, options.width)
        startTime = time.time()
        smooth(worldDir, edgeFilePath)
    elif options.fix_sea_level:
        fix_sea_level(worldDir, edgeFilePath)

if __name__ == "__main__":
    main()
