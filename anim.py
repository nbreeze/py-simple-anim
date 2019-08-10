'''
    A basic, general-purpose module that provides classes and functions related to 3D animation and transforms.
'''

import math
import numpy as np
from transformations import quaternion_from_matrix, quaternion_matrix
from typing import Union, List, Dict


class Matrix4x4:

    '''
        A singleton containing matrix 3D transformation functions used by this module.
        
        Some conventions that must be followed when using matrices with this library:
        - Row-major storage and notation.
        - Translation of a matrix must be stored on the 4th row.
        - All rotations are represented as quaternions in the format [w, x, y, z].
        - Rotational axes must be aligned with positional axes.
        - A positive rotation implies a counterclockwise rotation around the axis.

        If a source transform does not follow these parameters, invert the axes or transpose the matrix to make it compatible.
        Alternatively, use a CoordinateSystemConverter to automate this.
    '''

    @classmethod
    def getTranslation( cls, m: np.matrix ) -> [float, float, float]:

        '''
            Gets the translation of the transform in the 4th row of the matrix.
        '''

        return [
            m.item(3, 0),
            m.item(3, 1),
            m.item(3, 2)
        ]

    @classmethod
    def getRotation( cls, m: np.matrix ) -> list:

        return quaternion_from_matrix( m )

    @classmethod
    def setRotation( cls, m: np.matrix, rotation: list ):

        rmat = quaternion_matrix( rotation )

        m.itemset( 0, 0, rmat.item(0, 0) )
        m.itemset( 0, 1, rmat.item(0, 1) )
        m.itemset( 0, 2, rmat.item(0, 2) )
        m.itemset( 1, 0, rmat.item(1, 0) )
        m.itemset( 1, 1, rmat.item(1, 1) )
        m.itemset( 1, 2, rmat.item(1, 2) )
        m.itemset( 2, 0, rmat.item(2, 0) )
        m.itemset( 2, 1, rmat.item(2, 1) )
        m.itemset( 2, 2, rmat.item(2, 2) )

    @classmethod
    def buildRotation( cls, rotation: list ) -> np.matrix:
        return np.matrix( quaternion_matrix( rotation ) )

    @classmethod
    def rotate( cls, m: np.matrix, rotation: list ) -> np.matrix:

        return m @ cls.buildRotation( rotation )

    @classmethod
    def setTranslation( cls, m: np.matrix, pos: list ):

        m.itemset( 3, 0, pos[0] )
        m.itemset( 3, 1, pos[1] )
        m.itemset( 3, 2, pos[2] )

    @classmethod
    def buildTranslation( cls, pos: list ) -> np.matrix:

        m = np.matrix( np.identity(4) )
        cls.setTranslation( m, pos )
        return m

    @classmethod
    def translate( cls, m: np.matrix, pos: list ) -> np.matrix:
        return m @ cls.buildTranslation( pos )

    @classmethod
    def getScale( cls, m: np.matrix ) -> list:

        scale = [
            math.sqrt( m.item(0, 0) ** 2 + m.item(0, 1) ** 2 + m.item(0, 2) ** 2 ),
            math.sqrt( m.item(1, 0) ** 2 + m.item(1, 1) ** 2 + m.item(1, 2) ** 2 ),
            math.sqrt( m.item(2, 0) ** 2 + m.item(2, 1) ** 2 + m.item(2, 2) ** 2 )
        ]

        return scale

    @classmethod
    def setScale( cls, m: np.matrix, scale: list ):

        m.itemset( 0, 0, scale[0] )
        m.itemset( 1, 1, scale[1] )
        m.itemset( 2, 2, scale[2] )

    @classmethod
    def buildScale( cls, scale: list ) -> np.matrix:

        m = np.matrix( np.identity(4) )
        cls.setScale( m, scale )
        return m

    @classmethod
    def compose( cls, scale: list=[1, 1, 1], rotation: list=[1, 0, 0, 0], translation: list=[0, 0, 0] ):

        return cls.buildScale( scale ) @ cls.buildRotation( rotation ) @ cls.buildTranslation( translation )

    @classmethod
    def decompose( cls, m: np.matrix ):

        t = cls.getTranslation( m )
        s = cls.getScale( m )

        norm = np.identity(4)
        norm.itemset( 0, 0, m.item(0, 0) / s[0] )
        norm.itemset( 0, 1, m.item(0, 1) / s[0] )
        norm.itemset( 0, 2, m.item(0, 2) / s[0] )
        norm.itemset( 1, 0, m.item(1, 0) / s[1] )
        norm.itemset( 1, 1, m.item(1, 1) / s[1] )
        norm.itemset( 1, 2, m.item(1, 2) / s[1] )
        norm.itemset( 2, 0, m.item(2, 0) / s[2] )
        norm.itemset( 2, 1, m.item(2, 1) / s[2] )
        norm.itemset( 2, 2, m.item(2, 2) / s[2] )

        return s, cls.getRotation( norm ), t


class Skeleton:

    '''
        A class that maintains the hierarchy of bones and their local/global transforms.
    '''

    class Bone:

        def __init__(self, parent = None):

            self.name = ''
            self.children = [] #type: List[Bone]
            self.parent = None #type: Bone
            self.skeleton: Skeleton = None
            
            self.setLocalTransformMatrix( np.matrix( np.identity(4) ) )
            self.globalTransformMatrix = np.matrix( np.identity(4) )
            self._globalTransformDirty = False

            self.setParent( parent )

        def _markGlobalTransformDirtyInternal( self ):
            self._globalTransformDirty = True

        def _setSkeletonToParentInternal( self, keepRootSkeleton=False ):

            oldSkeleton = self.skeleton
            skeleton: Skeleton = None
            if self.parent is not None:
                skeleton = self.parent.skeleton
            elif keepRootSkeleton and oldSkeleton is not None and oldSkeleton.root is self:
                skeleton = oldSkeleton

            if skeleton is not oldSkeleton:

                self.skeleton = skeleton

                if oldSkeleton is not None:
                    oldSkeleton._onBoneRemovedInternal( self )

                if skeleton is not None:
                    skeleton._onBoneAddedInternal( self )

                self._markGlobalTransformDirtyInternal()

        def getLocalTransformMatrix( self ) -> np.matrix:

            '''
                Returns a copy of this bone's local transform matrix, relative to parent.
            '''

            return self.transformMatrix.copy()

        def markGlobalTransformDirty( self ):
            self.foreach( lambda x: x._markGlobalTransformDirtyInternal() )

        def setParent( self, parent=None ):

            '''
                Sets the parent of this bone to another. Clears the parent if None is given.
            '''

            oldParent = self.parent
            self.parent = parent
            if oldParent is not parent:
                if oldParent is not None:
                    try:
                        oldParent.children.remove( self )
                    except ValueError:
                        pass
                
                if parent is not None:
                    parent.children.append( self )

                # Notify skeletons of removed/added bones, if any.
                self.foreach( lambda x: x._setSkeletonToParentInternal() )

        def globalToLocalTransformMatrix( self, m: np.matrix ) -> np.matrix:

            '''
                Converts the given global transform into a local transform in respect to this bone's parent.
                If this bone has no parent, the given global transform is returned.
            '''

            if self.parent is None:
                return m.copy()

            return m @ np.linalg.inv( self.parent.getGlobalTransformMatrix() )

        def setLocalTransformMatrix( self, m: np.matrix ):

            '''
                Sets the local transform matrix of this bone.
            '''

            self.transformMatrix = np.matrix( m.copy() )
            self.markGlobalTransformDirty()

        def calcGlobalTransformMatrix( self, localTransformMatrix: np.matrix ) -> np.matrix:

            '''
                Calculates the global transform matrix for the local transform given,
                without setting the stored global transform of this node.
            '''

            if self.parent is None:
                return localTransformMatrix.copy()

            return localTransformMatrix @ self.parent.getGlobalTransformMatrix()

        def getGlobalTransformMatrix( self, force=False ) -> np.matrix:

            '''
                Calculates the global transform matrix for this bone and returns it.
                \nReturns the last calculated global transform matrix if not marked dirty.
            '''

            if self._globalTransformDirty or force:
                self.globalTransformMatrix = self.calcGlobalTransformMatrix( self.transformMatrix )
                self._globalTransformDirty = False

            return self.globalTransformMatrix.copy()

        def setGlobalTransformMatrix( self, m: np.matrix ) -> np.matrix:
            self.setLocalTransformMatrix( self.globalToLocalTransformMatrix( m ) )

        def foreach( self, func, *args ):
            func(self, *args)
            for child in self.children:
                child.foreach( func, *args )

    def toList( self, useCache=True ) -> List[ Bone ]:

        '''
            Converts the skeleton into a list of bones, generated pre-order.

            \nWARNING: If useCache is True, it will return the cached list of bones, not a copy. If you need to modify the list, make a copy of it first before doing so!
        '''

        if useCache and not self._boneListDirty:
            return self._boneListCached

        indices = []

        self.foreach( lambda x: indices.append( x ) )

        self._boneListCached = indices
        self._boneListDirty = False

        return indices

    def toDict( self, useCache=True ) -> Dict[ str, Bone ]:

        '''
            Converts the skeleton into a dictionary, keys being the bone names and values the bones themselves.

            \nWARNING: If useCache is True, it will return the cached dict of bones, not a copy. If you need to modify the dict, make a copy of it first before doing so!
        '''

        if useCache and not self._boneDictDirty:
            return self._boneDictCached

        d = {}

        # Can't assign values in lambdas, so no other choice but to do it this way.
        indices = self.toList( useCache=useCache )
        for bone in indices:
            d[bone.name] = bone

        self._boneDictCached = d
        self._boneDictDirty = False

        return d

    def getBoneCount( self ) -> int:

        '''
            Returns the number of bones in the skeleton.
        '''

        return len( self.toList() )
        
    def findBoneByName( self, name: str ) -> Bone:

        '''
            Retrieves a bone by name in the skeleton.
        '''

        d = self.toDict() # TODO: Use a cache if not dirty.
        if name in d:
            return d[name]
        return None

    def _onBoneAddedInternal( self, bone: Bone ):

        self._boneListDirty = True
        self._boneDictDirty = True

        self.onBoneAdded( bone )

    def _onBoneRemovedInternal( self, bone: Bone ):

        self._boneListDirty = True
        self._boneDictDirty = True

        self.onBoneRemoved( bone )

    def onBoneAdded( self, bone: Bone ):
        '''
            Called when a bone was added to this skeleton's hierarchy.
        '''
        pass

    def onBoneRemoved( self, bone: Bone ):
        '''
            Called when a bone is removed from this skeleton's hierarchy.
        '''
        pass

    def setRootBone( self, bone: Bone ):

        oldRootBone = self.root
        if oldRootBone is bone:
            return

        if oldRootBone is not None:
            oldRootBone.skeleton = None
            self._onBoneRemovedInternal( oldRootBone )
            oldRootBone._markGlobalTransformDirtyInternal()
            oldRootBone.foreach( lambda x: x._setSkeletonToParentInternal() )

        self.root = bone

        if bone is not None:
            bone.skeleton = self
            self._onBoneAddedInternal( bone )
            bone._markGlobalTransformDirtyInternal()
            self.foreach( lambda x: x._setSkeletonToParentInternal( keepRootSkeleton=True ) )
        
    def foreach( self, func, *args ):
        if self.root is None:
            return
        self.root.foreach( func, *args )

    def __init__( self ):
        self.root = None # type: Bone

        self._boneListDirty = False
        self._boneDictDirty = False
        self._boneListCached = []
        self._boneDictCached = {}

class Animation:
    '''
        A structure that contains animation keyframes.
        Keyframes must be sorted chronologically.
    '''

    class Keyframe:

        def __init__( self, animation ):

            self.animation = animation # type: Animation

        def getTime( self ) -> Union[int, float]:
            '''
                Returns the time which the keyframe starts.
            '''
            return 0

        def getDuration( self ) -> Union[int, float]:
            '''
                Returns how long this keyframe lasts.
            '''
            return 0

        def getBoneLocalTransform( self, bone: Skeleton.Bone ) -> np.matrix:
            '''
                Gets information about a bone's transform (position and rotation) in this frame.
            '''
            return None

    def __init__(self):

        self.name = ''
        self.keyframes = [] # type: List[Animation.Keyframe]

    def getDuration( self ) -> Union[int, float]:

        '''
            Returns how long the animation lasts.
        '''

        return 0

    def getKeyframeCount( self ) -> int:

        '''
            How many Keyframes this animation contains.
        '''

        return len( self.keyframes )

class CoordinateSystemConverter:
    '''
        A singleton class used to convert transformation matrices to Matrix4x4 format.
    '''

    @staticmethod
    def convertTransformMatrix( matrix: np.matrix ) -> np.matrix:
        return matrix

def animateBoneToKeyframe( bone: Skeleton.Bone, keyframe: Animation.Keyframe, conv: CoordinateSystemConverter=None ):
    
    '''
        Animates a bone to a Keyframe's local transform for the bone.
    '''

    altm = keyframe.getBoneLocalTransform( bone )
    if altm is None:
        altm = np.matrix( np.identity(4) ) # Default to identity matrix for transform.

    if conv is not None:
        altm = conv.convertTransformMatrix( altm )
    
    bone.setLocalTransformMatrix( altm )

def animateBoneToSkeleton( bone: Skeleton.Bone, skeleton: Skeleton, conv: CoordinateSystemConverter=None ):

    '''
        Animates a bone to the skeleton of a bone with the same name.
    '''

    bindBone = skeleton.findBoneByName( bone.name )
    if bindBone is None:
        return

    lbtm = bindBone.getLocalTransformMatrix()
    if conv is not None:
        lbtm = conv.convertTransformMatrix( lbtm )

    bone.setLocalTransformMatrix( lbtm )

class AnimationRetargetListener:

    '''
        A class that listens for certain events during the animation retargeting process.
    '''

    def onPreProcessAnimation( self, 
        animation: Animation, 
        sourceSkeleton: Skeleton, 
        targetSkeleton: Skeleton, 
        sourceBindSkeleton: Skeleton, 
        targetBindSkeleton: Skeleton,
        targetBaseSkeleton: Skeleton ):

        pass

    def onPostProcessAnimation( self, 
        animation: Animation, 
        sourceSkeleton: Skeleton, 
        targetSkeleton: Skeleton, 
        sourceBindSkeleton: Skeleton, 
        targetBindSkeleton: Skeleton,
        targetBaseSkeleton: Skeleton ):

        pass

    def onPreFrame( self, 
        animation: Animation, 
        frameNum: int, 
        keyframe: Animation.Keyframe, 
        sourceSkeleton: Skeleton,
        targetSkeleton: Skeleton, 
        sourceBindSkeleton: Skeleton, 
        targetBindSkeleton: Skeleton,
        targetBaseSkeleton: Skeleton ):

        pass

    def onPostFrame( self, 
        animation: Animation, 
        frameNum: int, 
        keyframe: Animation.Keyframe, 
        sourceSkeleton: Skeleton, 
        targetSkeleton: Skeleton, 
        sourceBindSkeleton: Skeleton, 
        targetBindSkeleton: Skeleton,
        targetBaseSkeleton: Skeleton ):

        pass

    def onPreProcessTargetBone( self, 
        animation: Animation, 
        frameNum: int, 
        keyframe: Animation.Keyframe, 
        targetBone: Skeleton.Bone ):

        pass

    def onPostProcessTargetBone( self, 
        animation: Animation, 
        frameNum: int, 
        keyframe: Animation.Keyframe, 
        targetBone: Skeleton.Bone ):

        pass

    def onPreCalcBoneTransform( self,
        animation: Animation,
        frameNum: int,
        keyframe: Animation.Keyframe,
        sourceBone: Skeleton.Bone,
        targetBone: Skeleton.Bone,
        sourceBindBone: Skeleton.Bone,
        targetBindBone: Skeleton.Bone,
        changeTransformMatrix: np.matrix ):

        '''
            Called before changeTransformMatrix is applied onto targetBone. Use this function to make any changes to changeTransformMatrix.
        '''

        pass

    def onPostCalcBoneTransform( self,
        animation: Animation,
        frameNum: int,
        keyframe: Animation.Keyframe,
        sourceBone: Skeleton.Bone,
        targetBone: Skeleton.Bone,
        sourceBindBone: Skeleton.Bone,
        targetBindBone: Skeleton.Bone,
        changeTransformMatrix: np.matrix ):

        '''
            Called after changeTransformMatrix has been applied onto targetBone.
        '''

        pass

    def shouldApplyRotation( self, targetBone: Skeleton.Bone, changeTransformMatrix: np.matrix ) -> bool:
        return True

    def shouldApplyTranslation( self, targetBone: Skeleton.Bone, changeTransformMatrix: np.matrix ) -> bool:
        return False

def retargetAnimation( animation: Animation,
    sourceSkeleton: Skeleton,
    targetSkeleton: Skeleton,
    sourceBindPoseSkeleton: Skeleton,
    targetBindPoseSkeleton: Skeleton,
    targetBoneMap: dict,
    animConv: CoordinateSystemConverter = None,
    sourceConv: CoordinateSystemConverter = None,
    targetConv: CoordinateSystemConverter = None,
    resetSourceSkeletonPerFrame = True,
    resetTargetSkeletonPerFrame = True,
    animRetargetListener: AnimationRetargetListener = None,
    targetBaseSkeleton: Skeleton=None ):

    '''
        Applies an animation from the source skeleton onto the target skeleton, taking into account changes in bone positions and orientation between skeletons.

        - `Animation` animation - The `Animation` that will be played and recorded. This will be played on sourceSkeleton.
        - `Skeleton` sourceSkeleton - The `Skeleton` that will act out the animation.
        - `Skeleton` targetSkeleton - The `Skeleton` that will copy sourceSkeleton's global movements.
        - `Skeleton` sourceBindPoseSkeleton - `Skeleton` that has all of sourceSkeleton's default bone positions.
        - `Skeleton` targetBindPoseSkeleton - `Skeleton` that has all of targetSkeleton's default bone positions. This skeleton must be posed similarly to sourceBindPoseSkeleton as much as possible!
        - `Dict[str, str]` targetBoneMap - A dictionary that maps targetSkeleton's bone names to sourceSkeleton's bone names. ( targetBoneMap[targetBone.name] = sourceBone.name )
        - `bool` resetSourceSkeletonPerFrame - If `True`, sourceSkeleton will be animated to sourceBindPoseSkeleton per frame before animation transforms take place.
        - `bool` resetTargetSkeletonPerFrame - If `True`, targetSkeleton will be animated to targetBindPoseSkeleton per frame before animation transforms take place.
        - `AnimationRetargetListener` animRetargetListener
        - `Skeleton` targetBaseSkeleton - If provided, it should be a `Skeleton` generated just like targetBindPoseSkeleton, but without any changes to rotation/position offsets in post; i.e. the original, unchanged target binding pose.
    '''

    def animateTargetBone( targetBone: Skeleton.Bone, keyframe: Animation.Keyframe, frameNum: int ):

        if animRetargetListener is not None:
            animRetargetListener.onPreProcessTargetBone( animation, frameNum, keyframe, targetBone )

        # Get this bone's default local transform.
        targetBindBone = targetBindPoseSkeleton.findBoneByName( targetBone.name )

        lbtm = targetBindBone.getLocalTransformMatrix()
        if targetConv is not None:
            lbtm = targetConv.convertTransformMatrix( lbtm )

        gbtm = targetBindBone.getGlobalTransformMatrix()
        if targetConv is not None:
            gbtm = targetConv.convertTransformMatrix( gbtm )

        ltm = targetBone.getLocalTransformMatrix()

        if targetBone.name in targetBoneMap:

            translatedName = targetBoneMap[ targetBone.name ]

            # In global space, find the animation rotation matrix from start (global base transform) -> finish (current global transform).
            sourceBindBone = sourceBindPoseSkeleton.findBoneByName( translatedName )
            sgbtm = sourceBindBone.getGlobalTransformMatrix()
            if sourceConv is not None:
                sgbtm = sourceConv.convertTransformMatrix( sgbtm )

            sourceBone = sourceSkeleton.findBoneByName( translatedName )
            sgtm = sourceBone.getGlobalTransformMatrix()

            # Get change matrix.
            # Ax = B, so x = A^-1 * B
            sgbtmtosgtm = np.linalg.inv( sgbtm ) @ sgtm

            if animRetargetListener is not None:
                animRetargetListener.onPreCalcBoneTransform( animation, frameNum, keyframe, sourceBone, targetBone, sourceBindBone, targetBindBone, sgbtmtosgtm )

            # Apply the change in global space to our own global base transform, then convert the matrix to joint space.
            ltm = targetBone.globalToLocalTransformMatrix( gbtm @ sgbtmtosgtm )

            shouldApplyRotation = ( animRetargetListener is None or animRetargetListener.shouldApplyRotation( targetBone, sgbtmtosgtm ) )
            shouldApplyTranslation = ( animRetargetListener is not None and animRetargetListener.shouldApplyTranslation( targetBone, sgbtmtosgtm ) )

            if not shouldApplyRotation:
                Matrix4x4.setRotation( ltm, Matrix4x4.getRotation( lbtm ) )

            if not shouldApplyTranslation:
                Matrix4x4.setTranslation( ltm, Matrix4x4.getTranslation( lbtm ) )

            if animRetargetListener is not None:
                animRetargetListener.onPostCalcBoneTransform( animation, frameNum, keyframe, sourceBone, targetBone, sourceBindBone, targetBindBone, sgbtmtosgtm )

        targetBone.setLocalTransformMatrix( ltm )

        if animRetargetListener is not None:
            animRetargetListener.onPostProcessTargetBone( animation, frameNum, keyframe, targetBone )

    if animRetargetListener is not None:
        animRetargetListener.onPreProcessAnimation( animation, sourceSkeleton, targetSkeleton, sourceBindPoseSkeleton, targetBindPoseSkeleton, targetBaseSkeleton )

    # Start animating.

    nFrames = animation.getKeyframeCount()

    for frameNum in range(nFrames):

        keyframe = animation.keyframes[ frameNum ] # type: Animation.Keyframe

        if animRetargetListener is not None:
            animRetargetListener.onPreFrame( animation, frameNum, keyframe, sourceSkeleton, targetSkeleton, sourceBindPoseSkeleton, targetBindPoseSkeleton, targetBaseSkeleton )

        if resetSourceSkeletonPerFrame:
            sourceSkeleton.foreach( animateBoneToSkeleton, sourceBindPoseSkeleton, sourceConv )

        if resetTargetSkeletonPerFrame:
            targetSkeleton.foreach( animateBoneToSkeleton, targetBindPoseSkeleton, targetConv )

        # Animate the source skeleton for this frame.
        sourceSkeleton.foreach( animateBoneToKeyframe, keyframe, animConv )
        
        # Animate the target skeleton for this frame.
        targetSkeleton.foreach( animateTargetBone, keyframe, frameNum )

        if animRetargetListener is not None:
            animRetargetListener.onPostFrame( animation, frameNum, keyframe, sourceSkeleton, targetSkeleton, sourceBindPoseSkeleton, targetBindPoseSkeleton, targetBaseSkeleton )

    if animRetargetListener is not None:
        animRetargetListener.onPostProcessAnimation( animation, sourceSkeleton, targetSkeleton, sourceBindPoseSkeleton, targetBindPoseSkeleton, targetBaseSkeleton )