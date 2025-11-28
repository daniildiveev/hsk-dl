module Core.Tensor (
  Tensor (..),
  BackwardOp (..),
  newTensor,
  numel,
  shape,
  values,
  requiresGrad,
  gradValues,
  zeroGrad,
  detach,
  fromList,
  fromListWithGrad,
  toList,
  scalar,
  zeros,
  ones,
  full,
  imageToTensor,
  add,
  sub,
  mul,
  matmul,
  addRowVector,
  relu,
  softmax,
  sumAll,
  meanAll,
  backward,
) where

import Control.Monad (foldM, forM_)
import Data.IORef
import qualified Data.Set as Set
import System.IO.Unsafe (unsafePerformIO)

data Tensor = Tensor
  { tensorId :: !Int
  , tShape :: ![Int]
  , tValues :: ![Double]
  , tRequiresGrad :: !Bool
  , tGrad :: !(IORef [Double])
  , tBackward :: ![BackwardOp]
  }

data BackwardOp = BackwardOp
  { dependency :: Tensor
  , gradTransform :: [Double] -> [Double]
  }

instance Eq Tensor where
  a == b = tensorId a == tensorId b

instance Show Tensor where
  show t =
    "Tensor(shape=" <> show (tShape t) <> ", values=" <> show (take 6 (tValues t)) <> suffix <> ")"
    where
      suffix =
        if length (tValues t) > 6
          then "..."
          else ""

{-# NOINLINE globalTensorId #-}
globalTensorId :: IORef Int
globalTensorId = unsafePerformIO (newIORef 0)

nextTensorId :: IO Int
nextTensorId = atomicModifyIORef' globalTensorId (\i -> (i + 1, i))

numel :: [Int] -> Int
numel [] = 1
numel ds = product ds

shape :: Tensor -> [Int]
shape = tShape

values :: Tensor -> [Double]
values = tValues

toList :: Tensor -> [Double]
toList = values

requiresGrad :: Tensor -> Bool
requiresGrad = tRequiresGrad

gradValues :: Tensor -> IO [Double]
gradValues = readIORef . tGrad

zeroGrad :: Tensor -> IO ()
zeroGrad t = writeIORef (tGrad t) (replicate (length (tValues t)) 0)

detach :: Tensor -> IO Tensor
detach t = fromListWithGrad (tShape t) (tValues t) (tRequiresGrad t)

newTensor :: [Int] -> [Double] -> Bool -> [BackwardOp] -> IO Tensor
newTensor dims vals reqGrad backwards = do
  let expected = numel dims
  if expected /= length vals
    then error $ "Shape " <> show dims <> " expects " <> show expected <> " values, got " <> show (length vals)
    else pure ()
  gradRef <- newIORef (replicate (length vals) 0)
  tid <- nextTensorId
  pure
    Tensor
      { tensorId = tid
      , tShape = dims
      , tValues = vals
      , tRequiresGrad = reqGrad
      , tGrad = gradRef
      , tBackward = backwards
      }

fromList :: [Int] -> [Double] -> IO Tensor
fromList dims vals = newTensor dims vals False []

fromListWithGrad :: [Int] -> [Double] -> Bool -> IO Tensor
fromListWithGrad dims vals reqGrad = newTensor dims vals reqGrad []

scalar :: Double -> Bool -> IO Tensor
scalar x reqGrad = fromListWithGrad [1] [x] reqGrad

zeros :: [Int] -> IO Tensor
zeros dims = full dims 0 False

ones :: [Int] -> IO Tensor
ones dims = full dims 1 False

full :: [Int] -> Double -> Bool -> IO Tensor
full dims v reqGrad = newTensor dims (replicate (numel dims) v) reqGrad []

imageToTensor :: Real a => [[a]] -> IO Tensor
imageToTensor rows =
  case rows of
    [] -> error "imageToTensor: empty image"
    r0 : _ ->
      if any (\r -> length r /= length r0) rows
        then error "imageToTensor: inconsistent row lengths"
        else fromList [length rows, length r0] (map realToFrac (concat rows))

zipWithTensor ::
  String ->
  (Double -> Double -> Double) ->
  ( [Double] -> [Double] -- gradient for left
  , [Double] -> [Double] -- gradient for right
  ) ->
  Tensor ->
  Tensor ->
  IO Tensor
zipWithTensor name f (gradLeft, gradRight) a b = do
  let sa = tShape a
      sb = tShape b
  if sa /= sb
    then error $ name <> ": shapes do not match, got " <> show sa <> " and " <> show sb
    else pure ()
  let vals = zipWith f (tValues a) (tValues b)
      reqGrad = tRequiresGrad a || tRequiresGrad b
      backwardLeft =
        if tRequiresGrad a
          then [BackwardOp a gradLeft]
          else []
      backwardRight =
        if tRequiresGrad b
          then [BackwardOp b gradRight]
          else []
  newTensor (tShape a) vals reqGrad (backwardLeft <> backwardRight)

add :: Tensor -> Tensor -> IO Tensor
add = zipWithTensor "add" (+) (id, id)

sub :: Tensor -> Tensor -> IO Tensor
sub = zipWithTensor "sub" (-) (id, map negate)

mul :: Tensor -> Tensor -> IO Tensor
mul a b =
  zipWithTensor
    "mul"
    (*)
    ( \upstream -> zipWith (*) upstream (tValues b)
    , \upstream -> zipWith (*) upstream (tValues a)
    )
    a
    b

relu :: Tensor -> IO Tensor
relu t =
  unary
    "relu"
    (\x -> if x > 0 then x else 0)
    (\x -> if x > 0 then 1 else 0)
    t

unary :: String -> (Double -> Double) -> (Double -> Double) -> Tensor -> IO Tensor
unary name f gradF t = do
  let vals = map f (tValues t)
      backward =
        if tRequiresGrad t
          then
            [ BackwardOp
                t
                (\up -> zipWith (*) up (map gradF (tValues t)))
            ]
          else []
  newTensor (tShape t) vals (tRequiresGrad t) backward

sumAll :: Tensor -> IO Tensor
sumAll t = do
  let s = sum (tValues t)
      backward =
        if tRequiresGrad t
          then
            [BackwardOp t (\[g] -> replicate (length (tValues t)) g)]
          else []
  newTensor [1] [s] (tRequiresGrad t) backward

meanAll :: Tensor -> IO Tensor
meanAll t = do
  let n = fromIntegral (length (tValues t))
  summed <- sumAll t
  unary "meanAll" (/ n) (const (1 / n)) summed

softmax :: Tensor -> IO Tensor
softmax t =
  case tShape t of
    [n] -> softmax =<< reshape1to2 t 1 n
    [batch, classes] -> do
      let rows = chunk classes (tValues t)
          probs = concatMap softmaxRow rows
          reqGrad = tRequiresGrad t
          backward =
            if reqGrad
              then
                [ BackwardOp t (\up -> softmaxBackward batch classes probs up)
                ]
              else []
      newTensor [batch, classes] probs reqGrad backward
    other -> error $ "softmax: expected rank-1 or rank-2 tensor, got shape " <> show other

softmaxRow :: [Double] -> [Double]
softmaxRow xs =
  let m = maximum xs
      exps = map (exp . subtract m) xs
      denom = sum exps
   in map (/ denom) exps

softmaxBackward :: Int -> Int -> [Double] -> [Double] -> [Double]
softmaxBackward batch classes probs upstream =
  concatMap step [0 .. batch - 1]
  where
    step i =
      let rowStart = i * classes
          row = take classes (drop rowStart probs)
          upRow = take classes (drop rowStart upstream)
          dotProd = sum (zipWith (*) row upRow)
       in zipWith (\s u -> s * (u - dotProd)) row upRow

matmul :: Tensor -> Tensor -> IO Tensor
matmul a b =
  case (tShape a, tShape b) of
    ([m, k1], [k2, n])
      | k1 == k2 ->
          let getA i k = tValues a !! (i * k1 + k)
              getB k j = tValues b !! (k * n + j)
              vals =
                [ sum [getA i k * getB k j | k <- [0 .. k1 - 1]]
                | i <- [0 .. m - 1]
                , j <- [0 .. n - 1]
                ]
              reqGrad = tRequiresGrad a || tRequiresGrad b
              backwardA =
                if tRequiresGrad a
                  then
                    [ BackwardOp
                        a
                        ( \up ->
                            [ sum [up !! (i * n + j) * getB k j | j <- [0 .. n - 1]]
                            | i <- [0 .. m - 1]
                            , k <- [0 .. k1 - 1]
                            ]
                        )
                    ]
                  else []
              backwardB =
                if tRequiresGrad b
                  then
                    [ BackwardOp
                        b
                        ( \up ->
                            [ sum [getA i k * up !! (i * n + j) | i <- [0 .. m - 1]]
                            | k <- [0 .. k1 - 1]
                            , j <- [0 .. n - 1]
                            ]
                        )
                    ]
                  else []
           in newTensor [m, n] vals reqGrad (backwardA <> backwardB)
    (sa, sb) ->
      error $
        "matmul: expected shapes [m,k] x [k,n], got "
          <> show sa
          <> " and "
          <> show sb

addRowVector :: Tensor -> Tensor -> IO Tensor
addRowVector matrix rowVec =
  case (tShape matrix, tShape rowVec) of
    ([rows, cols], [cols'])
      | cols == cols' ->
          let expanded = concat (replicate rows (tValues rowVec))
              vals = zipWith (+) (tValues matrix) expanded
              reqGrad = tRequiresGrad matrix || tRequiresGrad rowVec
              backwardMatrix =
                if tRequiresGrad matrix
                  then [BackwardOp matrix id]
                  else []
              backwardRow =
                if tRequiresGrad rowVec
                  then [BackwardOp rowVec (\up -> reduceRows up rows cols)]
                  else []
           in newTensor [rows, cols] vals reqGrad (backwardMatrix <> backwardRow)
      | otherwise -> error $ "addRowVector: column mismatch " <> show cols <> " vs " <> show cols'
    (s, rv) ->
      error $ "addRowVector: expected matrix [rows, cols] and row vector [cols], got " <> show s <> " and " <> show rv

reduceRows :: [Double] -> Int -> Int -> [Double]
reduceRows upstream rows cols =
  [ sum [upstream !! (r * cols + c) | r <- [0 .. rows - 1]]
  | c <- [0 .. cols - 1]
  ]

backward :: Tensor -> IO ()
backward target = do
  writeIORef (tGrad target) (replicate (length (tValues target)) 1)
  (_visited, order) <- topo target Set.empty
  forM_ order $ \t -> do
    g <- readIORef (tGrad t)
    forM_ (tBackward t) $ \(BackwardOp dep transform) -> do
      let gDep = transform g
      if tRequiresGrad dep
        then modifyIORef' (tGrad dep) (zipWith (+) gDep)
        else pure ()

topo :: Tensor -> Set.Set Int -> IO (Set.Set Int, [Tensor])
topo t visited
  | Set.member (tensorId t) visited = pure (visited, [])
  | otherwise = do
      let visited' = Set.insert (tensorId t) visited
      (visitedFinal, deps) <-
        foldM
          ( \(seen, acc) (BackwardOp dep _) -> do
              (seen', collected) <- topo dep seen
              pure (seen', acc <> collected)
          )
          (visited', [])
          (tBackward t)
      pure (visitedFinal, t : deps)

reshape1to2 :: Tensor -> Int -> Int -> IO Tensor
reshape1to2 t rows cols = do
  if rows * cols /= length (tValues t)
    then error "reshape1to2: element count mismatch"
    else
      newTensor
        [rows, cols]
        (tValues t)
        (tRequiresGrad t)
        (if tRequiresGrad t then [BackwardOp t id] else [])

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs =
  let (h, rest) = splitAt n xs
   in h : chunk n rest
