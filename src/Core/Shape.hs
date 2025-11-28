module Core.Shape (
  normalizeAxis,
  inferReshape,
  broadcastShapes,
) where

import Error

normalizeAxis :: Int -> [Int] -> Either TensorError Int
normalizeAxis axis dims =
  let rank = length dims
      axis'
        | axis < 0 = rank + axis
        | otherwise = axis
   in if axis' < 0 || axis' >= rank
        then Left (OutOfBounds ("Axis " <> show axis <> " is out of bounds for shape " <> show dims))
        else Right axis'

inferReshape :: [Int] -> [Int] -> Either TensorError [Int]
inferReshape oldShape newShape =
  let total = product oldShape
      minusOnes = length (filter (== (-1)) newShape)
   in case minusOnes of
        n | n > 1 -> Left (ShapeMismatch "Only one dimension can be inferred (-1)")
        _ ->
          let knownProduct = product (filter (/= (-1)) newShape)
           in if total `mod` knownProduct /= 0
                then Left (ShapeMismatch "Cannot infer reshape, element counts mismatch")
                else
                  let inferred =
                        map
                          (\d -> if d == (-1) then total `div` knownProduct else d)
                          newShape
                   in if product inferred /= total
                        then Left (ShapeMismatch "Reshape inference failed")
                        else Right inferred

broadcastShapes :: [Int] -> [Int] -> Either TensorError [Int]
broadcastShapes a b = go (reverse a) (reverse b) []
  where
    go [] [] acc = Right (reverse acc)
    go xs [] acc = Right (reverse acc <> reverse xs)
    go [] ys acc = Right (reverse acc <> reverse ys)
    go (x : xs) (y : ys) acc
      | x == y = go xs ys (x : acc)
      | x == 1 = go xs ys (y : acc)
      | y == 1 = go xs ys (x : acc)
      | otherwise = Left (BroadcastError ("Cannot broadcast " <> show a <> " and " <> show b))
