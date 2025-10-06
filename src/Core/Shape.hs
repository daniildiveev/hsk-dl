module Core.Shape (
  normalizeAxis,
  inferReshape,
  broadcastShapes,
) where

import Error

normalizeAxis :: Int -> [Int] -> Either TensorError Int
normalizeAxis _ _ = Left (NotImplemented "normalizeAxis")

inferReshape :: [Int] -> [Int] -> Either TensorError [Int]
inferReshape _ _ = Left (NotImplemented "inferReshape")

broadcastShapes :: [Int] -> [Int] -> Either TensorError [Int]
broadcastShapes _ _ = Left (NotImplemented "broadcastShapes")
