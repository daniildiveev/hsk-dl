module Core.Random (
  randu,
  randn,
  randRange,
) where

import Control.Monad (replicateM)
import Core.Tensor (numel)
import System.Random (randomRIO)

randu :: [Int] -> IO [Double]
randu dims = replicateM (numel dims) (randomRIO (0, 1))

randRange :: [Int] -> Double -> Double -> IO [Double]
randRange dims lo hi = replicateM (numel dims) (randomRIO (lo, hi))

randn :: [Int] -> IO [Double]
randn dims = do
  let n = numel dims
  pairs <- replicateM ((n + 1) `div` 2) boxMuller
  let vals = concatMap (\(a, b) -> [a, b]) pairs
  pure (take n vals)

boxMuller :: IO (Double, Double)
boxMuller = do
  u1 <- randomRIO (1e-7, 1) -- avoid log 0
  u2 <- randomRIO (0, 1)
  let r = sqrt (-2 * log u1)
      theta = 2 * pi * u2
  pure (r * cos theta, r * sin theta)
