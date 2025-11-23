module Core.Init (
  xavierUniform,
  heUniform,
) where

import Core.Random (randRange)

xavierUniform :: [Int] -> IO [Double]
xavierUniform dims = randRange dims (-limit) limit
  where
    fanIn = fromIntegral (headOrOne dims)
    fanOut = fromIntegral (headOrOne (drop 1 dims))
    limit = sqrt (6 / (fanIn + fanOut))

heUniform :: [Int] -> IO [Double]
heUniform dims = randRange dims (-limit) limit
  where
    fanIn = fromIntegral (headOrOne dims)
    limit = sqrt (6 / fanIn)

headOrOne :: [Int] -> Int
headOrOne [] = 1
headOrOne (x : _) = x
