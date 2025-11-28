module Main where

import Core.Tensor
import NN.Layer
import NN.Loss
import NN.Optim.SGD
import Data.IORef (IORef)
import Control.Monad (when)

xorInputs :: [Double]
xorInputs =
  [0, 0, 0, 1, 1, 0, 1, 1] -- flattened 4x2

xorTargets :: [Int]
xorTargets = [0, 1, 1, 0]

main :: IO ()
main = do
  putStrLn "Training a tiny XOR model (Linear -> ReLU -> Linear -> Softmax)..."
  l1 <- linear 2 4
  l2 <- linear 4 2
  let model = [l1, reluLayer, l2, softmaxLayer]
      params = concatMap parameters model
  input <- fromList [4, 2] xorInputs
  trainEpochs 2000 0.1 params model input
  finalOut <- forwardSequential model input
  putStrLn $ "Final predictions (rows = samples): " <> show (values finalOut)

trainEpochs :: Int -> Double -> [IORef Tensor] -> [Layer] -> Tensor -> IO ()
trainEpochs epochs lr params model input = mapM_ step [1 .. epochs]
  where
    step epoch = do
      zeroGradParameters params
      out <- forwardSequential model input
      loss <- nllLoss out xorTargets
      backward loss
      sgdStep lr params
      let [lossVal] = values loss
      Control.Monad.when (epoch `mod` 200 == 0) $ putStrLn ("epoch " <> show epoch <> ", loss=" <> show lossVal)
