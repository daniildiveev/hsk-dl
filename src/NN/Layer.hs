module NN.Layer (
  Layer (..),
  linear,
  reluLayer,
  softmaxLayer,
  forwardLayer,
  forwardSequential,
  parameters,
) where

import Control.Monad (foldM)
import Core.Init (xavierUniform)
import Core.Tensor
import Data.IORef

data Layer
  = Linear
      { weight :: IORef Tensor
      , bias :: IORef Tensor
      }
  | Relu
  | Softmax

linear :: Int -> Int -> IO Layer
linear inFeatures outFeatures = do
  wVals <- xavierUniform [inFeatures, outFeatures]
  let bVals = replicate outFeatures 0
  w <- fromListWithGrad [inFeatures, outFeatures] wVals True
  b <- fromListWithGrad [outFeatures] bVals True
  Linear <$> newIORef w <*> newIORef b

reluLayer :: Layer
reluLayer = Relu

softmaxLayer :: Layer
softmaxLayer = Softmax

forwardLayer :: Layer -> Tensor -> IO Tensor
forwardLayer (Linear wRef bRef) input = do
  w <- readIORef wRef
  b <- readIORef bRef
  out <- matmul input w
  addRowVector out b
forwardLayer Relu input = relu input
forwardLayer Softmax input = softmax input

forwardSequential :: [Layer] -> Tensor -> IO Tensor
forwardSequential layers input = foldM (flip forwardLayer) input layers

parameters :: Layer -> [IORef Tensor]
parameters (Linear w b) = [w, b]
parameters _ = []
