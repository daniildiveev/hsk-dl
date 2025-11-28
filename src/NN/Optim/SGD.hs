module NN.Optim.SGD (
  sgdStep,
  zeroGradParameters,
) where

import Core.Tensor
import Data.IORef
import Control.Monad (forM_)

sgdStep :: Double -> [IORef Tensor] -> IO ()
sgdStep lr params =
  forM_ params $ \pRef -> do
    p <- readIORef pRef
    g <- gradValues p
    let updated = zipWith (\w gw -> w - lr * gw) (values p) g
    p' <- fromListWithGrad (shape p) updated True
    writeIORef pRef p'

zeroGradParameters :: [IORef Tensor] -> IO ()
zeroGradParameters params =
  forM_ params $ \pRef -> do
    p <- readIORef pRef
    zeroGrad p
    writeIORef pRef p
