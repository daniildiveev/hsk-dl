module NN.Optim.SGD (
  sgdStep,
  zeroGradParameters,
) where

import Control.Monad (forM_)
import Core.Tensor
import Data.IORef

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
