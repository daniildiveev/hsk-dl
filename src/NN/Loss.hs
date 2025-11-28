module NN.Loss (
  mseLoss,
  nllLoss,
) where

import Control.Monad (when)
import Core.Tensor

mseLoss :: Tensor -> Tensor -> IO Tensor
mseLoss prediction target = do
  Control.Monad.when (shape prediction /= shape target) $ error $ "mseLoss: shapes do not match, got " <> show (shape prediction) <> " and " <> show (shape target)
  diff <- sub prediction target
  let vals = map (\x -> x * x) (values diff)
      reqGrad = requiresGrad diff
      backwardOps =
        ([BackwardOp diff (\up -> zipWith (\u x -> u * 2 * x) up (values diff)) | reqGrad])
  squared <- newTensor (shape diff) vals reqGrad backwardOps
  meanAll squared

nllLoss :: Tensor -> [Int] -> IO Tensor
nllLoss preds targets = do
  let eps = 1e-9
  (batch, classes) <-
    case shape preds of
      [b, c] -> pure (b, c)
      other -> error $ "nllLoss expects shape [batch, classes], got " <> show other
  Control.Monad.when (length targets /= batch) $ error "nllLoss: target length does not match batch size"
  Control.Monad.when (any (\c -> c < 0 || c >= classes) targets) $ error "nllLoss: target index out of range"
  let pick i c = values preds !! (i * classes + c)
      losses =
        [ -log (max eps (pick i cls))
        | (i, cls) <- zip [0 ..] targets
        ]
      lossVal = sum losses / fromIntegral batch
      reqGrad = requiresGrad preds
      gradFn [g] =
        concat
          [ [ if c == cls
                then (-(g / fromIntegral batch)) / max eps (pick i cls)
                else 0
            | c <- [0 .. classes - 1]
            ]
          | (i, cls) <- zip [0 ..] targets
          ]
      gradFn _ = error "nllLoss: unexpected upstream gradient shape"
      backwardOps =
        ([BackwardOp preds gradFn | reqGrad])
  newTensor [1] [lossVal] reqGrad backwardOps
