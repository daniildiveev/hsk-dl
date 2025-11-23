module NN.Loss (
  mseLoss,
  nllLoss,
) where

import Core.Tensor

mseLoss :: Tensor -> Tensor -> IO Tensor
mseLoss pred target = do
  if shape pred /= shape target
    then error $ "mseLoss: shapes do not match, got " <> show (shape pred) <> " and " <> show (shape target)
    else pure ()
  diff <- sub pred target
  let vals = map (\x -> x * x) (values diff)
      reqGrad = requiresGrad diff
      backward =
        if reqGrad
          then [BackwardOp diff (\up -> zipWith (\u x -> u * 2 * x) up (values diff))]
          else []
  squared <- newTensor (shape diff) vals reqGrad backward
  meanAll squared

nllLoss :: Tensor -> [Int] -> IO Tensor
nllLoss preds targets = do
  let eps = 1e-9
  (batch, classes) <-
    case shape preds of
      [b, c] -> pure (b, c)
      other -> error $ "nllLoss expects shape [batch, classes], got " <> show other
  if length targets /= batch
    then error "nllLoss: target length does not match batch size"
    else pure ()
  if any (\c -> c < 0 || c >= classes) targets
    then error "nllLoss: target index out of range"
    else pure ()
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
                then (-g / fromIntegral batch) / max eps (pick i cls)
                else 0
            | c <- [0 .. classes - 1]
            ]
          | (i, cls) <- zip [0 ..] targets
          ]
      gradFn _ = error "nllLoss: unexpected upstream gradient shape"
      backward =
        if reqGrad
          then [BackwardOp preds gradFn]
          else []
  newTensor [1] [lossVal] reqGrad backward
