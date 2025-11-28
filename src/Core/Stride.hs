module Core.Stride (
  isContiguousLayout,
  defaultStride,
  transpose2,
  permute,
  reshape,
  slice,
  select,
  expand,
) where

import Core.Tensor
import Error
import qualified Data.Set as Set
import Data.List (sortOn)
import System.IO.Unsafe (unsafePerformIO)

isContiguousLayout :: Tensor -> Bool
isContiguousLayout _ = True

defaultStride :: [Int] -> [Int]
defaultStride [] = []
defaultStride dims = tail (scanr (*) 1 dims)

transpose2 :: Tensor -> Int -> Int -> Either TensorError Tensor
transpose2 t ax1 ax2 =
  if length (shape t) /= 2 || not ((ax1, ax2) == (0, 1) || (ax1, ax2) == (1, 0))
    then Left (NotImplemented "transpose2 only supports swapping 0 and 1 for rank-2 tensors")
    else permute t [1, 0]

permute :: Tensor -> [Int] -> Either TensorError Tensor
permute t order =
  let rank = length (shape t)
      valid =
        length order == rank
          && Set.fromList order == Set.fromList [0 .. rank - 1]
   in if not valid
        then Left (ShapeMismatch ("Invalid permutation " <> show order <> " for shape " <> show (shape t)))
        else
          let newShape = map (shape t !!) order
              oldStrides = defaultStride (shape t)
              newStrides = defaultStride newShape
              allIndices = [0 .. numel newShape - 1]
              toCoords idx strides dims =
                if null dims
                  then []
                  else
                    let (q, r) = idx `divMod` head strides
                     in q : toCoords r (tail strides) (tail dims)
              coordsToLinear coords strides = sum (zipWith (*) coords strides)
              newValues =
                [ let newCoords = toCoords i newStrides newShape
                      oldCoords = map (newCoords !!) (inverse order)
                      oldIdx = coordsToLinear oldCoords oldStrides
                   in values t !! oldIdx
                | i <- allIndices
                ]
           in Right (unsafeNewTensor newShape newValues (requiresGrad t) t)
  where
    inverse xs = map snd (sortOn fst (zip xs [0 ..]))

reshape :: Tensor -> [Int] -> Either TensorError Tensor
reshape t newShape =
  if numel newShape /= numel (shape t)
    then Left (ShapeMismatch "reshape: element count mismatch")
    else Right (unsafeNewTensor newShape (values t) (requiresGrad t) t)

slice :: Tensor -> [(Int, Int, Int)] -> Either TensorError Tensor
slice _ _ = Left (NotImplemented "slice is not implemented in the minimal lab build")

select :: Tensor -> Int -> Int -> Either TensorError Tensor
select _ _ _ = Left (NotImplemented "select is not implemented in the minimal lab build")

expand :: Tensor -> [Int] -> Either TensorError Tensor
expand _ _ = Left (NotImplemented "expand is not implemented in the minimal lab build")

unsafeNewTensor :: [Int] -> [Double] -> Bool -> Tensor -> Tensor
unsafeNewTensor dims vals reqGrad parent =
  unsafePerformIO $
    newTensor
      dims
      vals
      reqGrad
      ([BackwardOp parent id | reqGrad])
